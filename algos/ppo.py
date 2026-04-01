import jax 
import jax.numpy as jnp 
from envs import UAVEnv
from flax import nnx 
import optax 
from models import ActorCritic 
import utils 
from structs import Transition,ModelSpec,TrainState,EnvState
from eval import metrics

def rollout(graphdef,
            train_state: TrainState,
            env_state: EnvState,
            env: UAVEnv,
            num_steps: int,
            rng):
    
    model = nnx.merge(graphdef,train_state.params,train_state.non_params)
    
    vmapped_step = jax.vmap(env.step,in_axes=(0,0))
    vmapped_randomize = jax.vmap(env.randomize,in_axes=(0,0))

    total_envs = len(env_state.obs)
    keyss = jax.random.split(rng,total_envs)
        
    base_mjx_data = env.base_reset()
    batched_base_mjx_data = jax.tree.map(lambda x: jnp.stack([x] * len(env_state.obs)), base_mjx_data)
    reset_mjx_data,reset_obs,DR_dict = vmapped_randomize(batched_base_mjx_data,keyss)

    reset_tau = DR_dict['motor_tau']
    reset_thrust_coeff = DR_dict['thrust_coeff']
    

    def auto_reset(done, current_state, reset_state):
        return jax.tree.map(
            lambda c, r: jnp.where(done.reshape((-1,) + (1,) * (c.ndim - 1)), r, c),
            current_state, reset_state
    )
    def _step(carry,_):
        current_env_state, rng, = carry
        rng,k1 = jax.random.split(rng,2)
        action,value,log_prob = model(current_env_state.obs,k1)
                
        new_env_state,reward,terminated,truncated = vmapped_step(current_env_state,
                                                  action)
        
        # either 1 or 0
        done = jnp.maximum(terminated,truncated)

        new_obs = auto_reset(done, 
                             new_env_state.obs, 
                             reset_obs)

        new_previous_obs = auto_reset(done,
                                      new_env_state.prev_obs,
                                      reset_obs)
        
        new_mjx_data = auto_reset(done,
                                   new_env_state.mjx_data,
                                   reset_mjx_data)
        
        new_previous_act = auto_reset(done,
                                      new_env_state.prev_actions,
                                      jnp.zeros_like(new_env_state.prev_actions))


        new_previous_ctrl = auto_reset(done,
                                       new_env_state.prev_ctrls,
                                       jnp.zeros_like(new_env_state.prev_ctrls)) 

        new_tau =  auto_reset(done, 
                              new_env_state.tau,
                              reset_tau)


        new_trust_coeff =  auto_reset(done, 
                              new_env_state.thrust_coeff,
                              reset_thrust_coeff)


        new_internal_step = jnp.where(done, 0.0, new_env_state.internal_step)
        new_success_counter = jnp.where(done,0.0,new_env_state.success_counter)
        
        # we care only about termination 
        # if termination then we cut off the next value
        # in GAE computation
        # next value is present IF it's time limit
        transition = Transition(
                        terminated = terminated,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        obs=current_env_state.obs) # We store the action we did FOR THIS obs 

        returned_env_state = EnvState(
            mjx_data=new_mjx_data,
            obs=new_obs,
            prev_obs=new_previous_obs,
            prev_actions=new_previous_act,
            prev_ctrls=new_previous_ctrl,
            internal_step=new_internal_step,
            success_counter=new_success_counter,
            tau=new_tau,
            thrust_coeff=new_trust_coeff
        )
        return (returned_env_state,rng),transition


    carry,traj_batch = jax.lax.scan(_step,
                                    (env_state,rng), 
                                    None,length=num_steps)

    return carry,traj_batch    

def calculate_gae(graphdef,
                  train_state,
                  next_obs,
                  traj,
                  rng,
                  config):
    def gae_step(carry_gae,transition):
        lastgaelam,next_value = carry_gae
            
        # mask = 0 if the episode has been finished else 1
        mask = 1.0 - transition.terminated

        delta = transition.reward + config['gamma'] * next_value * mask - transition.value

        advantage = delta + config['gamma'] * config['gae_lambda'] * mask *lastgaelam
        return (advantage,transition.value),advantage
    
    model = nnx.merge(graphdef,train_state.params,train_state.non_params)
    _,next_value,_ = model(next_obs,rng)

    initial_gae_state = (jnp.zeros_like(next_value),next_value)
    _,advantage = jax.lax.scan(
        gae_step,
        initial_gae_state,
        traj,
        reverse=True
    )
    
    returns = advantage + traj.value
    advantage = (advantage - advantage.mean()) /  (advantage.std() + 1e-8)
    return advantage,returns


def make_ppo_update(graphdef, tx, config):
    """Creates the PPO update function with mini-batching."""
    clip_eps = config['clip_eps']
    num_epochs = config['update_epochs']
    
    # Calculate batch dimensions
    batch_size = config['num_steps'] * config['num_envs']
    num_minibatches = config['num_minibatches']
    minibatch_size = batch_size // num_minibatches

    def ppo_loss(params: nnx.statelib.State, 
                 non_params: nnx.statelib.State, 
                 mb_traj: Transition, 
                 mb_advantage: jnp.array, 
                 mb_returns: jnp.array):
        model = nnx.merge(graphdef, params, non_params)
        
        _, new_value, new_log_prob = model(mb_traj.obs, action=mb_traj.action)

        ratio = jnp.exp(new_log_prob - mb_traj.log_prob)
        pg_loss1 = -mb_advantage * ratio
        pg_loss2 = -mb_advantage * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((new_value - mb_returns) ** 2).mean()
        
        return pg_loss + 0.5 * v_loss

    def ppo_update(train_state, 
                   traj: Transition, 
                   advantage: jnp.array, 
                   returns: jnp.array,
                   rng: jax.random.PRNGKey):

        def ppo_epoch(carry, _):
            params, non_params, opt_state, epoch_rng = carry
            epoch_rng, subkey = jax.random.split(epoch_rng)
            
            permutation = jax.random.permutation(subkey, batch_size)
            
            def prepare_minibatches(x):
                x_flat = x.reshape((batch_size,) + x.shape[2:])
                x_shuffled = x_flat[permutation]
                return x_shuffled.reshape((num_minibatches, minibatch_size) + x_flat.shape[1:])

            shuffled_traj = jax.tree.map(prepare_minibatches, traj)
            shuffled_advantage = prepare_minibatches(advantage)
            shuffled_returns = prepare_minibatches(returns)

            def minibatch_update(mb_carry, mb_data):
                mb_params, mb_non_params, mb_opt_state = mb_carry
                mb_traj, mb_adv, mb_ret = mb_data
                
                loss, grads = jax.value_and_grad(ppo_loss)(mb_params, mb_non_params, mb_traj, mb_adv, mb_ret)
                updates, new_opt_state = tx.update(grads, mb_opt_state, mb_params)
                new_params = optax.apply_updates(mb_params, updates)
                
                return (new_params, mb_non_params, new_opt_state), loss

            # Scan over the first axis of shuffled data (num_minibatches)
            (new_params, non_params, new_opt_state), mb_losses = jax.lax.scan(
                minibatch_update,
                (params, non_params, opt_state),
                (shuffled_traj, shuffled_advantage, shuffled_returns)
            )
            
            # Return the average loss for this epoch
            return (new_params, non_params, new_opt_state, epoch_rng), mb_losses.mean()

        # 3. Outer loop: Iterate over epochs
        (new_params, non_params, new_opt_state, _), epoch_losses = jax.lax.scan(
            ppo_epoch,
            (train_state.params, train_state.non_params, train_state.opt_state, rng),
            None,
            length=num_epochs
        )
        
        new_train_state = train_state._replace(
            params=new_params,
            non_params=non_params,
            opt_state=new_opt_state
        )   
        return new_train_state, epoch_losses

    return ppo_update

def make_train(config,actorSpec=None,criticSpec=None,ckpt_path=None):
    print("Starting PPO.")
    env = UAVEnv(config) 
    rng = jax.random.PRNGKey(config['base_rng'])
    rng,key1,key2 = jax.random.split(rng,3)
    

    if actorSpec is None:
        actorSpec = ModelSpec(
            hidden_sizes=jnp.array([64,64,env.act_size]),
            hidden_activation=nnx.relu,
            last_activation=config['actor_last_activation']
        )

    if criticSpec is None: 
        criticSpec = ModelSpec(
        hidden_sizes=jnp.array([64,64,1]),
        hidden_activation=nnx.relu,
        last_activation=None
        )

    model = ActorCritic(obs_dim=env.obs_size,
                    action_dim=env.act_size,
                    actor_spec=actorSpec,
                    critic_spec=criticSpec,
                    rngs=nnx.Rngs(key1))

    print("Model Initialized !")
    
    num_steps = config['num_steps']
    num_updates = int(config['total_timesteps'] // num_steps // config['num_envs'])
    total_optimizer_steps = num_updates * config['update_epochs'] * config['num_minibatches']

    lr_schedule = optax.cosine_decay_schedule(
        init_value=config['lr'],
        decay_steps=total_optimizer_steps,
        alpha=0.1  # final lr = 10% of initial
    )
    graphdef,params,non_params = nnx.split(model,nnx.Param,...)
    tx = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adamw(lr_schedule),
    )

    if ckpt_path is not None:model = utils.load_model(f'checkpoints/{ckpt_path}', graphdef)

    
    opt_state = tx.init(params)
    train_state = TrainState(params,non_params,opt_state)
    ppo_update = make_ppo_update(graphdef,tx,config)

    
    @jax.jit
    def full_train(
        train_state, 
        env_state: EnvState,
        rng:jax.random.PRNGKey):


        def _iteration(carry, _):
            train_state,curr_env_state, rng,  = carry
            rng, k1, k2,k3 = jax.random.split(rng, 4)

            # 1. Rollout
            (new_env_state,rng), traj = rollout(graphdef=graphdef,
                                       train_state=train_state, 
                                       env_state=curr_env_state,
                                       env=env, 
                                       num_steps=num_steps, 
                                       rng=k1)
            
            
            # 2. GAE
            advantage, returns = calculate_gae(graphdef,
                                               train_state,
                                               new_env_state.obs,
                                                traj,
                                                k2,config)

            # 3. PPO epochs
            train_state, losses = ppo_update(train_state, traj, advantage, returns,k3)


            x = metrics.compute_metrics(new_env_state.mjx_data,
                                        traj.obs,
                                        traj.action,
                                        returns,
                                        target_height=config['target_height'])
  
            jax.debug.callback(utils.log_metrics,x)

            return (train_state,new_env_state, rng), losses[-1]

        (train_state, _, _), all_losses = jax.lax.scan(
            _iteration,
            (train_state,env_state,rng),
            None,
            length=num_updates
        )

        return train_state, all_losses

    def train():

        keys = jax.random.split(rng,config['num_envs'])
    
        
        mjx_data,obs,intern_step,success_counter,previous_obs,tau,thrust_coeff = env.reset(keys)
        print("First Env Reset Done ")

        env_state = EnvState(mjx_data=mjx_data,
                             obs=obs,
                             prev_obs=previous_obs,
                             prev_actions=jnp.zeros((config['num_envs'],env.act_size)),
                             prev_ctrls=jnp.zeros((config['num_envs'],env.act_size)),
                             internal_step=intern_step,
                             success_counter=success_counter,
                             tau=tau,
                             thrust_coeff=thrust_coeff)
        
        print("Launching the Whole Training!")
        final_train_state, losses = full_train(
            train_state,
            env_state,
            key2,

        )


        print("Training Finished")

        utils.save_model(config['model_save_path'],
                         final_train_state.params,
                         final_train_state.non_params)
        
        print("Model Saved!")

        return "Finished"

    return train

