import jax 
import jax.numpy as jnp 
from envs import UAVEnv
from flax import nnx 
import optax 
from models import ActorCritic 
import utils 
from structs import Transition,ModelSpec,TrainState 
from eval import metrics

def rollout(graphdef,
            train_state,
            env,
            obs,
            mjx_data,
            num_steps,
            rng,
            internal_step,
            success_counter,
            previous_obs,
            previous_act,
            reset_obs,
            reset_mjx_data):
    model = nnx.merge(graphdef,train_state.params,train_state.non_params)
    
    
    vmapped_step = jax.vmap(env.step,in_axes=(0,0,0,0,0,0))

    def auto_reset(done, current_state, reset_state):
        return jax.tree.map(
            lambda c, r: jnp.where(done.reshape((-1,) + (1,) * (c.ndim - 1)), r, c),
            current_state, reset_state
    )
    def _step(carry,_):
        obs,mjx_data,rng,internal_step,success_counter,previous_obs,previous_act = carry
        rng,k1 = jax.random.split(rng,2)

        action,value,log_prob = model(obs,k1)
                
        next_mjx_data, next_obs,reward,done,next_internal_step,next_success_counter,next_previous_obs,next_previous_act = vmapped_step(mjx_data,
                                                                               action,
                                                                               internal_step,
                                                                               success_counter,
                                                                               previous_obs,
                                                                               previous_act)
        
        
        next_obs = auto_reset(done, next_obs, reset_obs)
        next_previous_obs = auto_reset(done,next_previous_obs,reset_obs)
        next_mjx_data = auto_reset(done, next_mjx_data, reset_mjx_data)
        next_internal_step = jnp.where(done, 0.0, next_internal_step)
        next_success_counter = jnp.where(done,0.0,next_success_counter)
        next_previous_act = auto_reset(done,next_previous_act,jnp.zeros_like(next_previous_act))

        transition = Transition(
                        done = done,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        obs=obs)

        return (next_obs,next_mjx_data,rng,next_internal_step,next_success_counter,next_previous_obs,next_previous_act),transition


    carry,traj_batch = jax.lax.scan(_step,
                                    (obs,mjx_data,rng,internal_step,success_counter,previous_obs,previous_act), 
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
        mask = 1.0 - transition.done 

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
        
        # mb_traj arrays are now perfectly 2D: (minibatch_size, features)
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
    
    graphdef,params,non_params = nnx.split(model,nnx.Param,...)
    tx = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adamw(config['lr']),
    )

    if ckpt_path is not None:
        path = 'baseline_hover_lr0.0003_gm0.99_steps100000000.0.pt' 
        model = utils.load_model(f'checkpoints/{path}', graphdef)

    
    opt_state = tx.init(params)

    train_state = TrainState(params,non_params,opt_state)
    
    num_steps = config['num_steps']
    num_updates = int(config['total_timesteps'] // num_steps // config['num_envs'])
    
    ppo_update = make_ppo_update(graphdef,tx,config)

    
    @jax.jit
    def full_train(
        train_state,
        reset_obs: jnp.array,
        reset_mjx_data:jnp.array, 
        rng:jax.random.PRNGKey,
        internal_step: jnp.array,
        success_counter:jnp.array,
        previous_obs: jnp.array,
        previous_act: jnp.array):

        def _iteration(carry, _):
            train_state, next_obs, rng,next_mjx_data,internal_step,success_counter,next_previous_obs,next_previous_act  = carry
            rng, k1, k2,k3 = jax.random.split(rng, 4)

            # 1. Rollout
            carry_roll, traj = rollout(graphdef,train_state, env, next_obs,
                                       next_mjx_data, num_steps, rng=k1,internal_step=internal_step,
                                       reset_obs=reset_obs,reset_mjx_data=reset_mjx_data,
                                       success_counter=success_counter,
                                       previous_obs=next_previous_obs,
                                       previous_act=next_previous_act)
            next_obs = carry_roll[0]
            next_mjx_data = carry_roll[1]
            internal_step = carry_roll[-4]
            success_counter = carry_roll[-3]
            next_previous_obs = carry_roll[-2]
            next_previous_act = carry_roll[-1]
            
            # 2. GAE
            advantage, returns = calculate_gae(graphdef,train_state, next_obs, traj, k2,config)

            # 3. PPO epochs
            train_state, losses = ppo_update(train_state, traj, advantage, returns,k3)


            x = metrics.compute_metrics(next_mjx_data,
                                        traj.obs,
                                        traj.action,
                                        returns,
                                        target_height=config['target_height'])
  
            jax.debug.callback(utils.log_metrics,x)

            return (train_state, next_obs, rng,next_mjx_data,internal_step,success_counter,next_previous_obs,next_previous_act), losses[-1]

        (train_state, _, _,_,_,_,_,_), all_losses = jax.lax.scan(
            _iteration,
            (train_state, reset_obs, rng,reset_mjx_data,internal_step,success_counter,previous_obs,previous_act),
            None,
            length=num_updates
        )

        return train_state, all_losses

    def train():
        mjx_data,obs,intern_step,success_counter,previous_obs = env.reset()
        print("First Env Reset Done ")

        batched_mjx_data = jax.tree.map(
            lambda  x: jnp.stack([x]*config['num_envs']),mjx_data 
        )
        batched_obs = jnp.stack([obs]*config['num_envs'])
        batched_previous_obs = jnp.stack([previous_obs]*config['num_envs'])
        batched_previous_act = jnp.zeros((config['num_envs'],env.act_size))
        batched_internal_step = jnp.stack([intern_step]*config['num_envs'])
        batched_success_counter = jnp.stack([success_counter]*config['num_envs'])

        print("Launching the Whole Training!")
        final_train_state, losses = full_train(
            train_state,
            batched_obs,
            batched_mjx_data,
            key2,
            batched_internal_step,
            batched_success_counter,
            batched_previous_obs,
            batched_previous_act
        )


        print("Training Finished")

        utils.save_model(config['model_save_path'],
                         final_train_state.params,
                         final_train_state.non_params)
        
        print("Model Saved!")

        return "Finished"

    return train



if __name__ == "__main__":
    config = {
        "lr": 3e-4,
        "num_envs": 2048,
        "num_steps": 256,
        "total_timesteps": 2e8,
        "update_epochs": 4,
        'num_minibatches': 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "max_grad_norm": 0.5,
    }
    utils.init_wandb(config)

    make_train(config)()
    