import jax 
import jax.numpy as jnp 
from envs import UAVEnv
from flax import nnx 
import optax 
from models import ActorCritic 
import utils 
from structs import Transition,ModelSpec,TrainState,EnvState


def rollout(graphdef,train_state,env,obs,
            mjx_data,num_steps,rng,internal_step,success_counter,
            reset_obs,reset_mjx_data):
    model = nnx.merge(graphdef,train_state.params,train_state.non_params)
    
    
    vmapped_step = jax.vmap(env.step,in_axes=(0,0,0,0))

    def auto_reset(done, current_state, reset_state):
        return jax.tree.map(
            lambda c, r: jnp.where(done.reshape((-1,) + (1,) * (c.ndim - 1)), r, c),
            current_state, reset_state
    )
    def _step(carry,_):
        obs,mjx_data,rng,internal_step,success_counter = carry
        rng,k1 = jax.random.split(rng,2)

        action,value,log_prob = model(obs,k1)
                
        next_mjx_data, next_obs,reward,done,next_internal_step,next_success_counter = vmapped_step(mjx_data,
                                                                               action,
                                                                               internal_step,
                                                                               success_counter)
        
        
        next_obs = auto_reset(done, next_obs, reset_obs)
        next_mjx_data = auto_reset(done, next_mjx_data, reset_mjx_data)
        next_internal_step = jnp.where(done, 0.0, next_internal_step)
        next_success_counter = jnp.where(done,0.0,next_success_counter)

        transition = Transition(
                        done = done,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        obs=obs)

        return (next_obs,next_mjx_data,rng,next_internal_step,next_success_counter),transition


    carry,traj_batch = jax.lax.scan(_step,
                                    (obs,mjx_data,rng,internal_step,success_counter), 
                                    None,length=num_steps)

    return carry,traj_batch    

def calculate_gae(graphdef,
                  train_state,
                  next_obs,
                  traj,
                  rng):
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
    return advantage,returns

def make_ppo_update(graphdef, tx, clip_eps, num_epochs):
    """Créé la fonction PPO update, JIT-able."""
    
    def ppo_loss(params: nnx.statelib.State, 
                 non_params:nnx.statelib.State, 
                 traj:Transition, 
                 advantage:jnp.array, 
                 returns:jnp.array):
        model = nnx.merge(graphdef, params, non_params)
        _, new_value, new_log_prob = model(traj.obs, action=traj.action)
        ratio = jnp.exp(new_log_prob - traj.log_prob)
        pg_loss1 = -advantage * ratio
        pg_loss2 = -advantage * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()
        return pg_loss + 0.5 * v_loss

    def ppo_update(train_state, 
                   traj:Transition, 
                   advantage:jnp.array, 
                   returns:jnp.array):

        def ppo_epoch(carry, _):
            params, non_params, opt_state = carry
            loss, grads = jax.value_and_grad(ppo_loss)(params, non_params, traj, advantage, returns)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, non_params, new_opt_state), loss


        (new_params, non_params, new_opt_state), losses = jax.lax.scan(
            ppo_epoch,
            (train_state.params,train_state.non_params,train_state.opt_state),
            None,
            length=num_epochs
        )
        # UPDATED TRAIN STATE
        new_train_state = train_state._replace(
            params=new_params,
            non_params=non_params,
            opt_state=new_opt_state
        )   
        return new_train_state, losses

    return ppo_update

def make_train(config):

    env = UAVEnv() 
    rng = jax.random.PRNGKey(30)
    rng,key1,key2 = jax.random.split(rng,3)
    
    actorSpec = ModelSpec(
        hidden_sizes=jnp.array([128,256,env.act_size]),
        hidden_activation=nnx.relu,
        last_activation=None
    )
    criticSpec = ModelSpec(
       hidden_sizes=jnp.array([128,256,1]),
       hidden_activation=nnx.relu,
       last_activation=nnx.sigmoid
    )

    model = ActorCritic(obs_dim=env.obs_size,
                    action_dim=env.act_size,
                    actor_spec=actorSpec,
                    critic_spec=criticSpec,
                    rngs=nnx.Rngs(key1))

    
    graphdef,params,non_params = nnx.split(model,nnx.Param,...)
    tx = optax.chain(
            optax.clip_by_global_norm(config['max_grad_norm']),
            optax.adamw(config['lr']),
    )
    
    opt_state = tx.init(params)

    train_state = TrainState(params,non_params,opt_state)
    
    num_steps = config['num_steps']
    num_updates = int(config['total_timesteps'] // num_steps // config['num_envs'])
    
    ppo_update = make_ppo_update(graphdef,tx,config['clip_eps'],config['update_epochs'])

    
    @jax.jit
    def full_train(
        train_state,
        reset_obs: jnp.array,
        reset_mjx_data:jnp.array, 
        rng:jax.random.PRNGKey,
        internal_step: jnp.array,
        success_counter:jnp.array):

        def _iteration(carry, _):
            train_state, next_obs, rng,next_mjx_data,internal_step,success_counter = carry
            rng, k1, k2 = jax.random.split(rng, 3)

            # 1. Rollout
            carry_roll, traj = rollout(graphdef,train_state, env, next_obs,
                                       next_mjx_data, num_steps, rng=k1,internal_step=internal_step,
                                       reset_obs=reset_obs,reset_mjx_data=reset_mjx_data,
                                       success_counter=success_counter)
            next_obs = carry_roll[0]
            next_mjx_data = carry_roll[1]
            internal_step = carry_roll[-2]
            success_counter = carry_roll[-1]
            
            # 2. GAE
            advantage, returns = calculate_gae(graphdef,train_state, next_obs, traj, k2)

            # 3. PPO epochs
            train_state, losses = ppo_update(train_state, traj, advantage, returns)

            return (train_state, next_obs, rng,next_mjx_data,internal_step,success_counter), losses[-1]

        (train_state, _, _,_,_,_), all_losses = jax.lax.scan(
            _iteration,
            (train_state, reset_obs, rng,reset_mjx_data,internal_step,success_counter),
            None,
            length=num_updates
        )

        return train_state, all_losses

    def train():
        mjx_data,obs,intern_step,success_counter = env.reset()
        
        batched_mjx_data = jax.tree.map(
            lambda  x: jnp.stack([x]*config['num_envs']),mjx_data 
        )
        batched_obs = jnp.stack([obs]*config['num_envs'])
        batched_internal_step = jnp.stack([intern_step]*config['num_envs'])
        batched_success_counter = jnp.stack([success_counter]*config['num_envs'])

        final_train_state, losses = full_train(
            train_state,
            batched_obs,
            batched_mjx_data,
            key2,
            batched_internal_step,
            batched_success_counter
        )

        utils.save_model('checkpoints/ppo_uav',final_train_state.params,final_train_state.non_params)

        return "Finished"

    return train


if __name__ == "__main__":
    config = {
        "lr": 3e-4,
        "num_envs": 2048,
        "num_steps": 256,
        "total_timesteps": 1e6,
        "update_epochs": 4,
        "num_minibatches": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
    }

    make_train(config)()
    