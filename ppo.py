import jax 
import jax.numpy as jnp 
from envs import UAVEnv 
from jax import jit 
from jax import random 
from jax import make_jaxpr
from jax import grad 
from typing import Sequence,NamedTuple,Any 
from flax import nnx 
import optax 
import distrax
import os 
import pickle 

import pickle

def save_model(filepath, params, non_params):
    with open(filepath, 'wb') as f:
        pickle.dump({"params": params, "non_params": non_params}, f)

def load_model(filepath, graphdef):
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return nnx.merge(graphdef, data["params"], data["non_params"])


class ActorCritic(nnx.Module):

    def __init__(self,obs_shape,act_shape,rngs:nnx.Rngs):
        
        self.actor = nnx.Sequential(
            nnx.Linear(obs_shape,128,rngs=rngs),
            nnx.relu,
            nnx.Linear(128, 256,rngs=rngs),
            nnx.relu,
            nnx.Linear(256,act_shape,rngs=rngs),
            nnx.sigmoid
        )
        self.log_std = nnx.Param(jnp.zeros(act_shape))

        self.critic = nnx.Sequential(
            nnx.Linear(obs_shape,128,rngs=rngs),
            nnx.relu,
            nnx.Linear(128,1,rngs=rngs)
        )

    def __call__(self,x,rng=None,action=None):
        mean = self.actor(x)
        std  = jnp.exp(self.log_std)
        dist = distrax.Normal(mean,std)
        
        # if None then we are taking steps 
        if action is None:
            action = dist.sample(seed=rng)
        
        value = self.critic(x).squeeze(-1)

        return action,value,dist.log_prob(action).sum(-1) 
        



class Transition(NamedTuple):
    done: jnp.ndarray 
    action: jnp.ndarray 
    value: jnp.ndarray 
    reward: jnp.ndarray 
    log_prob: jnp.ndarray 
    obs: jnp.ndarray


def rollout(graphdef,params,non_params,env,obs,mjx_data,num_steps,rng):
    model = nnx.merge(graphdef,params,non_params)

    vmapped_step = jax.vmap(env.step)
    def _step(carry,_):
        obs,mjx_data,rng = carry
        rng,k1 = jax.random.split(rng,2)

        action,value,log_prob = model(obs,k1)
                
        mjx_data, next_obs,reward,done = vmapped_step(mjx_data,action)
        if isinstance(next_obs,dict):
            next_obs = jnp.concatenate([jnp.array(x) for x in list(next_obs.values())])

        transition = Transition(
                        done = done,
                        action=action,
                        log_prob=log_prob,
                        value=value,
                        reward=reward,
                        obs=obs)

        return (next_obs,mjx_data,rng),transition


    carry,traj_batch = jax.lax.scan(_step,(obs,mjx_data,rng), None,length=num_steps)

    return carry,traj_batch    

@jax.jit
def calculate_gae(graphdef,params,non_params,next_obs,traj,rng):
    def gae_step(carry_gae,transition):
        lastgaelam,next_value = carry_gae
            
        # mask = 0 if the episode has been finished else 1
        mask = 1.0 - transition.done 

        delta = transition.reward + config['gamma'] * next_value * mask - transition.value

        advantage = delta + config['gamma'] * config['gae_lambda'] * mask *lastgaelam
        return (advantage,transition.value),advantage
    
    model = nnx.merge(graphdef,params,non_params)
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

    def ppo_loss(params, non_params, traj, advantage, returns):
        model = nnx.merge(graphdef, params, non_params)
        _, new_value, new_log_prob = model(traj.obs, action=traj.action)
        ratio = jnp.exp(new_log_prob - traj.log_prob)
        pg_loss1 = -advantage * ratio
        pg_loss2 = -advantage * jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()
        return pg_loss + 0.5 * v_loss

    @jax.jit
    def ppo_update(params, non_params, opt_state, traj, advantage, returns):

        def ppo_epoch(carry, _):
            params, non_params, opt_state = carry
            loss, grads = jax.value_and_grad(ppo_loss)(params, non_params, traj, advantage, returns)
            updates, new_opt_state = tx.update(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return (new_params, non_params, new_opt_state), loss

        (params, non_params, opt_state), losses = jax.lax.scan(
            ppo_epoch,
            (params, non_params, opt_state),
            None,
            length=num_epochs
        )
        return params, non_params, opt_state, losses

    return ppo_update

def make_train(config):

    env = UAVEnv() 
    rng = jax.random.PRNGKey(30)
    rng,_rng = jax.random.split(rng)
    
    model = ActorCritic(
        env.obs_size,
        env.act_size,
        nnx.Rngs(rng)
    )
    
    graphdef,params,non_params = nnx.split(model,nnx.Param,...)
    tx = optax.adamw(config['lr'])
    opt_state = tx.init(params)
    num_steps = config['num_steps']
    num_updates = int(config['total_timesteps'] // num_steps // config['num_envs'])
    

    ppo_update = make_ppo_update(graphdef,tx,config['clip_eps'],config['update_epochs'])


    @jax.jit
    def full_train(params, non_params, opt_state, obs,mjx_data, rng):

        def _iteration(carry, _):
            params, non_params, opt_state, obs, rng = carry
            rng, k1, k2 = jax.random.split(rng, 3)

            # 1. Rollout
            carry_roll, traj = rollout(graphdef, params, non_params, env, obs,mjx_data, num_steps, rng=k1)
            next_obs = carry_roll[0]

            # 2. GAE
            advantage, returns = calculate_gae(graphdef, params, non_params, next_obs, traj, k2)

            # 3. PPO epochs
            params, non_params, opt_state, losses = ppo_update(
                params, non_params, opt_state, traj, advantage, returns
            )

            return (params, non_params, opt_state, next_obs, rng), losses[-1]

        (params, non_params, opt_state, obs, _), all_losses = jax.lax.scan(
            _iteration,
            (params, non_params, opt_state, obs, rng),
            None,
            length=num_updates
        )

        return params, non_params, opt_state, all_losses

    def train():
        mjx_data,obs = env.reset()
        if isinstance(obs, dict):
            obs = jnp.concatenate([jnp.array(x) for x in list(obs.values())])
       
        batched_mjx_data = jax.tree.map(
            lambda  x: jnp.stack([x]*config['num_envs']),mjx_data 
        )
        batched_obs = jnp.stack([obs]*config['num_envs'])

        rng = jax.random.PRNGKey(42)
        final_params, final_non_params, final_opt_state, losses = full_train(
            params, non_params, opt_state, batched_obs,batched_mjx_data,rng
        )

        save_model('checkpoints/ppo_uav',final_params,final_non_params)
        return {
            "params": final_params,
            "non_params": final_non_params,
            "opt_state": final_opt_state,
            "losses": losses
        }

    return train


if __name__ == "__main__":
    config = {
        "lr": 3e-4,
        "num_envs": 2048,
        "num_steps": 3,
        "total_timesteps": 5,
        "update_epochs": 1,
        "num_minibatches": 32,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "activation": "tanh",
        "env_name": "hopper",
        "anneal_lr": False,
        "normalize_env": True,
        "debug": True,
        }

    make_train(config)()
    
