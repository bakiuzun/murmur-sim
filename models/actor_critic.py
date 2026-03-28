
import jax.numpy as jnp
import distrax
from flax import nnx 
import jax 
from structs.types import ModelSpec



class ActorCritic(nnx.Module):

    def __init__(self,obs_dim: int,
                      action_dim: int,
                      actor_spec: ModelSpec,
                      critic_spec: ModelSpec,
                      rngs = nnx.Rngs(42)):
        
        self.actor = nnx.Sequential(*self._build_model(in_ch=obs_dim,
                                                       spec=actor_spec,
                                                       rngs=rngs
                                                       ))
    
        self.log_std = nnx.Param(jnp.zeros(action_dim))

        self.critic = nnx.Sequential(*self._build_model(in_ch=obs_dim,
                                                        spec=critic_spec,
                                                        rngs=rngs))

    def _build_model(self,in_ch: int,spec:ModelSpec,rngs: nnx.Rngs):
        blocks = []
        
        for i,size in enumerate(spec.hidden_sizes):
            out_ch = size 

            is_last = (i == len(spec.hidden_sizes) - 1) 
            activation = spec.last_activation if is_last else spec.hidden_activation

            blocks.append(nnx.Linear(in_ch,out_ch,rngs=rngs))
                
            if activation is not None:blocks.append(activation)

            in_ch = out_ch

        return blocks
            

    def __call__(self,x: jnp.array,rng: jax.random.PRNGKey = None,action: jnp.array = None):
        mean = self.actor(x)
        std  = jnp.exp(self.log_std)
        dist = distrax.Normal(mean,std)
        
        # if None then we are taking steps 
        if action is None:
            action = dist.sample(seed=rng)
        
        value = self.critic(x).squeeze(-1)

        return action,value,dist.log_prob(action).sum(-1) 
