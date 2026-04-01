import jax 
import jax.numpy as jnp 
import mujoco
from mujoco import mjx 



def yo(x):
    return x + 1 


batched = jnp.array([1,2,3,4,5,5,8,9,0])
vmapeed = jax.vmap(yo,in_axes=0)
vmapeed(batched)