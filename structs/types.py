
from typing import NamedTuple,Callable
import jax.numpy as jnp
from flax import nnx 

class Transition(NamedTuple):
    done: jnp.ndarray 
    action: jnp.ndarray 
    value: jnp.ndarray 
    reward: jnp.ndarray 
    log_prob: jnp.ndarray 
    obs: jnp.ndarray



class ModelSpec(NamedTuple):
    hidden_sizes: jnp.ndarray 
    hidden_activation: Callable
    last_activation: Callable | None 
 


class TrainState(NamedTuple):
    params: nnx.statelib.State
    non_params: nnx.statelib.State
    opt_state: tuple


class EnvState(NamedTuple):
    mjx_data: jnp.ndarray
    obs: jnp.ndarray
    internal_step: jnp.ndarray
    success_counter: jnp.ndarray