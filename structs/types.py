
from typing import NamedTuple,Callable
import jax.numpy as jnp
from flax import nnx 

class Transition(NamedTuple):
    terminated: jnp.ndarray 
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


# CURRENTLY NOT USED 
class EnvState(NamedTuple):
    mjx_data: jnp.ndarray
    obs: jnp.ndarray
    prev_obs: jnp.ndarray
    """
    # prev_actions: used for the reward function we don't save the action in EnvState
    # it is already saved in Transition but we can't compute the previous action
    # from the Transitions tables because it changes after each rollout 
    # so for the next rollout if the state has not been finished prev actions would be 0
    # that's why I kept it here
    """
    prev_actions: jnp.ndarray # THIS IS WHAT THE MODEL OUTPUT
    prev_ctrls: jnp.ndarray # THIS IS WHAT WE SENT TO MUJOCO (Domain rando)
    internal_step: jnp.ndarray
    success_counter: jnp.ndarray
    
    # tau controlling how fast we should go from the actual motor command
    # to the desired output
    tau: float 

    # nominal thrust coeff is 13.0 but we randomize it after each episode
    thrust_coeff: float

