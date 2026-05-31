from quadrants import i
import torch
from typing import NamedTuple, Callable, Union,List
from dataclasses import dataclass, field


class Transition(NamedTuple):
    terminated: torch.Tensor 
    action: torch.Tensor 
    value: torch.Tensor 
    reward: torch.Tensor 
    log_prob: torch.Tensor 
    obs: torch.Tensor

class ModelSpec(NamedTuple):
    # Note: In PyTorch, hidden_sizes is usually better typed as a tuple or list of ints 
    # (e.g., tuple[int, ...]) rather than a Tensor, but kept as Tensor here for exact translation.
    sizes: torch.Tensor 
    # Activations in PyTorch are typically instances of nn.Module (e.g., nn.ReLU())
    hidden_activation: Callable
    last_activation: Union[Callable, None]
    norm: Union[str,None]
    layer: str


@dataclass
class MlpSpec:
    hidden_sizes: list[int]
    activation: str = "silu"
    norm: str | None = "layernorm"
    last_activation: str | None = None

@dataclass
class ConvSpec:
    hidden_sizes: list[int]
    kernel_sizes: list[int] | int 
    strides: list[int] | int 
    padding: list[int] | int 
    activation: str = "silu"
    norm: str | None = None 
    last_activation: str | None = None 
    



class EnvState(NamedTuple):
    # Note: mjx_data is specific to MuJoCo JAX. If you are fully porting to PyTorch 
    # and standard MuJoCo, this might become a generic object or standard numpy array, 
    # but it is kept as a Tensor here for consistency.
    mjx_data: torch.Tensor
    obs: torch.Tensor
    prev_obs: torch.Tensor
    """
    # prev_actions: used for the reward function we don't save the action in EnvState
    # it is already saved in Transition but we can't compute the previous action
    # from the Transitions tables because it changes after each rollout 
    # so for the next rollout if the state has not been finished prev actions would be 0
    # that's why I kept it here
    """
    prev_actions: torch.Tensor # THIS IS WHAT THE MODEL OUTPUT
    prev_ctrls: torch.Tensor   # THIS IS WHAT WE SENT TO MUJOCO (Domain rando)
    internal_step: torch.Tensor
    success_counter: torch.Tensor
    
    # tau controlling how fast we should go from the actual motor command
    # to the desired output
    tau: float 

    # nominal thrust coeff is 13.0 but we randomize it after each episode
    thrust_coeff: float

    waypoints: torch.Tensor
