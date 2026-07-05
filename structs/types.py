from quadrants import i
import torch
from typing import NamedTuple, Callable, Union,List
from dataclasses import dataclass, field

class AttrDict(dict):

  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

  def __getstate__(self):
    return self.copy()
  
  def __setstate__(self, mapping):
    self.update(mapping)



class Transition(NamedTuple):
    terminated: torch.Tensor 
    action: torch.Tensor 
    value: torch.Tensor 
    reward: torch.Tensor 
    log_prob: torch.Tensor 
    obs: torch.Tensor

class ModelSpec(NamedTuple):
    sizes: torch.Tensor 
    hidden_activation: Callable
    last_activation: Union[Callable, None]
    norm: Union[str,None]
    layer: str


@dataclass
class MlpSpec:
    hidden_sizes: list[int]
    activation: str | None = "silu"
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
    


@dataclass 
class InitialRSSM:
    stoch: torch.Tensor  
    mean: torch.Tensor 
    std: torch.Tensor 
    deter: torch.Tensor 


class EnvState(NamedTuple):
    mjx_data: torch.Tensor
    obs: torch.Tensor
    prev_obs: torch.Tensor
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
