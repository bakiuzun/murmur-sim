import torch 
import torch.nn as nn 
from torch.distributions import Normal
from structs.types import MlpSpec
import torch.nn.functional as F
from ..layers import buildMLP


class PolicyNet(nn.Module):

    def __init__(self, 
                 policy_spec: MlpSpec,
                 min_std:float = 0.1,
                 max_std:float = 1.0):
        
        super().__init__()
        
        self.policy = nn.Sequential(*buildMLP(policy_spec)) 
        self.min_std = min_std
        self.max_std = max_std

            
    def forward(self, x: torch.Tensor):
        
        # NOT SURE ABOUT THE LATENT SHAPE THOUGH I THINK IT4S A GRID  
        mean, std = torch.chunk(self.policy(x), chunks=2, dim=-1)

        mean = F.tanh(mean)

            
        std = (self.max_std - self.min_std) * F.sigmoid(std + 2.0) + self.min_std

        
        return Normal(mean,std)
   



