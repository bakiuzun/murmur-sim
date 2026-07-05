import torch 
import torch.nn as nn 
from torch.distributions import Normal
from structs.types import ModelSpec,MlpSpec
import torch.nn.functional as F
from ..layers import buildConv,buildMLP,SymLogDiscreteDist


class RewardNet(nn.Module):

    def __init__(self, 
                 reward_spec:MlpSpec, 
                 linear_proj_spec: MlpSpec
    ): 

        super().__init__()
        
        self.mlp = nn.Sequential(*buildMLP(reward_spec))
        self.linear_proj = nn.Sequential(*buildMLP(linear_proj_spec))


            
    def forward(self, x: torch.Tensor):
        
        # NOT SURE ABOUT THE LATENT SHAPE THOUGH I THINK IT4S A GRID
        x = self.mlp(x)
        x = self.linear_proj(x)


        reward_dist = SymLogDiscreteDist(logits=x,reinterpreted_batch_ndims=1,low=-20,high=20)

        return reward_dist
