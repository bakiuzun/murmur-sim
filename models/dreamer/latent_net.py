import torch 
import torch.nn as nn 
from torch.distributions import Normal
from structs.types import ModelSpec,MlpSpec
import torch.nn.functional as F
from ..layers import buildConv


class LatentNet(nn.Module):

    def __init__(self, 
                 latent_spec:MlpSpec, 
    ): 

        super().__init__()
        
        self.latentNet = nn.Sequential(*buildConv(latent_spec))

            
    def forward(self, x: torch.Tensor):
        
        # NOT SURE ABOUT THE LATENT SHAPE THOUGH I THINK IT4S A GRID  
        return self.latentNet(x)


