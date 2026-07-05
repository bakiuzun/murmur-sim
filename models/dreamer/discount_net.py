import torch.nn as nn 
import torch 
from structs.types import MlpSpec
from ..layers import buildMLP

class DiscountNet(nn.Module):

    def __init__(
        self,
        mlp_spec: MlpSpec,
        linear_proj_spec: MlpSpec,
    ):
        super(DiscountNet, self).__init__()


        self.mlp = nn.Sequential(*buildMLP(mlp_spec))
        self.linear_proj = nn.Sequential(*buildMLP(linear_proj_spec))


    def forward(self, x):

        # MLP Layers
        x = self.mlp(x)

        # Output Proj
        x = self.linear_proj(x)

        # Normal Distribution
        value_dist = torch.distributions.Independent(torch.distributions.Bernoulli(logits=x), 1)

        return value_dist
    
