
import torch.nn as nn
from models.layers import buildMLP,buildConv,MSEDist
from structs.types import MlpSpec,ConvSpec
import torch.nn.functional as F 

# NeuralNets

class ObsNet(nn.Module):

    def __init__(
        self,
        proj_spec: MlpSpec,
        cnn_spec: ConvSpec,
        dim_cnn=96
    ):
        super(ObsNet, self).__init__()

        self.dim_cnn = dim_cnn

        # CNN proj
        self.proj = nn.Sequential(*buildMLP(proj_spec))


        # CNN
        # 4 -> 8 -> 16 -> 32 -> 64
        self.cnn = nn.Sequential(*buildConv(cnn_spec,deconv=True))




    def forward_cnn(self, x):

        # (B, N, D) -> (B, N, C * 4 * 4)
        x = self.proj(x)

        # (B, N, C * 4 * 4) -> (B * N, C, 4 * 4)
        shape = x.shape
        x = x.reshape(-1, 8 * self.dim_cnn, 4, 4)

        # (B * N, C, 4, 4) -> (B * N, 3, 64, 64)
        x = self.cnn(x)

        # Add 0.5 for image scaling [0:1]
        #x = x + 0.5

        # (B * N, 3, 64, 64) -> (B, N, 3, 64, 64)
        x = x.reshape(shape[:-1] + x.shape[1:])

        # Normal Distribution
        obs_dist = MSEDist(x, reinterpreted_batch_ndims=3)

        return obs_dist
    
    
    def forward(self, inputs):

        # Outputs
        outputs = []

        # Forward
        outputs.append(self.forward_cnn(inputs))
        
        if len(outputs) == 1:
            outputs = outputs[0]

        return outputs
