import torch.nn as nn 
from structs.types import MlpSpec,ConvSpec
import torch 


NORM_REGISTRY = {
    "layernorm": nn.LayerNorm,
    "rmsnorm": nn.RMSNorm,
    "batchnorm": nn.BatchNorm1d,
    None: None,
}
ACT_REGISTRY = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,

}

# FROM Maxime Burchi: https://github.com/burchim/DreamerV3-PyTorch
# A SLIGHT VARIATION OF THE standart GRUCell from Torch 
class DreamerV3GRUCell(nn.Module):

    def __init__(
            self, 
            gru_cell_spec: MlpSpec
        ):
        super(DreamerV3GRUCell, self).__init__()

        # Weights and biases
        self.linear = nn.Sequential(*buildMLP(gru_cell_spec))

    def forward(self, x, hidden):

        # Forward
        h = self.forward_rnn(x, hidden)

        # New Hidden
        new_hidden = h

        return h, new_hidden

    def forward_rnn(self, input, state):

        # Linear Proj + Norm 
        parts = self.linear(torch.cat([input, state], dim=-1))

        # Chunk
        reset, cand, update = parts.chunk(chunks=3, dim=-1)

        # Apply reset Sigmoid
        reset = torch.sigmoid(reset)

        # Apply cand Tanh
        cand = torch.tanh(reset * cand)

        # Apply update Sigmoid
        update = torch.sigmoid(update - 1)

        # Hidden
        h = update * cand + (1 - update) * state

        return h


def buildMLP(spec: MlpSpec):
    blocks = []
        
    for i in range(len(spec.hidden_sizes) -1):
    
        in_ch,out_ch = spec.hidden_sizes[i],spec.hidden_sizes[i+1]  
        is_last = (i == len(spec.hidden_sizes) - 2)


        activation = spec.last_activation if is_last else spec.activation

        blocks.append(nn.Linear(in_ch, out_ch))
            
        # HANLDE ONLY THE LayerNorm, I give the out_ch but other norms might need this
        if spec.norm is not None and not is_last:
            blocks.append(NORM_REGISTRY[spec.norm](out_ch))

        if activation is not None:
            blocks.append(ACT_REGISTRY[activation]())


    return blocks


def buildConv(spec: ConvSpec):

    blocks = []

    for i in range(len(spec.hidden_sizes)-1):

        in_ch,out_ch = spec.hidden_sizes[i],spec.hidden_sizes[i+1] 
        is_last = (i == len(spec.hidden_sizes) - 2)

        kernel_size = spec.kernel_sizes[i] if isinstance(spec.kernel_sizes,list) else spec.kernel_sizes
        stride = spec.strides[i] if isinstance(spec.strides,list) else spec.strides
        padding = spec.padding[i] if isinstance(spec.padding,list) else spec.padding
        
        activation = spec.last_activation if is_last else spec.activation

        blocks.append(nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding))

        if spec.norm is not None and not is_last:
            blocks.append(NORM_REGISTRY[spec.norm](out_ch))

        if activation is not None:
            blocks.append(ACT_REGISTRY[activation]())


    return blocks 
