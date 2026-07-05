from torch.autograd import forward_ad
import torch.nn as nn 
from structs.types import MlpSpec,ConvSpec
import torch 
import torch.nn.functional as F 

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

# FROM Maxime Burchi 
class Sigmoid2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,x):
        # if x = 0 -> this returns 1.
        # x
        return 2 * F.sigmoid(x / 2)

class MSEDist:
    
    def __init__(self, mode, agg="sum", reinterpreted_batch_ndims=0):
        self._mode = mode
        self._agg = agg
        self.reduce_dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])

    def mode(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(dim=self.reduce_dims)
        elif self._agg == "sum":
            loss = distance.sum(dim=self.reduce_dims)
        else:
            raise NotImplementedError(self._agg)
        return - loss




class SymLogDiscreteDist:

    def __init__(self, 
                 logits, 
                 reinterpreted_batch_ndims=1, 
                 low=-20, high=20):

        self.logits = logits
        self.reinterpreted_batch_ndims = reinterpreted_batch_ndims
        self.probs = logits.softmax(dim=-1)
        self.reduce_dims = tuple([-x for x in range(1, reinterpreted_batch_ndims + 1)])
        self.bins = torch.linspace(low, high, steps=logits.shape[-1], device=logits.device, dtype=logits.dtype)
    


    def sym_log(self,x):
        return torch.sign(x) * torch.log(torch.abs(x) + 1.0)
    
    def sym_exp(self,x):
        return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)
    
    def mode(self):
        return self.sym_exp(torch.sum(self.probs * self.bins, dim=-1, keepdim=True))
    
    def mean(self):
        return self.sym_exp(torch.sum(self.probs * self.bins, dim=-1, keepdim=True))
    
    def log_prob(self, x):

        # sym log target (..., 1)
        x = self.sym_log(x)

        # (..., 1) -> (..., 1, N) -> (..., 1)
        below = torch.sum((self.bins <= x.unsqueeze(dim=-1)).type(torch.int32), dim=-1) - 1
        above = len(self.bins) - torch.sum((self.bins > x.unsqueeze(dim=-1)).type(torch.int32), dim=-1)

        # clip 0:N-1
        below = torch.clip(below, 0, len(self.bins) - 1)
        above = torch.clip(above, 0, len(self.bins) - 1)

        # Equal
        equal = (below == above)

        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        
        # (..., 1)
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total

        # (..., 1) -> # (..., 1, N)
        target = (F.one_hot(below, num_classes=len(self.bins)) * weight_below.unsqueeze(dim=-1) + F.one_hot(above, len(self.bins)) * weight_above.unsqueeze(dim=-1))

        # Normalize
        log_pred = self.logits - torch.logsumexp(self.logits, dim=-1, keepdims=True)
        
        target = target.squeeze(dim=-2)

        # (..., N) -> (...)
        return (target * log_pred).sum(dim=-1)



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


def buildConv(spec: ConvSpec,deconv=False):

    blocks = []

    for i in range(len(spec.hidden_sizes)-1):

        in_ch,out_ch = spec.hidden_sizes[i],spec.hidden_sizes[i+1] 
        is_last = (i == len(spec.hidden_sizes) - 2)

        kernel_size = spec.kernel_sizes[i] if isinstance(spec.kernel_sizes,list) else spec.kernel_sizes
        stride = spec.strides[i] if isinstance(spec.strides,list) else spec.strides
        padding = spec.padding[i] if isinstance(spec.padding,list) else spec.padding
        
        activation = spec.last_activation if is_last else spec.activation

        if not deconv:
            layer = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding)
        else:
            layer = nn.ConvTranspose2d(in_ch,out_ch,kernel_size,padding)

        blocks.append(layer)

        if spec.norm is not None and not is_last:
            blocks.append(NORM_REGISTRY[spec.norm](out_ch))

        if activation is not None:
            blocks.append(ACT_REGISTRY[activation]())


    return blocks 
