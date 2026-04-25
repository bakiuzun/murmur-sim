import torch
import torch.nn as nn
from torch.distributions import Normal
from structs.types import ModelSpec

class ActorCritic(nn.Module):

    def __init__(self, 
                 obs_dim: int,
                 action_dim: int,
                 actor_spec: ModelSpec,
                 critic_spec: ModelSpec):
        
        super().__init__()
        
        self.actor = nn.Sequential(*self._build_model(in_ch=obs_dim,
                                                      spec=actor_spec))
    
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.critic = nn.Sequential(*self._build_model(in_ch=obs_dim,
                                                       spec=critic_spec))

    def _build_model(self, in_ch: int, spec: ModelSpec):
        blocks = []
        
        for i, size in enumerate(spec.hidden_sizes):
            out_ch = size 

            is_last = (i == len(spec.hidden_sizes) - 1) 
            activation = spec.last_activation if is_last else spec.hidden_activation

            blocks.append(nn.Linear(in_ch, out_ch))
                
            if activation is not None:
                # Note: ensure your ModelSpec provides instantiated PyTorch 
                # modules (e.g., nn.ReLU()) rather than functions.
                blocks.append(activation)

            in_ch = out_ch

        return blocks
            
    def forward(self, x: torch.Tensor, action: torch.Tensor = None):
        mean = self.actor(x)
        std  = torch.exp(self.log_std)
        dist = Normal(mean, std)
        
        # if None then we are taking steps 
        if action is None:
            action = dist.sample()
        
        value = self.critic(x).squeeze(-1)

        return action, value, dist.log_prob(action).sum(-1)