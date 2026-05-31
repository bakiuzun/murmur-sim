import torch
import torch.nn as nn
import torch.nn.functional as F
from structs import MlpSpec
from ..layers import buildMLP,DreamerV3GRUCell

# TO UPDATE BECAUSE deter size and stoch size are ALREADY in the MlpSpec 
class RSSM(nn.Module):

    def __init__(
            self, 
            sequence_model_spec:  MlpSpec,
            repre_model_spec: MlpSpec,
            dynamic_model_spec: MlpSpec,
            gru_cell_spec: MlpSpec,
            deter_size=4096, 
            stoch_size=32,  
            min_std=0.1,
            learn_initial=True, 
            uniform_mix=0.01, 
            action_clip=1.0, 
        ):
        super(RSSM, self).__init__()

        # Params
        self.stoch_size = stoch_size
        self.deter_size = deter_size
        self.min_std = min_std
        #self.std_fun = std_fun()
        self.uniform_mix = uniform_mix
        self.learn_initial = learn_initial
        self.action_clip = action_clip

        # Sequence Model

        self.mlp_img1 = nn.Sequential(*buildMLP(sequence_model_spec))

        self.gru = DreamerV3GRUCell(gru_cell_spec=gru_cell_spec)

        
        # Representation Model
        self.mlp_img2 = nn.Sequential(*buildMLP(repre_model_spec))


        self.mlp_obs1 = nn.Sequential(*buildMLP(dynamic_model_spec))


        if self.learn_initial:
            self.weight_init = nn.Parameter(torch.zeros(self.deter_size))

    """
    def get_stoch(self, deter):
        
        # MLP Img 2
        mean, std = torch.chunk(self.mlp_img2(deter), chunks=2, dim=-1)

        # Born std to [self.min_std:+inf]
        std = self.std_fun(std) + self.min_std

        # Dist Params
        dist_params = {'mean': mean, 'std': std}
    
        # Get Mode
        stoch = self.get_dist(dist_params).mode()

        return stoch

    def initial(self, batch_size=1, dtype=torch.float32, device="cpu", detach_learned=False):

        initial_state = structs.AttrDict(
                mean=torch.zeros(batch_size, self.stoch_size, dtype=dtype, device=device),
                std=torch.zeros(batch_size, self.stoch_size, dtype=dtype, device=device),
                stoch=torch.zeros(batch_size, self.stoch_size, dtype=dtype, device=device),
                deter=torch.zeros(batch_size, self.deter_size, dtype=dtype, device=device)
        )

        # Learned Initial
        if self.learn_initial:
            initial_state.deter = F.tanh(self.weight_init).repeat(batch_size, 1)
            initial_state.stoch = self.get_stoch(initial_state.deter) 

            # Detach Learned
            if detach_learned:
                initial_state.deter = initial_state.deter.detach()
                initial_state.stoch = initial_state.stoch.detach()

        return initial_state

    def observe(self, embed, prev_actions, is_firsts, prev_state=None):

        # Initial State
        if prev_state is None:
            prev_actions[:, 0] = 0.0
            prev_state = self.initial(batch_size=embed.shape[0], dtype=embed.dtype, device=embed.device)

        posts = {"stoch": [], "deter": [], "mean": [], "std": []}    
        priors = {"stoch": [], "deter": [], "mean": [], "std": []}
        for l in range(embed.shape[1]):
                
            # Forward Model
            post, prior = self(prev_state, prev_actions[:, l], embed[:, l], is_firsts[:, l])

            # Update previous state, Teacher Forcing
            prev_state = post

            # Append to Lists
            for key, value in post.items():
                posts[key].append(value)
            for key, value in prior.items():
                priors[key].append(value)

        # Stack Lists
        posts = {k: torch.stack(v, dim=1) for k, v in posts.items()} # (B, L, D)
        priors = {k: torch.stack(v, dim=1) for k, v in priors.items()} # (B, L, D)

        return posts, priors

    def imagine(self, p_net, prev_state, img_steps=1):

        # Policy
        policy = lambda s: p_net(self.get_feat(s).detach()).rsample()

        # Current state action
        prev_state["action"] = policy(prev_state)

        # Model Recurrent loop with St, At
        img_states = {"stoch": [prev_state["stoch"]], "deter": [prev_state["deter"]], "mean": [prev_state["mean"]], "std": [prev_state["std"]], "action": [prev_state["action"]]}
        for h in range(img_steps):
                
            # Forward Model
            img_state = self.forward_img(prev_state, prev_state["action"])

            # Current state action
            img_state["action"] = policy(img_state)

            # Update previous state
            prev_state = img_state

            # Append to Lists
            for key, value in img_state.items():
                img_states[key].append(value)

        # Stack Lists
        img_states = {k: torch.stack(v, dim=1) for k, v in img_states.items()} # (B, 1+img_steps, D)

        return img_states

    def get_feat(self, state):

        # Flatten stoch size and discrete size
        stoch = state["stoch"]

        return torch.cat([stoch, state["deter"]], dim=-1)
    
    def get_dist(self, state):

        return torch.distributions.Independent(distributions.Normal(loc=state['mean'], scale=state['std']), 1)

    def forward_img(self, prev_state, prev_action):


        # espace latent prev state, prev action qui peut etre imaginer OU le reel 

        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_action *= (self.action_clip / torch.clip(torch.abs(prev_action), min=self.action_clip)).detach()

        # Flatten stoch size and discrete size
        stoch = prev_state["stoch"]

        # MLP Img  
        # we take the stoch part of the previous state 
        # this means prev state contains stoch and deterministic 
        # the stoch + previous action give me something 
        x = self.mlp_img1(torch.concat([stoch, prev_action], dim=-1))

        # Recurrent
        # this something is given to the gru + the deterministic part
        # the x is the representation of the latent after some mlp 
        # so LATENT SPACE + ACTION + THE PREVIOUS HISTORY 
        # THIS GIVE ME THE NEW HISTORY AND changes also the representation of the x 
        x, deter = self.gru(x, prev_state["deter"])

        # MLP Img 2
        mean, std = torch.chunk(self.mlp_img2(x), chunks=2, dim=-1)

        # Born std to [self.min_std:+inf]
        std = self.std_fun(std) + self.min_std

        # Dist Params
        dist_params = {'mean': mean, 'std': std}
    
        # Sample
        # I GET A NEW STHOCHASTIC PART, a latent that changes 
        # this is the next state but the deter part is SAME
        # so forward img take as input a LATENT SPACE with action 
        # and output a new one a new latent 
        stoch = self.get_dist(dist_params).rsample()

        # Return Prior
        return {"stoch": stoch, "deter": deter, **dist_params}
    
    def forward_obs(self, deter, embed):

        # Concat deter and Emb
        emb_h = torch.concat([embed, deter], dim=-1)

        # MLP Obs 1
        mean, std = torch.chunk(self.mlp_obs1(emb_h), chunks=2, dim=-1)

        # Born std to [self.min_std:+inf]
        std = self.std_fun(std) + self.min_std

        # Dist Params
        dist_params = {'mean': mean, 'std': std}

        # Sample
        stoch = self.get_dist(dist_params).rsample()

        return {"stoch": stoch, "deter": deter, **dist_params}

    def forward(self, prev_state, prev_action, embed, is_first):

        assert embed.dim() == 2

        # Clip Action -c:+c
        if self.action_clip > 0.0:
            prev_action *= (self.action_clip / torch.clip(torch.abs(prev_action), min=self.action_clip)).detach()

        # Reset First States and Actions, necessary for traj buffer since some states will be reset mid-sequence
        if is_first.any():

            # Unsqueeze is_first (B, 1)
            is_first = is_first.unsqueeze(dim=-1)

            # Reset first Actions
            prev_action *= (1.0 - is_first)

            # Reset first States
            init_state = self.initial(embed.shape[0], dtype=embed.dtype, device=embed.device)
            for key, value in prev_state.items():
                is_first_r = torch.reshape(is_first, is_first.shape + (1,) * (len(value.shape) - len(is_first.shape)))
                prev_state[key] = value * (1.0 - is_first_r) + init_state[key] * is_first_r

        # Forward Img
        prior = self.forward_img(prev_state, prev_action)

        # Forward Obs
        post = self.forward_obs(prior["deter"], embed)

        # Return post and prior
        return post, prior
    """
