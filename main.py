import os

from torch._higher_order_ops.hints_wrap import hints_wrapper_dense
from torch.nn.modules import activation

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # put this BEFORE importing jax
import os
import itertools
import utils 
import genesis as gs
from algos import ppo 
import torch
from envs import VisionTargetFollowingEnv,WayPointsFollowEnv
from envs import SimpleVisionTargetFollowingEnv
from models import PolicyNet,LatentNet,RSSM
import torch.nn as nn 
from structs.types import ModelSpec,MlpSpec,ConvSpec
import torch.nn.functional as F


policyNet = PolicyNet(
           policy_spec=MlpSpec(hidden_sizes=[32*32+1024,1024,1024,1024,1024,2*4],
                               activation="silu",
                               norm="layernorm",
                               last_activation=None),
           max_std=2.0,
           min_std=1.0)



latentNet = LatentNet(
        latent_spec=ConvSpec(
                hidden_sizes=[48,2*48,4*48,8*48],
                activation="silu",
                kernel_sizes=3,
                padding=1,
                strides=1,
                norm="layernorm",
                last_activation=None)
            )


# stoch size is 32 and we have 4 actions 
stateNet = RSSM(
            sequence_model_spec=MlpSpec(hidden_sizes=[32+4,1024],
                                        activation="silu",
                                        norm='layernorm',
                                        last_activation=None),

            repre_model_spec=MlpSpec(hidden_sizes=[4096,1024,32*2],
                                     activation="silu",
                                     norm="layernorm",
                                     last_activation=None),

            dynamic_model_spec=MlpSpec(hidden_sizes=[4*4*8*96+4096,1024,32*2],
                                       activation="silu",
                                       norm="layernorm",
                                       last_activation=None),

        )


"""
device = gs.cuda if torch.cuda.is_available() else gs.cpu

gs.init(backend=device,logging_level="warning")

lr = 3e-4
total_steps = 1e9
gm = 0.99

reward_presets = {
    
    'baseline': {
        'delta_angvel': 0.0002, 
        'delta_linvel': 0.001,
        'yaw_delta': 0.0,
        'delta_yaw': 0.01,
        'delta_actions': 0.0001,
        'delta_crash': 1.0,
        'delta_cosim': 1.0,
        'dt': 0.01 # Mujoco env
    },  
}  

for reward_name,reward_config in reward_presets.items():
    config = {
        "lr": lr,
        "num_envs": 2,
        "num_steps": 256,
        "total_timesteps": total_steps,
        "update_epochs": 4,
        'episode_length_s': 15.0,
        "gamma": gm,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "max_grad_norm": 0.5,
        'num_minibatches': 32,
        'target_features_path': 'target_features/e_greedy_features.pt',
        "reward_config": reward_config,
        'actor_last_activation': torch.nn.Tanh(),
        'model_save_path': f"checkpoints/{reward_name}.pt",
        'dt': 0.01,
        'show_viewer': True
    }

    
    
    run_name = f"{reward_name}"
    utils.init_wandb(config, name=run_name)
    env = SimpleVisionTargetFollowingEnv(config)
    ppo = ppo.PPO(env,config,actorSpec=None,criticSpec=None)
    ppo.train()
    utils.finish_wandb()

"""
