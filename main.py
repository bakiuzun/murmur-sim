import os

from torch._higher_order_ops.hints_wrap import hints_wrapper_dense
from torch.nn.modules import activation

from models.dreamer.obs_net import ObsNet

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # put this BEFORE importing jax
import os
import itertools
import utils 
import genesis as gs
from algos import ppo 
import torch
from envs import VisionTargetFollowingEnv,WayPointsFollowEnv
from envs import SimpleVisionTargetFollowingEnv,DreamerEnv
from envs import DreamerV3ReplayBuffer
from models import PolicyNet,LatentNet,RSSM,RewardNet,ValueNet,DiscountNet
import torch.nn as nn 
from structs.types import ModelSpec,MlpSpec,ConvSpec
import torch.nn.functional as F
from models import DreamerV3



"""
training_dataset = nnet.datasets.replay_buffers.DreamerV3ReplayBuffer(
    batch_size=model.config.batch_size,
    root=callback_path,
    buffer_capacity=model.config.buffer_capacity,
    epoch_length=epoch_length,
    sample_length=model.config.L,
    collate_fn=model.config.collate_fn,
    save_trajectories=save_trajectories
)
model.set_replay_buffer(training_dataset)

evaluation_dataset = nnet.datasets.VoidDataset(num_steps=model.config.eval_epidodes)
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

    

    dreamerEnv = DreamerEnv(config)
    
    dreamerEnv.reset()
    
    while True:
      dreamerEnv.step(torch.zeros(2,4))


    dreamer = DreamerV3(dreamerEnv)
    dreamer.compile() # set optimizer, lr scheduling for each network

    dreamer_replay_buffer = DreamerV3ReplayBuffer(
        batch_size=16,
        root=f"callbacks/DreamerV3/uavenv",
        buffer_capacity=int(1e6),
        epoch_length=12500,
        sample_length=64,
        save_trajectories=True)

    dreamer.set_replay_buffer(dreamer_replay_buffer)

    dreamer.on_train_begin()
    #run_name = f"{reward_name}"
    #utils.init_wandb(config, name=run_name)
    #env = SimpleVisionTargetFollowingEnv(config)
    #ppo = ppo.PPO(env,config,actorSpec=None,criticSpec=None)
    #ppo.train()
    #utils.finish_wandb()




