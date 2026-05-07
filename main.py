import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # put this BEFORE importing jax
import os
import itertools
import utils 
import genesis as gs
from algos import ppo 
import torch
from envs import VisionTargetFollowingEnv,WayPointsFollowEnv
from envs import SimpleVisionTargetFollowingEnv

device = gs.cuda if torch.cuda.is_available() else gs.cpu

gs.init(backend=device,logging_level="warning")

lr = 3e-4
total_steps = 1e9
gm = 0.99

reward_presets = {
    
    'best_waypoint_following': {
        'delta_angvel': 0.0002, 
        'delta_linvel': 0.001,
        'yaw_delta': -10.0,
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
        "num_envs": 2048,
        "num_steps": 256,
        "total_timesteps": total_steps,
        "update_epochs": 4,
        'episode_length_s': 15.0,
        "gamma": gm,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "max_grad_norm": 0.5,
        'num_minibatches': 32,
        'target_features_path': 'models/target_features/features.pt',
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

