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
        'target_height': 5.0, 
        'delta_angvel': 0.0002, 
        'delta_linvel': 0.001,
        'delta_prog': 0.0,
        'yaw_delta': -10.0,
        'delta_yaw': 0.01,
        'delta_actions': 0.0001,
        'action_threshold': 0.0,
        'delta_lateral_vel': 0.00,
        'delta_crash': 1.0,
        'delta_hover': 0.01,
        'delta_closetarget': 10.0,
        'v_max': 10.0,
        'dt': 0.01 # Mujoco env
    },  
}  

for reward_name,reward_config in reward_presets.items():
    config = {
        "lr": lr,
        "num_envs": 5,
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
        'target_height': reward_config['target_height'],
        'actor_last_activation': torch.nn.Tanh(),
        'model_save_path': f"checkpoints/{reward_name}.pt",
        'dt': 0.01,
        'show_viewer': False
    }

    
    
    run_name = f"{reward_name}"
    utils.init_wandb(config, name=run_name)
    env = SimpleVisionTargetFollowingEnv(config)
    ppo = ppo.PPO(env,config,actorSpec=None,criticSpec=None)
    ppo.train()
    utils.finish_wandb()

