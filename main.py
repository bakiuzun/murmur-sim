import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # put this BEFORE importing jax
import os
import itertools
import utils 
import genesis as gs
from algos import ppo 
import torch
from envs import VisionTargetFollowingEnv,WayPointsFollowEnv

gs.init(backend=gs.cuda,logging_level="warning")

lrs = [3e-4]
total_timesteps = [1e9]
gammas = [0.99]

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

DR_config = {
    'randomize_height': False,
    'randomize_quat': False,
    'randomize_linvel': False,
    'randomize_angvel': False,
    'randomize_thrust': False,
    'randomize_motor_constant': False,
    'randomize_waypoints': True,
    'waypoints_x': (-5,5),
    'waypoints_y': (-5,5),
    'waypoints_z': (1,5), 
    'nominal_thrust': 13.0,
    'thrust_variation': 0.0, 
    'motor_tau': 0.0,
    'motor_tau_variation': 0.0, 
    'quat_angle': 0.3, 
    'angvel_val': 0.05,
    'linvel_val': 0.5
}


for lr, total_steps, gm, (reward_name, reward_config) in itertools.product(
    lrs, total_timesteps, gammas, reward_presets.items()
):

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
        'target_height': reward_config['target_height'],
        'actor_last_activation': torch.nn.Tanh(),
        'model_save_path': f"checkpoints/{reward_name}.pt",
        'dt': 0.01,
        'show_viewer': False
    }

    
    
    run_name = f"{reward_name}"
    utils.init_wandb(config, name=run_name)
    env = WayPointsFollowEnv(config)
    ppo = ppo.PPO(env,config,actorSpec=None,criticSpec=None)
    ppo.train()
    utils.finish_wandb()

