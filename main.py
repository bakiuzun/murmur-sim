import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '2'  # put this BEFORE importing jax

from algos import ppo
import utils 
import itertools
import os
from flax import nnx 

lrs = [3e-4]
total_timesteps = [1e9]
gammas = [0.99]
target_heights = [1.0]


reward_presets = {
    
    'test_massreward_angvel': {
        'target_height': 5.0, 
        'delta_angvel': 0.005, # not even helping i think
        'delta_linvel': 0.001, # just some testing... 
        'delta_prog': 10.0,
        'delta_actions': 0.005,
        'action_threshold': 0.0,
        'delta_lateral_vel': 0.00,
        'delta_crash': 1,
        'delta_hover': 0.01,
        'delta_closetarget': 10.0,
        'v_max': 10.0,
        'dt': 0.002 # Mujoco env
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
    'thrust_variation': 0.0, # [13 - 0.1*13,13 + 0.1*13]
    'motor_tau': 0.0,
    'motor_tau_variation': 0.0, # [0.3,0.7]
    'quat_angle': 0.3, # 1 rad ~ 50 degree 
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
        "gamma": gm,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "max_grad_norm": 0.5,
        'num_minibatches': 32,
        "reward_config": reward_config,
        'target_height': reward_config['target_height'],
        'DR_config':DR_config,
        'base_rng': 62,
        'actor_last_activation': None,
        'model_save_path': f"checkpoints/{reward_name}.pt" 
    }

    run_name = f"{reward_name}"
    utils.init_wandb(config, name=run_name)
    ppo.make_train(config,ckpt_path=None)()
    utils.finish_wandb()



