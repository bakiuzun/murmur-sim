import os 
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # put this BEFORE importing jax

from algos import ppo
import utils 
import itertools
import os
from flax import nnx 
#rest 4e-4,5e-4
lrs = [3e-4]
total_timesteps = [1e8]
gammas = [0.99]
target_heights = [1.0]

"""
without lin vel penalty it doesnt go up good 
without angel BUT with lin vel it go very good up 
without any it does shitty things 
so we going with lin vel + hover reward 
"""

"""
"baseline": {
        'target_height': 5.0, 
        'delta_angvel': 0.000,
        'delta_linvel': 0.001,
        'delta_prog': 1.0,
        'delta_actions': 0.001,
        'delta_crash': 10.0,
        'delta_hover': 0.01,
        'v_max': 10.0,
        'dt': 0.002 # Mujoco env
}

'baseline_2': {
        'target_height': 5.0, 
        'delta_angvel': 0.000,
        'delta_linvel': 0.001,
        'delta_prog': 1.0,
        'delta_actions': 0.002,
        'delta_lateral_vel': 0.01,
        'delta_crash': 10.0,
        'delta_hover': 0.01,
        'v_max': 10.0,
        'dt': 0.002 # Mujoco env
    },
}
"""

reward_presets = {
    'motorteau0025': {
        'target_height': 5.0, 
        'delta_angvel': 0.001, # not even helping i think
        'delta_linvel': 0.001,
        'delta_prog': 1.0,
        'delta_actions': 0.002,
        'delta_lateral_vel': 0.01,
        'delta_crash': 10.0,
        'delta_hover': 0.01,
        'v_max': 10.0,
        'dt': 0.002 # Mujoco env
    },
}  



DR_config = {
    'randomize_height': True,
    'randomize_quat': True,
    'randomize_linvel': True,
    'randomize_angvel': True,
    'randomize_thrust': False,
    'randomize_motor_constant': False,
    'nominal_thrust': 13.0,
    'thrust_variation': 0.0, # [13 - 0.1*13,13 + 0.1*13]
    'motor_tau': 0.025,
    'motor_tau_variation': 0.0, # [0.3,0.7]
    'quat_angle': 0.3, # 1 rad ~ 50 degree 
    'angvel_val': 0.05,
    'linvel_val': 0.5
}


""" 
"""


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
        'model_save_path': f"checkpoints/{reward_name}_lr{lr}_gm{gm}_steps{total_steps}.pt" 
    }

    run_name = f"{reward_name}_lr{lr}_gm{gm}_steps{total_steps}"
    utils.init_wandb(config, name=run_name)
    ppo.make_train(config,ckpt_path=None)()
    utils.finish_wandb()



