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

reward_presets = {
    "lin_hov_action_0000": {
        'target_height': 5.0, 
        'delta_angvel': 0.000,
        'delta_linvel': 0.001,
        'delta_prog': 1.0,
        'delta_actions': 0.000,
        'delta_crash': 10.0,
        'delta_hover': 0.01,
        'v_max': 10.0,
        'dt': 0.002 # Mujoco env
    },
} 

# TRUE -> check if curr dist is < 0.5 of target
# the other -> 1 - prog so when prog is tiny we get reward


ckpt_path = None



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
        'base_rng': 62,
        'actor_last_activation': None,
        'model_save_path': f"checkpoints/{reward_name}_lr{lr}_gm{gm}_steps{total_steps}.pt" 
    }

    run_name = f"{reward_name}_lr{lr}_gm{gm}_steps{total_steps}"
    utils.init_wandb(config, name=run_name)
    ppo.make_train(config,ckpt_path=ckpt_path)()
    utils.finish_wandb()



