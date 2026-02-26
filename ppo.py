import torch
import torch.nn as nn
import jax 
import jax.numpy as jnp 
import env 
import gymnasium as gym 



envs = gym.make_vec('uavenv',num_envs=10,
                    vectorization_mode='sync')


print(envs.action_space)






