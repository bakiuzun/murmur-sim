import numpy as np
import torch  
from . import utils 
import time 
import genesis as gs
from genesis.utils import geom 
import math
from models import VisionModule
import cv2
from abc import ABC,abstractmethod

from structs.types import AttrDict



IDENTITY_QUAT = (1.0, 0.0, 0.0, 0.0)

class UAVEnv(ABC):
    def __init__(self,config):
        self.reward_config = config['reward_config']

        self.num_envs = config['num_envs']
        self.rendered_env_num = min(500,self.num_envs)

        self.dt = config['dt']

        self.max_episode_length = math.ceil(config["episode_length_s"] / self.dt)




    # ---- the single source of truth for "what an observation is" ----
    def _proprio(self):
        """Returns the low-dim state vector only. 
           Drone internal state
        """
        if not hasattr(self,'gs_drone'):
            return AttributeError('Error: gs drone is not defined') 


        quats   = self.gs_drone.get_quat()
        lin_vel = self.gs_drone.get_vel()
        ang_vel = self.gs_drone.get_ang()
        height  = self.gs_drone.get_pos()[:, -1:]

        R = geom.quat_to_R(quats)
        world = torch.stack([lin_vel, ang_vel], dim=-1)
        body  = R.mT @ world
        return torch.cat(
            (R.reshape(-1, 9), 
            body[..., 0], # lin vel
            body[..., 1], # ang vel
            height), dim=1
        )  # (N, 16)
    

    def _obs_dict(self):
        """The full structured observation. Pixels added by subclass override."""
        n = self.num_envs
        obs = AttrDict()
        obs.state = {'proprio': self._proprio()}
        obs.is_first = torch.zeros(n,dtype=torch.bool,device=gs.device) # (N,) bool
        obs.is_last = torch.zeros(n, dtype=torch.bool, device=gs.device)
        obs.is_terminal = torch.zeros(n, dtype=torch.bool, device=gs.device) 
        obs.reward = torch.zeros(n,dtype=torch.float32,device=gs.device) 
        obs.done = torch.zeros(n,dtype=torch.bool,device=gs.device)
        return obs


    def _init_buffers(self):
        """Allocate all persistent per-env state once. On-device, typed."""
        N, dev = self.num_envs, gs.device

        self.internal_step  = torch.zeros(N,1, dtype=torch.long,  device=dev)
        self.success_counter = torch.zeros(N,1, dtype=torch.long,  device=dev)

        self._init_quat = torch.tensor(IDENTITY_QUAT, dtype=gs.tc_float, device=dev)



    def _reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        self.gs_drone.set_quat(
            self._init_quat.expand(n, 4),   
            zero_velocity=True, envs_idx=envs_idx,
            )
        self.gs_drone.zero_all_dofs_velocity(envs_idx)
        self.internal_step[envs_idx]  = 0
        self.success_counter[envs_idx] = 0         


    def reset(self,envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs)

        self._reset_idx(envs_idx)
       
        obs = self._obs_dict()
        
        if envs_idx is None:
            obs['is_first'] = True 
        else:
            obs['is_first'][envs_idx] = True
       
        return obs
