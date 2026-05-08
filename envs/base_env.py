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

class UAVEnv(ABC):
    def __init__(self,config):
        self.reward_config = config['reward_config']

        self.num_envs = config['num_envs']
        self.rendered_env_num = min(500,self.num_envs)

        self.dt = config['dt']

        self.max_episode_length = math.ceil(config["episode_length_s"] / self.dt)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt,substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(6.0,0.0,4.0),
                camera_lookat=(0.0,0.0,1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num)),
                ambient_light=(1.0, 1.0, 1.0),
                shadow=False
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=config['show_viewer'],
        )

        self.gs_drone = self.scene.add_entity(
            gs.morphs.Drone(
                file='urdf/drones/cf2x.urdf'
            )
        )

    def _get_obs(self):
        quats = self.gs_drone.get_quat()
        lin_vel = self.gs_drone.get_vel() 
        ang_vel = self.gs_drone.get_ang()
        
        height = self.gs_drone.get_pos()[:,-1:]  

        rotation_matrices = geom.quat_to_R(quats)
        
        world_frame_states = torch.stack([lin_vel,ang_vel],dim=-1)

        
        body_frame_states = rotation_matrices.mT @ world_frame_states
        
        body_frame_lin_vel = body_frame_states[..., 0]
        body_frame_ang_vel = body_frame_states[..., 1]

        obs = torch.cat(
            (rotation_matrices.reshape(-1,9), # [0,1,2,...,7,8] 8 included
            body_frame_lin_vel, #91011
            body_frame_ang_vel,#121314
            height
            ),
            dim=1) 

        return {'obs': obs}  
 
    def init_base_obs(self):

        init_quat = torch.tensor([1.0,0.0,0.0,0.0],device=gs.device)

        self.success_counter = torch.zeros(self.num_envs)
        self._internal_step =  torch.zeros((self.num_envs,1))
        
        # this is used for reset  
        self.base_quat = torch.ones((self.num_envs,4),device=gs.device,dtype=gs.tc_float) * init_quat
        self.base_lin_vel = torch.zeros((self.num_envs,3),device=gs.device,dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs,3),device=gs.device,dtype=gs.tc_float)
          
        self.drone_lin_vel = self.base_lin_vel.clone()
        self.drone_ang_vel = self.base_ang_vel.clone()

        # drone pos to init in subclass 

    def _reset_idx(self,envs_idx):   
        if len(envs_idx) == 0:return 
        
        self.drone_lin_vel[envs_idx] = self.base_lin_vel[envs_idx]
        self.drone_ang_vel[envs_idx] = self.base_ang_vel[envs_idx]

        self.gs_drone.set_quat(self.base_quat[envs_idx], 
                            zero_velocity=True, 
                            envs_idx=envs_idx)

        self.gs_drone.zero_all_dofs_velocity(envs_idx)
            
        self._internal_step[envs_idx] = 0 
        self.success_counter[envs_idx] = 0

        # subclass implement the drone reset pos 

    def reset(self,envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs)

        self._reset_idx(envs_idx)
        
        obs = self._get_obs()['obs']
        
        return obs[envs_idx]

    @abstractmethod
    def step(self, actions):pass



    @abstractmethod
    def compute_reward(self,obs,actions):pass
    