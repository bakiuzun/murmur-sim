import numpy as np
import torch 
from . import utils 
import time 
import genesis as gs
from genesis.utils import geom 
import math
from .base_env import UAVEnv

class WayPointsFollowEnv(UAVEnv):
    def __init__(self,config):
        super().__init__(config)

        
        self.scene.add_entity(gs.morphs.Plane())

        self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True, # one open cv per env 
        )

        self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.05,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(1.0, 0.5, 0.5),
                    ),
                ),
        )
        
        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(1.0,1.0))

        # pos,quat,lin_vel,ang_vel

        self.init_base_obs()

        self.obs_size = self._get_obs().shape[-1]
        self.act_size = 4 # each motor RPM

        self.previous_obs = torch.zeros((self.num_envs,self.obs_size),device=gs.device,dtype=gs.tc_float)
        self.previous_acts = torch.zeros((self.num_envs,self.act_size),device=gs.device,dtype=gs.tc_float)


    def _get_obs(self):
        base_obs = super()._get_obs()

        # all expressed in world frame 
        rotation_matrices = base_obs[:,:9]

        rotation_matrices = rotation_matrices.reshape(-1,3,3)


        dist_to_waypoints = self.waypoints - self.gs_drone.get_pos()
        dist_to_waypoints = dist_to_waypoints.unsqueeze(-1)
        

        body_frame_waypoints = rotation_matrices.mT @ dist_to_waypoints
        body_frame_waypoints = body_frame_waypoints[...,0]
        

        return torch.cat(
            (base_obs,
            body_frame_waypoints),# waypoints dist [19,20,21] 3  
            dim=1)
   
    def init_base_obs(self):
        super().init_base_obs()
        
        init_pos = torch.tensor([0.,0.,1.],device=gs.device)

        self.base_pos = torch.ones((self.num_envs,3),device=gs.device,dtype=gs.tc_float) * init_pos 
        
        self.drone_poses = self.base_pos.clone()
        
        self.waypoints = self.random_waypoints(n=self.num_envs)
        

    def random_waypoints(self, n):
        x = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 4.0  # [0, 4)
        y = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 4.0  # [0, 4)
        z = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3.0 + 1.0  # [1, 4)
        return torch.cat((x, y, z), dim=1)

    def _reset_idx(self,envs_idx):   
        super()._reset_idx(envs_idx)

        self.drone_poses[envs_idx] = self.base_pos[envs_idx]
        
        self.gs_drone.set_pos(self.base_pos[envs_idx], 
                            zero_velocity=True,
                             envs_idx=envs_idx)
    
        self.waypoints[envs_idx] = self.random_waypoints(n=len(envs_idx))
        self.previous_obs[envs_idx] = 0 
        self.previous_acts[envs_idx] = 0
        
        
    def reset(self,envs_idx=None):
        obs = super().reset(envs_idx)

        self.previous_obs[envs_idx] = obs
        
        return obs


    def step(self, actions):
        
        cliped_actions = torch.clip(actions,-1.0,1.0)
        
        # hover RPM
        target_rpm = (1 + cliped_actions * 0.8) * 14468.429183500699
        
        self.gs_drone.set_propellers_rpm(target_rpm)

        #self.target.set_pos(self.waypoints, zero_velocity=True)
        
        # Genesis step 


        self.scene.step() 

        self.save_imgs()



        obs = self._get_obs()

        # if we are at step 0 there is no previous action so it is set to actions
        # maybe we will change its place in the future
        self.previous_acts = torch.where(self._internal_step == 0,actions,self.previous_acts)
        
        reward,touched_waypoints = self.compute_reward(obs,actions)


        self.success_counter = self.success_counter + touched_waypoints

        # randomize waypoints 
        before = self.waypoints.clone()

        self.waypoints = torch.where(
                              touched_waypoints.bool().unsqueeze(1),
                              self.random_waypoints(self.num_envs),
                              self.waypoints)
        
        
        # we need to recompute...... cuz maybe the waypoints changed 
        obs = self._get_obs()
        
        self.previous_obs = obs 
        self.previous_acts = actions

        rotation_z = obs[:,8]  # fixed indexing
        height = obs[:,15]    
        ang_vel  = obs[:,12:15]

        ang_vel_mag = torch.linalg.norm(ang_vel, dim=-1)
        SPIN_LIMIT = 360 / 57.2958  
        terminated = (height < 0.1) | (ang_vel_mag > SPIN_LIMIT)
        #terminated = (height < 0.1) 


        truncated = (self._internal_step >= self.max_episode_length).squeeze(-1)

        self._internal_step += 1 

        return obs,reward,terminated,truncated


    def compute_reward(self,obs, actions):
        height = obs[:, 15]
        ang_vel = obs[:, 12:15]
        lin_vel = obs[:, 9:12]
        curr_waypoints_dist = obs[:, 16:19]
        prev_waypoints_dist = self.previous_obs[:, 16:19]
        
        R = obs[:, :9].reshape(-1, 3, 3)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        yaw_bonus = torch.exp(-torch.square(yaw) / (self.reward_config['yaw_delta'] ** 2))
        
        progression = torch.sum(torch.square(prev_waypoints_dist), dim=-1) \
                    - torch.sum(torch.square(curr_waypoints_dist), dim=-1)
        
        batched_vmax = torch.ones_like(progression) * (self.reward_config['v_max'] * self.reward_config['dt'])
        actions_diff = torch.sum(torch.square(actions - self.previous_acts), dim=-1)
        crash_p = torch.where(height < 0.1, 1, 0)
        dist_scalar = torch.linalg.norm(curr_waypoints_dist, dim=-1)
        close_target_r = torch.where(dist_scalar < 1., 1, 0)
        
        reward = (
            self.reward_config['delta_prog'] * torch.minimum(progression, batched_vmax)
            + self.reward_config['delta_closetarget'] * close_target_r
            + self.reward_config['delta_yaw'] * yaw_bonus                               # ← AJOUT
            - self.reward_config['delta_linvel'] * torch.square(torch.linalg.norm(lin_vel, dim=-1))
            - self.reward_config['delta_actions'] * torch.maximum(actions_diff, torch.zeros_like(actions_diff))
            - self.reward_config['delta_crash'] * crash_p
            - self.reward_config['delta_angvel'] * torch.square(torch.linalg.norm(ang_vel, dim=-1))
        
        )
            
        return reward, close_target_r
