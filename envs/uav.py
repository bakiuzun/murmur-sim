import numpy as np
import torch  
from .rewards import compute_reward
from . import utils 
import time 
import genesis as gs
from genesis.utils import geom 
import math

class UAVEnv:
    def __init__(self,config):
        self.reward_config = config['reward_config']

        self.num_envs = config['num_envs']
        self.rendered_env_num = min(500,self.num_envs)


        self.obs_size = self._get_obs_size()
        self.act_size = 4 # each motor RPM

        self.dt = config['dt']

        self.max_episode_length = math.ceil(config["episode_length_s"] / self.dt)

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt,substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=60,
                camera_pos=(3.0,0.0,3.0),
                camera_lookat=(0.0,0.0,1.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(self.rendered_env_num))
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=config['show_viewer']
        )

        # first plane 
        self.scene.add_entity(gs.morphs.Plane())

        self.cam = self.scene.add_camera(
                res=(640, 480),
                pos=(3.5, 0.0, 2.5),
                lookat=(0, 0, 0.5),
                fov=30,
                GUI=True, # one open cv per env 
        )

    
        self.gs_drone = self.scene.add_entity(
            gs.morphs.Drone(
                file='urdf/drones/cf2x.urdf'
            )
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

    def _get_obs_size(self):
        rotate_mat_size = 9  
        lin_vel = 3 
        height_drone_size = 1 
        waypoints_dist = 3
        ang_vel = 3 
        return rotate_mat_size + ang_vel + lin_vel + height_drone_size + waypoints_dist
         
    def _get_obs(self):
        
        # all expressed in world frame 
        quats = self.gs_drone.get_quat()
        lin_vel = self.gs_drone.get_vel() 
        ang_vel = self.gs_drone.get_ang()
        
        # for all env we retrieve just the Z axis
        height = self.gs_drone.get_pos()[:,-1:]  

        # (n_envs,3,3)
        rotation_matrices = geom.quat_to_R(quats)
        
        
        dist_to_waypoints = self.waypoints - self.gs_drone.get_pos()
        
        world_frame_states = torch.stack([
            dist_to_waypoints,
            lin_vel,
            ang_vel
        ],dim=-1)

        body_frame_states = rotation_matrices.mT @ world_frame_states
        
        body_frame_waypoints = body_frame_states[...,0]
        body_frame_lin_vel = body_frame_states[..., 1]
        body_frame_ang_vel = body_frame_states[..., 2]

        return torch.cat(
            (rotation_matrices.reshape(-1,9), # [0,1,2,...,7,8] 8 included
            body_frame_lin_vel,
            body_frame_ang_vel,
            body_frame_waypoints,
            height),# waypoints dist [19,20,21] 3  
            dim=1)
   
    def init_base_obs(self):

        init_pos = torch.tensor([0.,0.,1.],device=gs.device)
        init_quat = torch.tensor([1.0,0.0,0.0,0.0],device=gs.device)

        self.success_counter = torch.zeros(self.num_envs)
        self._internal_step =  torch.zeros((self.num_envs,1))
        
        # this is used for reset 
        self.base_pos = torch.ones((self.num_envs,3),device=gs.device,dtype=gs.tc_float) * init_pos 
        self.base_quat = torch.ones((self.num_envs,4),device=gs.device,dtype=gs.tc_float) * init_quat
        self.base_lin_vel = torch.zeros((self.num_envs,3),device=gs.device,dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs,3),device=gs.device,dtype=gs.tc_float)
        
        # this is used for tracking
        self.drone_poses = self.base_pos.clone()
        self.drone_lin_vel = self.base_lin_vel.clone()
        self.drone_ang_vel = self.base_ang_vel.clone()
        

        # Current Env set up this is DEPENDANT ON THE TASK so it's place should change 
        self.waypoints = self.random_waypoints(n=self.num_envs)
        self.previous_obs = torch.zeros((self.num_envs,self.obs_size),device=gs.device,dtype=gs.tc_float)
        self.previous_acts = torch.zeros((self.num_envs,self.act_size),device=gs.device,dtype=gs.tc_float)
    
    def random_waypoints(self, n):
        x = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 4.0  # [0, 4)
        y = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 4.0  # [0, 4)
        z = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3.0 + 1.0  # [1, 4)
        return torch.cat((x, y, z), dim=1)

    def _reset_idx(self,envs_idx):   
        if len(envs_idx) == 0:return 

        self.drone_poses[envs_idx] = self.base_pos[envs_idx]
        self.drone_lin_vel[envs_idx] = self.base_lin_vel[envs_idx]
        self.drone_ang_vel[envs_idx] = self.base_ang_vel[envs_idx]

        self.gs_drone.set_pos(self.base_pos[envs_idx], 
                            zero_velocity=True,
                             envs_idx=envs_idx)

        self.gs_drone.set_quat(self.base_quat[envs_idx], 
                            zero_velocity=True, 
                            envs_idx=envs_idx)

        self.gs_drone.zero_all_dofs_velocity(envs_idx)
            
        self.waypoints[envs_idx] = self.random_waypoints(n=len(envs_idx))
        self.previous_obs[envs_idx] = 0 
        self.previous_acts[envs_idx] = 0
        self._internal_step[envs_idx] = 0 
        self.success_counter[envs_idx] = 0

        
    def reset(self,envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs)
        self._reset_idx(envs_idx)
        
        obs = self._get_obs()
        self.previous_obs[envs_idx] = obs[envs_idx]
        
        return obs[envs_idx]



    def step(self, actions):
        
        cliped_actions = torch.clip(actions,-1.0,1.0)
        
        # hover RPM
        
        target_rpm = (1 + cliped_actions * 0.8) * 14468.429183500699
        
        self.gs_drone.set_propellers_rpm(target_rpm)

        #self.target.set_pos(self.waypoints, zero_velocity=True)
        
        # Genesis step 
        self.scene.step() 


        obs = self._get_obs()

        # if we are at step 0 there is no previous action so it is set to actions
        # maybe we will change its place in the future
        self.previous_acts = torch.where(self._internal_step == 0,actions,self.previous_acts)
        
        reward,touched_waypoints = compute_reward(obs,
                                self.previous_obs,
                                actions,
                                self.previous_acts,
                                self.reward_config)


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
        height = obs[:,-1]    

        terminated = (height < 0.1) 


        truncated = (self._internal_step >= self.max_episode_length).squeeze(-1)

        self._internal_step += 1 

        return obs,reward,terminated,truncated




