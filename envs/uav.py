import numpy as np
import torch  
from .rewards import compute_reward
from . import utils 
import time 
import genesis as gs
from genesis.utils import geom 
import math
import cv2

"""
tensor([[[-1.8312, -1.7946, -0.0040],
[ 2.3690,  2.1237,  1.4010]]])
"""
# Arena constants
CITY_MIN = (-1.83, -1.79, 0.0)
CITY_MAX = ( 2.37,  2.12, 1.40)
MARGIN = 0.0
WALL_H = 1.5
WALL_T = 0.2

ARENA_X_MIN = CITY_MIN[0] - MARGIN
ARENA_X_MAX = CITY_MAX[0] + MARGIN
ARENA_Y_MIN = CITY_MIN[1] - MARGIN
ARENA_Y_MAX = CITY_MAX[1] + MARGIN
ARENA_SX = ARENA_X_MAX - ARENA_X_MIN
ARENA_SY = ARENA_Y_MAX - ARENA_Y_MIN
ARENA_CX = (ARENA_X_MAX + ARENA_X_MIN) / 2
ARENA_CY = (ARENA_Y_MAX + ARENA_Y_MIN) / 2

TREE_PER_ROW = 8
MUSH_PER_ROW = 8
TREE_ROWS = 5
MUSH_ROWS = 5

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
            show_viewer=config['show_viewer']
        )

        self.scene.add_entity(morph=gs.morphs.Box(
                                size=(ARENA_SX,ARENA_SY,0.1),
                                pos=(ARENA_CX,ARENA_CY,-0.05),
                                fixed=True
                              ),
                              surface=gs.surfaces.Rough(
                              diffuse_texture=gs.textures.ColorTexture(color=(0.35,0.5,0.3))
                              ),
                            )


        for size, pos in [
            ((ARENA_SX, WALL_T, WALL_H), (ARENA_CX, ARENA_Y_MAX, WALL_H/2)),  # north
            ((ARENA_SX, WALL_T, WALL_H), (ARENA_CX, ARENA_Y_MIN, WALL_H/2)),  # south
            ((WALL_T, ARENA_SY, WALL_H), (ARENA_X_MAX, ARENA_CY, WALL_H/2)),  # east
            ((WALL_T, ARENA_SY, WALL_H), (ARENA_X_MIN, ARENA_CY, WALL_H/2)),  # west
        ]:
            self.scene.add_entity(
                morph=gs.morphs.Box(size=size, pos=pos, fixed=True, collision=True),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.55, 0.55, 0.65))
                ),
            )

        
        # ---------- Trees: aligned in rows on LEFT half ----------
        self.tree_trunks = []
        self.tree_foliage = []

        tree_xs = torch.linspace(ARENA_X_MIN + 0.3, ARENA_CX - 0.2, TREE_ROWS).tolist()
        tree_ys = torch.linspace(ARENA_Y_MIN + 0.3, ARENA_Y_MAX - 0.3, TREE_PER_ROW).tolist()

        for tx in tree_xs:
            for ty in tree_ys:
                trunk = self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        radius=0.04, height=0.4,
                        pos=(tx, ty, 0.2),
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.35, 0.22, 0.1))
                    ),
                )
                foliage = self.scene.add_entity(
                    morph=gs.morphs.Sphere(
                        radius=0.18,
                        pos=(tx, ty, 0.55),
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.15, 0.55, 0.2))
                    ),
                )
                self.tree_trunks.append(trunk)
                self.tree_foliage.append(foliage)


        # Stocke toutes les positions (x, y) des arbres pour pouvoir les sampler
        self.tree_positions = torch.tensor(
            [[tx, ty] for tx in tree_xs for ty in tree_ys],
            device=gs.device, dtype=gs.tc_float,
        )  # shape: (n_trees, 2)


        # ---------- Mushrooms: aligned in rows on RIGHT half ----------
        self.mushroom_stems = []
        self.mushroom_caps = []

        mush_xs = torch.linspace(ARENA_CX + 0.2, ARENA_X_MAX - 0.3, MUSH_ROWS).tolist()
        mush_ys = torch.linspace(ARENA_Y_MIN + 0.3, ARENA_Y_MAX - 0.3, MUSH_PER_ROW).tolist()

        for mx in mush_xs:
            for my in mush_ys:
                stem = self.scene.add_entity(
                    morph=gs.morphs.Cylinder(
                        radius=0.05, height=0.25,
                        pos=(mx, my, 0.125),
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.95, 0.92, 0.85))
                    ),
                )
                cap = self.scene.add_entity(
                    morph=gs.morphs.Sphere(
                        radius=0.13,
                        pos=(mx, my, 0.30),
                        fixed=True, collision=False,
                    ),
                    surface=gs.surfaces.Rough(
                        diffuse_texture=gs.textures.ColorTexture(color=(0.85, 0.15, 0.15))
                    ),
                )
                self.mushroom_stems.append(stem)
                self.mushroom_caps.append(cap)

     
                
        self.gs_drone = self.scene.add_entity(
            gs.morphs.Drone(
                file='urdf/drones/cf2x.urdf'
            )
        )

        self.camera = self.scene.add_sensor(
            gs.sensors.RasterizerCameraOptions(
                res=(224,224),
                pos=(0.0,0.0,-0.1),
                lookat=(0.0,0.0,0.0),
                entity_idx=self.gs_drone.idx,
                fov=90
            )
        )


        self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.15,
                    fixed=False,
                    collision=False,
                ),
                surface=gs.surfaces.Emission(
                color=(0.7, 0.2, 1.0)  
            ),
        )
        
        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(10.0,10.0))


        self.init_base_obs()
        #self._randomize_obstacles()


    def random_drone_spawn(self, n):
        """Spawn the drone randomly in the arena, above the trees."""
        buf = 0.4
        x = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) \
            * (ARENA_SX - 2 * buf) + (ARENA_X_MIN + buf)
        y = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) \
            * (ARENA_SY - 2 * buf) + (ARENA_Y_MIN + buf)
        # Au-dessus du feuillage (z=0.73 max) et sous le plafond (mur z=1.5)
        z = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 0.3 + 1.0  # entre 1.0 et 1.3
        return torch.cat((x, y, z), dim=1)
        
    def _randomize_obstacles(self):
        """Re-place trees on left half, rocks on right half."""
        buf = 0.3

        for trunk, foliage in zip(self.tree_trunks, self.tree_foliage):
            x = float(torch.rand(1).item()) * (ARENA_CX - buf - (ARENA_X_MIN + buf)) + (ARENA_X_MIN + buf)
            y = float(torch.rand(1).item()) * (ARENA_Y_MAX - buf - (ARENA_Y_MIN + buf)) + (ARENA_Y_MIN + buf)
            trunk_pos = torch.tensor([[x, y, 0.2]], device=gs.device,
                                    dtype=gs.tc_float).expand(self.num_envs, 3)
            foliage_pos = torch.tensor([[x, y, 0.55]], device=gs.device,
                                    dtype=gs.tc_float).expand(self.num_envs, 3)
            trunk.set_pos(trunk_pos, zero_velocity=True)
            foliage.set_pos(foliage_pos, zero_velocity=True)

        for stem, cap in zip(self.mushroom_stems, self.mushroom_caps):
            x = float(torch.rand(1).item()) * (ARENA_X_MAX - buf - (ARENA_CX + buf)) + (ARENA_CX + buf)
            y = float(torch.rand(1).item()) * (ARENA_Y_MAX - buf - (ARENA_Y_MIN + buf)) + (ARENA_Y_MIN + buf)
            stem_pos = torch.tensor([[x, y, 0.125]], device=gs.device,
                                    dtype=gs.tc_float).expand(self.num_envs, 3)
            cap_pos = torch.tensor([[x, y, 0.30]], device=gs.device,
                                dtype=gs.tc_float).expand(self.num_envs, 3)
            stem.set_pos(stem_pos, zero_velocity=True)
            cap.set_pos(cap_pos, zero_velocity=True)

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
        self.drone_poses =   self.random_drone_spawn(n=self.num_envs)   #self.base_pos.clone()
        self.drone_lin_vel = self.base_lin_vel.clone()
        self.drone_ang_vel = self.base_ang_vel.clone()
        

        # Current Env set up this is DEPENDANT ON THE TASK so it's place should change 
        self.waypoints = self.random_waypoints(n=self.num_envs)
        self.previous_obs = torch.zeros((self.num_envs,self.obs_size),device=gs.device,dtype=gs.tc_float)
        self.previous_acts = torch.zeros((self.num_envs,self.act_size),device=gs.device,dtype=gs.tc_float)
    

    """
    def random_waypoints(self, n):
        x = torch.randint(low=0, high=4, size=(n, 1), device=gs.device,dtype=gs.tc_float)
        y = torch.randint(low=0, high=4, size=(n, 1), device=gs.device,dtype=gs.tc_float)
        z = torch.randint(low=1, high=4, size=(n, 1), device=gs.device,dtype=gs.tc_float)
        return torch.cat((x, y, z), dim=1)
    """

    def random_waypoints(self, n):
        n_trees = self.tree_positions.shape[0]
        idx = torch.randint(TREE_PER_ROW, n_trees, (n,), device=gs.device)
        xy = self.tree_positions[idx]  # (n, 2)
        
        angle = torch.rand(n, device=gs.device) * 2 * math.pi
        offset_dist = 0.3
        dx = torch.cos(angle) * offset_dist
        dy = torch.sin(angle) * offset_dist
        
        x = xy[:, 0:1] + dx.unsqueeze(1)
        y = xy[:, 1:2] + dy.unsqueeze(1)
        
        z = torch.full((n, 1), 0.12, device=gs.device, dtype=gs.tc_float)
        
        return torch.cat((x, y, z), dim=1)

    def _reset_idx(self,envs_idx):   
        if len(envs_idx) == 0:return 
        
        new_spans = self.random_drone_spawn(n=len(envs_idx))

        self.drone_poses[envs_idx] = new_spans #  self.base_pos[envs_idx]
        self.drone_lin_vel[envs_idx] = self.base_lin_vel[envs_idx]
        self.drone_ang_vel[envs_idx] = self.base_ang_vel[envs_idx]

        self.gs_drone.set_pos(new_spans, 
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

        self.drone_poses[envs_idx] = torch.tensor([self.waypoints[envs_idx][0][0],
                                                  self.waypoints[envs_idx][0][1],
                                                  self.waypoints[envs_idx][0][2] + 0.5 ]) 

        self.gs_drone.set_pos(self.drone_poses, 
                            zero_velocity=True,
                             envs_idx=envs_idx)

        #self.waypoints[envs_idx] = torch.tensor([1.0,0.0,1.0])



    
    def reset(self,envs_idx=None):
        if envs_idx is None:
            envs_idx = torch.arange(self.num_envs)
        self._reset_idx(envs_idx)
        
        obs = self._get_obs()
        self.previous_obs[envs_idx] = obs[envs_idx]
        
        return obs[envs_idx]


    def save_multiple_target_img(self):
        import os 
        os.makedirs('target_imgs',exist_ok=True)

        # variate Z 
        def variate(axis=[2]):
            for i in range(-5,5): 
                new_x = self.waypoints[0][0]+i*0.05 if 0 in axis else self.waypoints[0][0]
                new_y = self.waypoints[0][1]+i*0.05 if 1 in axis else self.waypoints[0][1]
                new_z = self.waypoints[0][2]+abs(i)*0.2 if 2 in axis else self.waypoints[0][2] + 0.5

                #new_z = 1
                self.gs_drone.set_pos(
                    torch.tensor([new_x,
                                  new_y,
                                  new_z]),
                    zero_velocity=True,
                    envs_idx=[0]
                )

                
                self.scene.step()

                data = self.camera.read()

                rgb = data.rgb

                if isinstance(rgb,torch.Tensor):rgb = rgb.cpu().numpy()
                if rgb.ndim == 4:rgb = rgb[0]

                bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

                cv2.imwrite(f"target_imgs/img_{axis}-{i}.jpg",bgr)

                
        variate(axis=[2])
        variate(axis=[1])
        variate(axis=[0])
        variate(axis=[1,2])
        variate(axis=[0,2])
        variate(axis=[0,1,2])
        

    def step(self, actions):
        
        cliped_actions = torch.clip(actions,-1.0,1.0)
        
        # hover RPM
        
        target_rpm = (1 + cliped_actions * 0.8) * 14468.429183500699
        
        self.gs_drone.set_propellers_rpm(target_rpm)

        self.target.set_pos(self.waypoints, zero_velocity=True)
        
        # Genesis step 
        self.scene.step() 

        self.save_multiple_target_img()

        """
        data = self.camera.read()
        rgb = data.rgb

        if isinstance(rgb,torch.Tensor):
            rgb = rgb.cpu().numpy()

        if rgb.ndim == 4:
            rgb = rgb[0]


        bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)

        cv2.imshow("FPV Drone",bgr)
        cv2.waitKey(1)

        cv2.imwrite()
        """

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




