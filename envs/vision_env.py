import numpy as np
import torch  
from . import utils 
import time 
import genesis as gs
from genesis.utils import geom 
import math
from models import VisionModule
import cv2
from .base_env import UAVEnv

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

class VisionTargetFollowingEnv(UAVEnv):
    def __init__(self,config):
        super().__init__(config)

        
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
            renderer=gs.renderers.BatchRenderer()
        )

        self._dummy_cam = self.scene.add_camera(
            res=(64, 64), pos=(0,0,5), lookat=(0,0,0), fov=60, GUI=False
        )

        self.scene.add_light(
            pos=(0.0,0.0,1.0),
            dir=(0.0,0.0,-1.0),
            color=(1.0,1.0,1.0),
            intensity=30.0,
            directional=True,
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


        self._build_map()
     
    
        self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.15,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Emission(
                color=(0.7, 0.2, 1.0)  
            ),
        )

        self.gs_drone = self.scene.add_entity(
            gs.morphs.Drone(
                file='urdf/drones/cf2x.urdf'
            )
        )

        # drone camera 
        self.camera = self.scene.add_sensor(
            gs.sensors.BatchRendererCameraOptions(
                res=(224,224),
                pos=(0.0,0.0,-0.05),
                lookat=(0.0,0.0,-1.0),
                entity_idx=self.gs_drone.idx,
                fov=90
            )
        )

        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(20.0,20.0))


        self.vision_module = VisionModule(img_size=224)
        self.vision_module.load_features(config['target_features_path'])

        self.init_base_obs()

        self.obs_size = self._get_obs().shape[-1]
        self.act_size = 4 # each motor RPM

        self.previous_obs = torch.zeros((self.num_envs,self.obs_size),device=gs.device,dtype=gs.tc_float)
        self.previous_acts = torch.zeros((self.num_envs,self.act_size),device=gs.device,dtype=gs.tc_float)
    

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
            
    def _build_map(self):

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

    def _get_obs(self):
        base_obs = super()._get_obs()

        data = self.camera.read()
        rgb = data.rgb


        # (n_envs,256,384)
        features = self.vision_module.get_features(rgb,
                                        block_index=11,
                                        apply_transform=True)
 
        ret = torch.cat(
            (base_obs,
            features.reshape(self.num_envs,-1)),
            dim=1) 

        return ret  
 
    def init_base_obs(self):
        super().init_base_obs()

        init_pos = torch.tensor([0.,0.,1.],device=gs.device)
        
        # this is used for tracking
        self.drone_poses =  self.random_drone_spawn(n=self.num_envs)  
        self.waypoints = self.random_waypoints(n=self.num_envs)
        
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
        super()._reset_idx(envs_idx)
        new_spans = self.random_drone_spawn(n=len(envs_idx))

        self.drone_poses[envs_idx] = new_spans 

        self.gs_drone.set_pos(new_spans, 
                            zero_velocity=True,
                             envs_idx=envs_idx)

            
        self.waypoints[envs_idx] = self.random_waypoints(n=len(envs_idx))
        self.previous_obs[envs_idx] = 0 
        self.previous_acts[envs_idx] = 0

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
        
    def save_current_view(self):
        data = self.camera.read()
        rgb = data.rgb

        if isinstance(rgb,torch.Tensor):
            rgb = rgb.cpu().numpy()

        for i in range(min(self.num_envs,5)):
            bgr = cv2.cvtColor(rgb[i],cv2.COLOR_RGB2BGR)

            #cv2.imshow("FPV Drone",bgr)
            #cv2.waitKey(1)
            cv2.imwrite(f"img_{i}.jpg",bgr)


    def step(self, actions):
        
        cliped_actions = torch.clip(actions,-1.0,1.0)
        
        # hover RPM
        #print("Clippeda actions = ",cliped_actions)
        target_rpm = (1 + cliped_actions * 0.2) * 14468.429183500699
        
        self.gs_drone.set_propellers_rpm(target_rpm)

        self.target.set_pos(self.waypoints, zero_velocity=True)
        
        # Genesis step 
        self.scene.step() 

        self.save_multiple_target_img()
        obs = self._get_obs()

        #self.save_current_view()
        

        # if we are at step 0 there is no previous action so it is set to actions
        # maybe we will change its place in the future
        self.previous_acts = torch.where(self._internal_step == 0,actions,self.previous_acts)
        
        reward,touched_waypoints = compute_reward(obs,actions)

        pixels = obs[:,16:]
        reshaped_pixels = pixels.reshape((self.num_envs,384))
        
        cos_sim = self.vision_module.cosine_sim(reshaped_pixels,
                                                block_index=11,
                                                compute_features=False)

        reward += cos_sim
    
        self.success_counter = self.success_counter + touched_waypoints

        # randomize waypoints 
        before = self.waypoints.clone()
        self.previous_obs = obs 
        self.previous_acts = actions

        rotation_z = obs[:,8]  
        height = obs[:,15]    

        terminated = (height < 0.1) 

        truncated = (self._internal_step >= self.max_episode_length).squeeze(-1)

        self._internal_step += 1 

        return obs,reward,terminated,truncated



    def compute_reward(self,obs,actions):pass

