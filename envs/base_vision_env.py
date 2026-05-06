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


class SimpleVisionTargetFollowingEnv(UAVEnv):
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



        self.scene.add_entity(morph=gs.morphs.Plane(),
                              surface=gs.surfaces.Rough(
                                diffuse_texture=gs.textures.ColorTexture(color=(0.35,0.5,0.3))
                              ))

        self.target = self.scene.add_entity(
                morph=gs.morphs.Mesh(
                    file="meshes/sphere.obj",
                    scale=0.15,
                    fixed=True,
                    collision=False,
                ),
                surface=gs.surfaces.Emission(
                color=(1.0, 1.0, 1.0)  
            ),
        )

        self.gs_drone = self.scene.add_entity(
            gs.morphs.Drone(
                file='urdf/drones/cf2x.urdf'
            )
        )


        self.camera = self.scene.add_sensor(
            gs.sensors.BatchRendererCameraOptions(
                res=(224,224),
                pos=(0.0,0.0,-0.1),
                lookat=(0.0,0.0,0.0),
                entity_idx=self.gs_drone.idx,
                fov=90,
                lights=[{
                    "pos": (2.0, 2.0, 5.0),
                    "color": (1.0, 1.0, 1.0),
                    "intensity": 0.3,
                    "directional": True,
                    "castshadow": True,
                }],
            )
        )

        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(20.0,20.0))


        
        self.init_base_obs()

        self.obs_size = self._get_obs().shape[-1]
        self.act_size = 4 # each motor RPM


    def save_multiple_target_img(self):
        import os 
        os.makedirs('target_imgs',exist_ok=True)

        # variate Z 
        def variate(axis=[2]):
            for i in range(-5,5): 
                new_x = self.waypoints[0][0]+i*0.05 if 0 in axis else self.waypoints[0][0]
                new_y = self.waypoints[0][1]+i*0.05 if 1 in axis else self.waypoints[0][1]
                new_z = self.waypoints[0][2]+i*0.2 if 2 in axis else self.waypoints[0][2] + 0.5

                
                self.gs_drone.set_pos(
                    torch.tensor([new_x,
                                  new_y,
                                  new_z]),
                    zero_velocity=True,
                    envs_idx=[0,1,2,3,4]
                )
                
                
                self.scene.step()

                data = self.camera.read()

                rgb = data.rgb

            
                if isinstance(rgb,torch.Tensor):rgb = rgb.cpu().numpy()
                #if rgb.ndim == 4:rgb = rgb[0]

                #bgr = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                
                cv2.imwrite(f"target_imgs/img_{axis}-{i}.jpg",rgb[0])

                print("I = ",i)
                
        variate(axis=[2])

        """
        variate(axis=[1])
        variate(axis=[0])
        variate(axis=[1,2])
        variate(axis=[0,2])
        variate(axis=[0,1,2])
        """
    


    def step(self, actions):
        self.target.set_pos(self.waypoints, zero_velocity=True)
        
        self.scene.step() 


        self.save_multiple_target_img()

        return None


    def compute_reward(self,obs,actions):
        return torch.randn((obs.shape[0]))


    def random_drone_spawn(self, n):
        """Spawn the drone randomly in the arena, above the trees."""
        x = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3
        y = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3 
        z = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3
        z = torch.clamp(z,1.)

        return torch.cat((x, y, z), dim=1)
  
    def _get_obs(self):
        base_obs = super()._get_obs()
        return base_obs
 
    def init_base_obs(self):
        super().init_base_obs()

        init_pos = torch.tensor([0.,0.,1.],device=gs.device)
        
        # this is used for tracking
        self.drone_poses =  self.random_drone_spawn(n=self.num_envs)  
        self.waypoints = self.random_waypoints(n=self.num_envs)
        

    def random_waypoints(self, n):
        x = torch.ones(n,1,device=gs.device, dtype=gs.tc_float) * 0.0  # [0, 4)
        y = torch.ones(n, 1, device=gs.device, dtype=gs.tc_float) * 0.0  # [0, 4)
        z = torch.ones(n, 1, device=gs.device, dtype=gs.tc_float) * 1.0 
        return torch.cat((x, y, z), dim=1)


    def _reset_idx(self,envs_idx):   
        super()._reset_idx(envs_idx)
        new_spans = self.random_drone_spawn(n=len(envs_idx))

        self.drone_poses[envs_idx] = new_spans 

        self.gs_drone.set_pos(new_spans, 
                            zero_velocity=True,
                             envs_idx=envs_idx)

            
        self.waypoints[envs_idx] = self.random_waypoints(n=len(envs_idx))

