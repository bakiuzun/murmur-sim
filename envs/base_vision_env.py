import torch  
import genesis as gs
from models import VisionModule
import cv2
from .base_env import UAVEnv
import numpy as np 


class SimpleVisionTargetFollowingEnv(UAVEnv):
    def __init__(self,config):
        super().__init__(config)

        self.batch_rendering = True if torch.cuda.is_available() else False
        self.rendering_frequency = 4

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
                shadow=False,
                env_separate_rigid=True if not self.batch_rendering else False, 
                segmentation_level="entity"
            ),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=config['show_viewer'],
            renderer=gs.renderers.BatchRenderer() if self.batch_rendering else gs.renderers.Rasterizer()
        )

        """
        self._dummy_cam = self.scene.add_camera(
            res=(64, 64), pos=(0,0,5), lookat=(0,0,0), fov=60, GUI=False
        )
        """


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
       
        
        # pos and lookat is not used WHEN it is attached to a drone!!!!!!!!!
        self.camera = self.scene.add_camera(res=(98,98),
                                         pos=(0.0,0.0,0.0),
                                         lookat=(0.0,0.0,0),
                                         fov=90,GUI=True) 
        

        self.camera.attach(self.gs_drone,offset_T=np.array([[1,0,0,0.],
                                                         [0,1,0,0],
                                                         [0,0,1,0],
                                                         [0,0,0,1]]))
        
        self.scene.build(n_envs=self.num_envs,
                         env_spacing=(0,0))


        self.init_base_obs()

        # Vision Module 
        self.vision_module = VisionModule(img_size=98)
        self.vision_module.load_features(config['target_features_path'])

        self.obs_size = self._get_obs()['obs'].shape[-1]
        self.act_size = 4 # each motor RPM

        self.previous_obs = torch.zeros((self.num_envs,self.obs_size),device=gs.device,dtype=gs.tc_float)
        self.previous_acts = torch.zeros((self.num_envs,self.act_size),device=gs.device,dtype=gs.tc_float)
        
        # DINO OUTPUT SIZE 
        self.cached_features = torch.zeros((self.num_envs,384))
        self.segmentation = torch.zeros((self.num_envs,98,98))
        self.index_waypoint_segmentation = 1 


    def save_multiple_target_img(self):
        import os 
        os.makedirs('target_imgs',exist_ok=True)

        # variate Z 
        def variate(axis=[2]):
            for i in range(-5,5): 
                new_x = self.waypoints[0][0]+i*0.05 if 0 in axis else self.waypoints[0][0]
                new_y = self.waypoints[0][1]+i*0.05 if 1 in axis else self.waypoints[0][1]
                new_z = self.waypoints[0][2]+abs(i)*0.3 if 2 in axis else self.waypoints[0][2] + 0.5

                
                self.gs_drone.set_pos(
                    torch.tensor([new_x,
                                  new_y,
                                  new_z]),
                    zero_velocity=True,
                    envs_idx=[0]
                )
                
                
                self.scene.step()

                rgb,_,_,_ = self.camera.render(rgb=True)


                cos_sim = self.vision_module.cosine_sim(rgb,compute_features=True)

                print("Cos sim = ",cos_sim)

            
                
                cv2.imwrite(f"target_imgs/img_{axis}-{i}.jpg",rgb[0])

                
        variate(axis=[2])
        variate(axis=[0])
        variate(axis=[1])
        variate(axis=[1,2])
        variate(axis=[0,2])
        variate(axis=[0,1,2])
        

    def step(self, actions):
        
        cliped_actions = torch.clip(actions,-1.0,1.0)
        cliped_actions = torch.zeros_like(cliped_actions)
        target_rpm = (1 + cliped_actions) * 14468.429183500699
        self.gs_drone.set_propellers_rpm(target_rpm)
        self.camera.move_to_attach()

        self.target.set_pos(self.waypoints, zero_velocity=True)        
        self.scene.step() 

        #self.save_multiple_target_img()

        obs = self._get_obs()
        network_obs = obs['obs']

        self.previous_acts = torch.where(self._internal_step == 0,actions,self.previous_acts)
        
        reward,high_cos_sim = self.compute_reward(obs,actions)
 
        self.success_counter = self.success_counter + high_cos_sim.long()

        self.previous_obs = network_obs 
        self.previous_acts = actions

        height = network_obs[:,15]    
        ang_vel  = network_obs[:,12:15]

        ang_vel_mag = torch.linalg.norm(ang_vel, dim=-1)
        SPIN_LIMIT = 360 / 57.2958 
        
        terminated = (height < 0.1) | (ang_vel_mag > SPIN_LIMIT) | (self.success_counter > 200)
        
        truncated = (self._internal_step >= self.max_episode_length).squeeze(-1)

        self._internal_step += 1
        

        return network_obs,reward,terminated,truncated

    def compute_reward(self,obs, actions):
       
        network_obs = obs['obs']
        segmentation = obs['segmentation'].reshape(self.num_envs,-1)

        height = network_obs[:, 15]
        ang_vel = network_obs[:, 12:15]
        lin_vel = network_obs[:, 9:12]
        features = network_obs[:,16:]

        # making sure the yaw is near 0 to avoid spinnign  
        """
        R = obs[:, :9].reshape(-1, 3, 3)
        yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
        yaw_bonus = torch.exp(-torch.square(yaw) / (self.reward_config['yaw_delta'] ** 2))
        """

        segmentation_sum = torch.sum(torch.tensor(segmentation) == self.index_waypoint_segmentation,axis=-1)
 

        # penality IF we take too much time it might be linear or exponential or quadratic 
        
        actions_diff = torch.sum(torch.square(actions - self.previous_acts), dim=-1)
        crash_p = torch.where(height < 0.1, 1, 0)
        
        cos_sim = self.vision_module.cosine_sim(features,compute_features=False)
        
        cos_sim_sup = cos_sim > 0.90 
        reward = (
            + torch.clamp(segmentation_sum / segmentation.shape[1]*0.5,0,1) 
            #+ self.reward_config['delta_yaw'] * yaw_bonus  
            # + self.reward_config['delta_cosim'] * cos_sim 
            - self.reward_config['delta_linvel'] * torch.square(torch.linalg.norm(lin_vel, dim=-1))
            - self.reward_config['delta_actions'] * actions_diff
            - self.reward_config['delta_crash'] * crash_p
            - self.reward_config['delta_angvel'] * torch.square(torch.linalg.norm(ang_vel, dim=-1))
        )

            
        return reward,cos_sim_sup

    def random_drone_spawn(self, n):
        """Spawn the drone randomly in the arena, above the trees."""
        x = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3
        y = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3 
        z = torch.rand((n, 1), device=gs.device, dtype=gs.tc_float) * 3
        z = torch.clamp(z,1.)

        return torch.cat((x, y, z), dim=1)


    def spawn_next_to_waypoints(self):

        x = self.waypoints[:,0:1] * torch.randn((self.num_envs,1)) * 0.5
        y = self.waypoints[:,1:2] * torch.randn((self.num_envs,1)) * 0.5
        z = self.waypoints[:,2:3] * 1.5 
        
        return torch.cat((x,y,z),dim=1)

    def _get_obs(self):
        base_obs = super()._get_obs()['obs']

        should_read_rgb = self._internal_step % self.rendering_frequency == 0
            
        
        if torch.any(should_read_rgb): 
            rgb_pixels,_,segmentation,_ = self.camera.render(rgb=True,segmentation=True)
            dino_features = self.vision_module.get_features(rgb_pixels)
            self.cached_features = dino_features
            self.cached_segmentation = segmentation 
        else:
            dino_features = self.cached_features
            segmentation = self.cached_segmentation
        
        obs = torch.cat((base_obs,dino_features),dim=1)
        

        return {'obs': obs,
                'segmentation': segmentation} 

 
    def init_base_obs(self):
        super().init_base_obs()

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
           
        self.waypoints[envs_idx] = self.random_waypoints(n=len(envs_idx))

        new_spans = self.spawn_next_to_waypoints() 
        print("NEW SANS = ",new_spans)
        self.drone_poses[envs_idx] = new_spans[envs_idx]
        
        self.gs_drone.set_pos(new_spans[envs_idx], 
                            zero_velocity=True,
                             envs_idx=envs_idx)

        

