import mujoco
import numpy as np
import jax.numpy as jnp
import jax 
from mujoco import mjx 
from .rewards import compute_reward
from . import utils 
from structs import EnvState
import domain_randomization as DR
import time 
import mujoco_warp as mjw 
from mujoco.mjx import create_render_context, render,get_rgb


class UAVEnv:
    def __init__(self,config, drone_model_path='drone_models/skydio_x2/scene.xml'):
        self.reward_config = config['reward_config']
        self.DR_config = config['DR_config']
        self.mj_model = mujoco.MjModel.from_xml_path(drone_model_path)
        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjw_model = mjx.put_model(self.mj_model, impl='warp')
        

        # Sensor slices — constants, calculés une fois
        quat_id = self.mj_model.sensor('body_quat').id
        accel_id = self.mj_model.sensor('body_linacc').id
        gyro_id = self.mj_model.sensor('body_gyro').id
        adr = self.mj_model.sensor_adr
        dim = self.mj_model.sensor_dim

        self._quat_slice = slice(adr[quat_id], adr[quat_id] + dim[quat_id])
        self._accel_slice = slice(adr[accel_id], adr[accel_id] + dim[accel_id])
        self._gyro_slice = slice(adr[gyro_id], adr[gyro_id] + dim[gyro_id])

        # quat,accel,gyro
        rotate_mat_size = 9 
        accel_size = 3 
        velocity_size = 3 
        gyro_size = 3
        height_drone_size = 1 
        waypoints_dist = 3
        self.obs_size = rotate_mat_size + accel_size + velocity_size + height_drone_size + gyro_size + waypoints_dist
        self.act_size = 4 # each motor thrust

        self.dt = 0.002


        """
        # CAMERA PART
        
        self.fpv_cam_id = self.mj_model.camera('fpv').id
        self.nworld = config['num_envs']  

        cam_active = [False] * self.mj_model.ncam
        cam_active[self.fpv_cam_id] = True

        self.rc = create_render_context(
            mjm=self.mj_model,
            nworld=self.nworld,
            cam_res=(64, 64),
            render_rgb=cam_active,
            render_depth=[False] * self.mj_model.ncam,
            enabled_geom_groups=[0],
        )
        self.rc_pytree = self.rc.pytree()
        """

    def _get_obs(self, mjx_data,waypoints):
        sd = mjx_data.sensordata

        rotation_matrice = utils.quat_to_rotmat(sd[self._quat_slice])
        dist_to_waypoints = waypoints - mjx_data.qpos[:3]
        body_frame_waypoints = utils.world_to_body(rotation_matrice,dist_to_waypoints)

        
        """
        #CAMERA

        mj_data = mujoco.MjData(self.mj_model)
        mjx_data = mjx.put_data(self.mj_model, mj_data, impl='warp')

        rgb_data,depth_data = render(self.mjw_model,
                                      mjx_data, 
                                      self.rc_pytree)
        
        rgb = get_rgb(self.rc_pytree, 
                      cam_id=self.fpv_cam_id, 
                      rgb_data=rgb_data[0])
        """

        return jnp.concatenate([
            rotation_matrice, # [0,1,2,...,7,8] 8 included
            sd[self._accel_slice], # [9,10,11,12] 12 included
            sd[self._gyro_slice], # [12,13,14] 15 included
            mjx_data.qpos[2:3], # [15] only height
            mjx_data.qvel[0:3], # Z  # [16,17,18] 18 included,
            body_frame_waypoints,# waypoints dist [19,20,21] 3 
            #rgb.reshape(-1)
        ])

    def base_reset(self):
        mj_data = mujoco.MjData(self.mj_model)
        mjx_data = mjx.put_data(self.mj_model, mj_data)        
        return mjx_data
    
    def randomize(self,base_mjx_data,rng):
        
        
        DR_dict  = DR.randomize(base_mjx_data,self.DR_config,rng)

        mjx_data = DR_dict['mjx_data']
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        
        waypoints = DR_dict['waypoints']

        obs = self._get_obs(mjx_data,waypoints)
        
        return mjx_data,obs,DR_dict
        
    def reset(self,rng):    
        """
        # This function is vmapped so We need to separate the Python stuff out 
        # mjx_data basically come directly from base_reset
        RNG is a BATCH of keys thoughh equal to the length of the env
        """
        mjx_data = self.base_reset()
        def _reset(rng):

            DR_dict = DR.randomize(mjx_data,self.DR_config,rng)
            data = DR_dict['mjx_data']
            data = mjx.forward(self.mjx_model, data)
            
            waypoints = DR_dict['waypoints']
            
            obs = self._get_obs(data,waypoints)

            previous_obs = obs
            success_counter = 0 # NOT USED!
            _internal_step = jnp.float32(0.0)

            tau = DR_dict['motor_tau']
            thrust_coeff = DR_dict['thrust_coeff'] 
            waypoints = DR_dict['waypoints']

            return data, obs,_internal_step,success_counter,previous_obs,tau,thrust_coeff,waypoints  # retourne le state, pas self.xxx = 

        return jax.vmap(_reset,in_axes=0)(rng)
    
    def step(self, env_state: EnvState, actions: jnp.ndarray):
        """
        The reward is computed in SIMULATED output
        The executed one is noisy and we use target ctrl with previous ctrl
        because a motor can't go full instantly so we add the delay
        """
        mjx_data = env_state.mjx_data
        _internal_step = env_state.internal_step
        success_counter = env_state.success_counter
        previous_obs = env_state.prev_obs
        previous_actions = env_state.prev_actions
        previous_ctrls = env_state.prev_ctrls 
        tau = env_state.tau
        thrust_coeff = env_state.thrust_coeff
        waypoints = env_state.waypoints

        # 0.002 / (0.002+0.025)
        # 0.002 / (0027 )
        alpha = self.dt / (tau + self.dt) 
        target_ctrl = actions*thrust_coeff

        
        def substep(carry,_):
            data,current_ctrl = carry

            current_ctrl = alpha * target_ctrl + (1 - alpha) * current_ctrl

            data = data.replace(ctrl=current_ctrl)

            data = mjx.step(self.mjx_model,data)
            return (data,current_ctrl), None 
        
        (mjx_data,actual_ctrl),_ = jax.lax.scan(
            substep,
            (mjx_data,previous_ctrls),
            None,
            length=5 # 100 Hz when dt = 0.002
        )
        obs = self._get_obs(mjx_data,waypoints)

        # if we are at step 0 there is no previous action so it is set to actions
        # maybe we will change its place in the future
        previous_actions = jnp.where(_internal_step == 0.0,actions,previous_actions)

        reward,touched_waypoints = compute_reward(obs,
                                previous_obs,
                                actions,
                                previous_actions,
                                self.reward_config)


        new_success_counter = success_counter + touched_waypoints.astype(jnp.float32)
        
        # maybe should be changed after thoughh with a key in the state OR in uavenv
        # virtually impossible to have two time the same... 
        
        key = jax.random.fold_in(jax.random.PRNGKey(0),
                                 success_counter + _internal_step)
        
        
        # randomize waypoints 
        waypoints = jnp.where(touched_waypoints,
                              DR.randomize_waypoints(key,
                                                     z=(1,5)),
                              waypoints)
        


        # we need to recompute...... cuz maybe the waypoints changed 
        obs = self._get_obs(mjx_data, waypoints)
        
        rotation_z = obs[8]  # fixed indexing

        terminated = jnp.float32(
            (mjx_data.qpos[2] < 0.1)   # z drift
            | (rotation_z < 0.0) # 90 degree 
            )   

        truncated =  (_internal_step >= 6100)       
        

        new_env_state = EnvState(
            mjx_data=mjx_data,
            obs=obs,
            prev_obs=obs,          # we do this because obs will get updated before computing the reward
            prev_actions=actions,            
            internal_step=_internal_step + 1,
            success_counter=new_success_counter,
            prev_ctrls=actual_ctrl,
            tau=tau,
            thrust_coeff=thrust_coeff,
            waypoints=waypoints
        )

        return new_env_state,reward,terminated,truncated