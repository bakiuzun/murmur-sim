import mujoco
import numpy as np
import jax.numpy as jnp
import jax 
from mujoco import mjx 
from .rewards import compute_reward
from .utils import quat_to_rotmat

class UAVEnv:
    def __init__(self,config, drone_model_path='drone_models/skydio_x2/scene.xml'):
        self.reward_config = config['reward_config']
        
        self.mj_model = mujoco.MjModel.from_xml_path(drone_model_path)
        self.mjx_model = mjx.put_model(self.mj_model)
        #print("OPT TIMESTEP = ",self.mjx_model.opt.timestep)

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
        self.obs_size = rotate_mat_size + accel_size + velocity_size + height_drone_size + gyro_size
        self.act_size = 4 # each motor thrust

        

    def _get_obs(self, mjx_data):
        sd = mjx_data.sensordata

        rotation_matrice = quat_to_rotmat(sd[self._quat_slice])
        return jnp.concatenate([
            rotation_matrice, # [0,8] 8 included
            sd[self._accel_slice], # [9,12] 12 included
            sd[self._gyro_slice], # [12,14] 15 included
            mjx_data.qpos[2:3], # [15] only height
            mjx_data.qvel[0:3] # Z  # [16,18] 18 included
        ])

    
    def reset(self,rng=None):
        """Pas de self mutation — retourne le state."""
        mj_data = mujoco.MjData(self.mj_model)
        mjx_data = mjx.put_data(self.mj_model, mj_data)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)

        obs = self._get_obs(mjx_data)
        previous_obs = obs
        success_counter = 0 # NOT USED!
        _internal_step = jnp.float32(0.0)
        return mjx_data, obs,_internal_step,success_counter,previous_obs  # retourne le state, pas self.xxx = 

    def step(self, mjx_data, actions, _internal_step, success_counter,previous_obs,previous_actions):
        
        
        mjx_data = mjx_data.replace(ctrl=actions * 13.0)

        def substep(i,data):
            return mjx.step(self.mjx_model,data)
        
        mjx_data = jax.lax.fori_loop(0,5,substep,mjx_data)
        
        obs = self._get_obs(mjx_data)


        # if we are at step 0 there is no previous action so it is set to actions
        previous_actions = jnp.where(_internal_step == 0.0,actions,previous_actions)


        reward = compute_reward(obs,
                                previous_obs,
                                actions,
                                previous_actions,
                                self.reward_config)
        previous_obs = obs
        previous_actions = actions
        
        rotation_z = obs[8]  # fixed indexing

        done = jnp.float32(
            (_internal_step >= 6100)
            | (jnp.abs(mjx_data.qpos[0]) >= 10.0)  # x drift
            | (jnp.abs(mjx_data.qpos[1]) >= 10.0)   # y drift
            | (jnp.abs(mjx_data.qpos[2]) >= 15.0)   # z drift
            | (rotation_z < 0.) # from 1 to -1 if it's 0 this means 90 degre -> about to drop we stop
            )          
        

        _internal_step += 1
        return mjx_data, obs, reward, done, _internal_step, success_counter,previous_obs,previous_actions