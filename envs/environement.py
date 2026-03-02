import mujoco
import mujoco.viewer
import numpy as np
import time 
import jax
import jax.numpy as jnp
from mujoco import mjx 


class UAVEnv:
    def __init__(self, xml_path='skydio_x2/scene.xml'):
        self.mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mjx_model = mjx.put_model(self.mj_model)

        # Sensor slices — constants, calculés une fois
        quat_id = self.mj_model.sensor('body_quat').id
        accel_id = self.mj_model.sensor('body_linacc').id
        gyro_id = self.mj_model.sensor('body_gyro').id
        adr = self.mj_model.sensor_adr
        dim = self.mj_model.sensor_dim

        self._quat_slice = slice(adr[quat_id], adr[quat_id] + dim[quat_id])
        self._accel_slice = slice(adr[accel_id], adr[accel_id] + dim[accel_id])
        self._gyro_slice = slice(adr[gyro_id], adr[gyro_id] + dim[gyro_id])
        

        self.obs_size = 10  # 4 + 3 + 3.0
        
        self.obs_size += 1 # height of the drone 

        self.act_size = 4

    def _get_obs(self, mjx_data):
        sd = mjx_data.sensordata
        return jnp.concatenate([
            sd[self._quat_slice],
            sd[self._accel_slice],
            sd[self._gyro_slice],
            mjx_data.qpos[2]
        ])

    def gaussian_reward(x,target,sigma):
        return jnp.exp(-((x - target)**2) / (sigma**2))


    def _get_reward(self, obs):
        # how fast the drone rotates ? 
        gyro = obs[7:10] 
        # the rotation of the drone 
        quat = obs[0:4] 
        # acceleration du drone 
        accel = obs[4:7]
        
        height = obs[-1]

        r_height = gaussian_reward(height,target=2.0,sigma=0.5)
        #r_hover = gaussian_reward(accel[2],target=9.81,sigma=2.0)
        r_upright = gaussian_reward(quat[0],target=1.0,sigma=0.3)

        r_gyro = (gaussian_reward(gyro[0], 0.0, sigma=1.0)
            + gaussian_reward(gyro[1], 0.0, sigma=1.0)
            + gaussian_reward(gyro[2], 0.0, sigma=1.0)) / 3.0


        reward = 0.4 * r_height + 0.3 * r_upright + 0.3 * r_gyro
        return reward

    # ============ Les 2 seuls changements importants ============

    def reset(self,rng=None):
        """Pas de self mutation — retourne le state."""
        mj_data = mujoco.MjData(self.mj_model)
        mjx_data = mjx.put_data(self.mj_model, mj_data)
        mjx_data = mjx.forward(self.mjx_model, mjx_data)
        obs = self._get_obs(mjx_data)

        return mjx_data, obs  # retourne le state, pas self.xxx = 

    def step(self, mjx_data, action):
        """Pas de self mutation — prend et retourne le state."""
        mjx_data = mjx_data.replace(ctrl=action)
        mjx_data = mjx.step(self.mjx_model, mjx_data)
        obs = self._get_obs(mjx_data)
        reward = self._get_reward(obs)
        done = jnp.float32(0.0)
        return mjx_data, obs, reward, done
