import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time 
import jax
import jax.numpy as jnp
from mujoco import mjx 

class UAVEnvironement(gym.Env):
    metadata = {'render_modes':['none','human']}
    def __init__(self,render_mode='none',xml_path='skydio_x2/scene.xml'):
        super().__init__()
        
        self.render_mode = render_mode 
        self.xml_path = xml_path

        # This will control directly the motors
        self.action_space = gym.spaces.Box(
            low=np.array([0.0,0.0,0.0,0.0],dtype=np.float32),
            high=np.array([1.0,1.0,1.0,1.0],dtype=np.float32),
            dtype=np.float32
        )

        # This is apparently real values..
        self.observation_space = gym.spaces.Dict(
            {
                'quat': gym.spaces.Box(low=-1,
                                       high=1,
                                       shape=(4,),
                                       dtype=np.float32),
                'accel': gym.spaces.Box(low=-156.96,
                                        high=156.96,
                                        shape=(3,),
                                        dtype=np.float32),

                'gyro': gym.spaces.Box(low=-2000.0,
                                       high=2000.0,
                                       shape=(3,),
                                       dtype=np.float32)
            }
        )   
        
        self.mj_model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_viewer = None

        self.mjx_model = mjx.put_model(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model,self.mj_data)
        
        quat_id = self.mj_model.sensor('body_quat').id 
        accel_id = self.mj_model.sensor('body_linacc').id 
        gyro_id = self.mj_model.sensor('body_gyro').id 

        adr = self.mj_model.sensor_adr 
        dim = self.mj_model.sensor_dim 
         
        self._quat_slice = slice(adr[quat_id],adr[quat_id] + dim[quat_id])

        self._accel_slice = slice(adr[accel_id],adr[accel_id] + dim[accel_id])
        self._gyro_slice = slice(adr[gyro_id],adr[gyro_id] + dim[gyro_id])

    
    def step(self,action): 
        """
        obs,reward,terminated,truncated,info
        """
        self.mjx_data = self.mjx_data.replace(ctrl=action)
        self.mjx_data = mjx.step(self.mjx_model,self.mjx_data)
        obs = self.get_observation()
        print("Obs = ",obs)
        reward = self.get_reward(obs)
        terminated = False 
        truncated = False 
        info = self.get_info()
        self.render()
        return obs,reward,terminated,truncated,info 
    

    def get_info(self):
        return {'product_name':'MDD'}
    
    
    def get_observation(self):
        sensordata = self.mjx_data.sensordata
        quat = np.asarray(sensordata[self._quat_slice])
        accel = np.asarray(sensordata[self._accel_slice])
        gyro = np.asarray(sensordata[self._gyro_slice])
        return {'quat':quat,'accel':accel,'gyro':gyro}
    

    def get_reward(self,obs):
        quat,accel,gyro = obs['quat'],obs['accel'],obs['gyro']
        # ET SI je regardais sa comme une gaussienne ? 
        # moyenne 0 variance faible??? avec un PDF grand dcp 
        # go simple way IF bigger than some numbers THEN reward -1 IF stable reward = +1.0
        # gyro 3 axis 
        reward = 0
        if np.any(gyro > 3):
            reward += -1 

        if np.any(quat >= 1):
            reward += -1 

        
        return reward 

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        
        mj_data = mujoco.MjData(self.mj_model)
        self.mjx_data = mjx.put_data(self.mj_model,mj_data)
        # just to calculate sensor data..
        self.mjx_data = mjx.forward(self.mjx_model,self.mjx_data)
        #mujoco.mj_resetData(self.mj_model,self.mj_data)
        obs = self.get_observation()
        info = self.get_info() 
        
        self.render()
        return obs,info
    


    def render(self): 
        if self.render_mode == "human":
            if self.mj_viewer is None: 
                self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model,self.mj_data)
            
            mjx.get_data_into(self.mj_data,self.mj_model,self.mjx_data)
            self.mj_viewer.sync()
        else:pass


    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
            self.mj_viewer = None
            # sleep otherwise segmentation fault 
            time.sleep(0.1)

