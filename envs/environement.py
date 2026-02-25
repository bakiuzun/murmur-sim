import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import time

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


    def step(self,action): 
        """
        obs,reward,terminated,truncated,info
        """
        ## APPLY FORCE TO THE DRONE CHECK IT'S STATE RETURN THE OBS 
        self.mj_data.ctrl[0] = action[0]
        self.mj_data.ctrl[1] = action[1]
        self.mj_data.ctrl[2] = action[2]
        self.mj_data.ctrl[3] = action[3]
        mujoco.mj_step(self.mj_model,self.mj_data)
        # NOW GET THE NEW OBSERVATIONS 
        obs = self.get_observation()
        reward = self.get_reward(obs)
        terminated = False 
        truncated = False 
        info = self.get_info()
        
        self.render()

        return obs,reward,terminated,truncated,info 
    

    def get_info(self):
        return {'product_name':'MDD'}
    def get_observation(self):
        quat = self.mj_data.sensor('body_quat').data.astype(np.float32)
        accel = self.mj_data.sensor('body_linacc').data.astype(np.float32)
        gyro = self.mj_data.sensor('body_gyro').data.astype(np.float32)
        return {'quat':quat,'accel':accel,'gyro':gyro}

    def get_reward(self,obs):
        quat,accel,gyro = obs['quat'],obs['accel'],obs['gyro']
        # ET SI je regardais sa comme une gaussienne ? 
        # moyenne 0 variance faible??? avec un PDF grand dcp 
        # go simple way IF bigger than some numbers THEN reward -1 IF stable reward = +1.0
        # gyro 3 axis 
        reward = 0  
        if gyro.any() > 3:
            reward += -1 

        if quat.any() >= 1:
            reward += -1 

        
        return reward 

    def reset(self,seed=None,options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.mj_model,self.mj_data)

        obs = self.get_observation()
        info = self.get_info() 
        
        self.render()
        
        return obs,info
    


    def render(self): 
        if self.render_mode == "human":
            if self.mj_viewer is None: 
                self.mj_viewer = mujoco.viewer.launch_passive(self.mj_model,self.mj_data)
 
            self.mj_viewer.sync()
        else:pass


    def close(self):
        if self.mj_viewer is not None:
            self.mj_viewer.close()
            self.mj_viewer = None
            # sleep otherwise segmentation fault 
            time.sleep(0.1)

