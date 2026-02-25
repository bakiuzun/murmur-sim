import envs 
import gymnasium as gym
import numpy as np
import time
env = gym.make('uavenv',render_mode='human',xml_path='skydio_x2/scene.xml')


obs,info =env.reset()

actions = np.array([0]*4,dtype=np.float32)
actions[0] = 3.8
env.step(np.ones(4,dtype=np.float32) * 4)
for i in range(150):
    env.step(actions) 
    time.sleep(0.02)
env.close()
    
