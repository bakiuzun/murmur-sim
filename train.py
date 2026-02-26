import envs 
import gymnasium as gym
import numpy as np
import time
import jax 
import jax.numpy as jnp 
env = gym.make('uavenv',render_mode='none',xml_path='skydio_x2/scene.xml')


obs,info =env.reset()

actions = np.array([0]*4,dtype=np.float32)
actions[0] = 3.8
env.step(np.ones(4,dtype=np.float32) * 4)
for i in range(150):
    p = np.array([3.2495625, 3.2495625, 3.2495625, 3.2495625],dtype=np.float32)
    p = jnp.array([10,20,30,40])
    env.step(p)  
    time.sleep(0.02)
env.close()
    
