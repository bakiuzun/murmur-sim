import jax.numpy as jnp




def gaussian_reward(x,target,sigma):
    """
    Close to target we get reward ~1   
    """
    return jnp.exp(-((x - target)**2) / ((sigma+1e-8)**2))


def compute_reward(obs,reward_config):
    rotation_obs = obs[0:9]
    accel_obs = obs[9:12]
    gyro_obs = obs[12:15]
    height = obs[15]
    vz = obs[16]

    TARGET_HEIGHT = reward_config['target_height']

    r_pos = reward_config['height_scale'] * gaussian_reward(x=height,
                            target=TARGET_HEIGHT,
                            sigma=reward_config['sigma_height'])

    proximity = jnp.exp(-((height - TARGET_HEIGHT)**2) / ((reward_config['sigma_height']+1e-8)**2))

    # closer -> bigger penalty for being fast 
    r_vz =  -reward_config['vz_scale'] * vz**2 * proximity 
    
    r_steady = -reward_config['steady_scale'] * vz**2 

    """
    r_upright = reward_config['quat_w_scale'] * gaussian_reward(x=quat[0],
                                target=1,
                                sigma=reward_config['quat_w_sigma'])
    
    r_gyro = -reward_config['gyro_scale'] * jnp.sum(gyro**2)
    """
    reward = r_pos + r_vz + r_steady + 0 + 0

    return reward


