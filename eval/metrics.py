import jax 
import jax.numpy as jnp 

def compute_metrics(mjx_data, 
                    obs, 
                    actions,
                    reward,
                    target_height=1.0):


    height = obs[:,15]
    gyro = obs[:,12:15]
    lin_vel = obs[:,16:19]
    
    x = mjx_data.qpos[0]
    y = mjx_data.qpos[1]

    vz = lin_vel[:,2]
    vxy = lin_vel[:,:2]

    # maybe an episode has been finished so the previous is not accurate anymore but 
    # for now it's all good
    prev_action = actions[:-1]
    actions = actions[1:]

    return {
        "height_error": jnp.abs(height - target_height).mean(),
        "xy_drift": jnp.sqrt(x**2 + y**2).mean(),
        "vz": jnp.abs(vz).mean(),
        "vxy": jnp.linalg.norm(vxy),
        "gyro_norm": jnp.linalg.norm(gyro).mean(),
        "action_mean": actions.mean(),
        "action_jerk": jnp.sum((actions - prev_action)**2),
        "reward_mean": reward.mean()
    }


