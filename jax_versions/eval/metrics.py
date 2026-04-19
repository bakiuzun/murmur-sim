import jax 
import jax.numpy as jnp 

def compute_metrics(mjx_data, 
                    obs, 
                    actions,
                    reward,
                    sucess_counter):


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
        "vxyz": jnp.linalg.norm(lin_vel,axis=-1).mean(),
        "gyro_norm": jnp.linalg.norm(gyro,axis=-1).mean(),
        "action_mean": actions.mean(),
        "action_jerk": jnp.sum((actions - prev_action)**2,axis=-1).mean(),
        "reward_mean": reward.mean(),
        'success_counter': sucess_counter.mean()
    }


