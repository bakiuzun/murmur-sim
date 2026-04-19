import jax.numpy as jnp
import jax 

def gaussian_reward(x,target,sigma):
    """
    Close to target we get reward ~1   
    """
    return jnp.exp(-((x - target)**2) / ((sigma+1e-8)**2))


def compute_reward(obs, 
                   previous_obs,
                   current_actions,
                   previous_actions,
                   config):
    height = obs[15]
    prev_height = previous_obs[15]
    
    gyro = obs[12:15]
    lin_vel = obs[16:19]
    curr_waypoints_dist = obs[19:22]
    prev_waypoints_dist = previous_obs[19:22]
     
    
    progression = jnp.linalg.norm(prev_waypoints_dist) - jnp.linalg.norm(curr_waypoints_dist)
    
    
    # MONO Race Paper Inspired
    batched_vmax = jnp.ones_like(progression) * (config['v_max']*config['dt'])
    
    
    actions = jnp.abs(current_actions - previous_actions)
    actions = jnp.sum(actions - config['action_threshold'],axis=-1)
    
    crash_p = jnp.where(height < 0.1,1,0)

    dist_scalar = jnp.linalg.norm(curr_waypoints_dist)
    close_target_r = jnp.where(dist_scalar < 1.,1,0)  

    # right now it moves super fast toward target and he does lose control 

    # squared linalg norm works better if velocity is near 0 then no need huge penalty
    # if velocity is big now huge penalty 
    reward = (
        config['delta_prog'] * jnp.minimum(progression,batched_vmax)
        + config['delta_closetarget'] * close_target_r
        -config['delta_linvel'] * jnp.square(jnp.linalg.norm(lin_vel,axis=-1)) 
        -config['delta_actions'] * jnp.maximum(actions,0)
        -config['delta_angvel'] * jnp.square(jnp.linalg.norm(gyro,axis=-1))
        -config['delta_crash'] * crash_p
    )

    

    return reward,close_target_r


"""
def compute_reward(obs, 
                   previous_obs,
                   current_actions,
                   previous_actions,
                   config):
    height = obs[15]
    gyro = obs[12:15]
    lin_vel = obs[16:19]
    prev_height = previous_obs[15]
    
    TARGET_HEIGHT = config['target_height']
    
    prev_dist = jnp.abs(prev_height - TARGET_HEIGHT)
    curr_dist = jnp.abs(height - TARGET_HEIGHT)
    # MONO Race Paper Inspired
    batched_vmax = jnp.ones_like(curr_dist) * (config['v_max']*config['dt'])
    progression = (prev_dist - curr_dist)

    # If close to target we reward when it is barely moving 
    # this can be changed as reward 1 - linear vel OR ang vel 
    hover = 1 - progression
    
    xy_vel = lin_vel[0:2]
    lateral_vel = 1 - jnp.linalg.norm(xy_vel,axis=-1) # reward for not moving laterally

    actions = jnp.linalg.norm(current_actions - previous_actions,axis=-1)

    # squared linalg norm works better if velocity is near 0 then no need huge penalty
    # if velocity is big now huge penalty 
    reward = (
        config['delta_prog'] * jnp.minimum(progression,batched_vmax)
        + config['delta_hover'] * hover
        + config['delta_lateral_vel'] * lateral_vel
        -config['delta_linvel'] * jnp.square(jnp.linalg.norm(lin_vel,axis=-1)) 
        -config['delta_actions'] * jnp.square(actions)
        -config['delta_angvel'] * jnp.square(jnp.linalg.norm(gyro,axis=-1))
    )

    return reward
"""