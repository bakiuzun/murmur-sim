import torch 



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

    height = obs[:,-1]
    ang_vel = obs[:,9+3:9+3+3]
    lin_vel = obs[:,9:9+3]
    curr_waypoints_dist = obs[:,9+3+3:9+3+3+3]
    prev_waypoints_dist = previous_obs[:,9+3+3:9+3+3+3]
     
    progression = torch.linalg.norm(prev_waypoints_dist, dim=-1) - torch.linalg.norm(curr_waypoints_dist, dim=-1)
    
    
    # MONO Race Paper Inspired
    batched_vmax = torch.ones_like(progression) * (config['v_max']*config['dt'])
    
    
    actions = torch.abs(current_actions - previous_actions)
    actions = torch.sum(actions - config['action_threshold'],axis=-1)
    



    crash_p = torch.where(height < 0.1,1,0)

    dist_scalar = torch.linalg.norm(curr_waypoints_dist,axis=-1)
    
    close_target_r = torch.where(dist_scalar < 1.,1,0)  

    # right now it moves super fast toward target and he does lose control 

    # squared linalg norm works better if velocity is near 0 then no need huge penalty
    # if velocity is big now huge penalty 
    reward = (
        config['delta_prog'] * torch.minimum(progression,batched_vmax)
        + config['delta_closetarget'] * close_target_r
        -config['delta_linvel'] * torch.square(torch.linalg.norm(lin_vel,axis=-1)) 
        -config['delta_actions'] * torch.maximum(actions,torch.zeros_like(actions))
        -config['delta_angvel'] * torch.square(torch.linalg.norm(ang_vel,axis=-1))
        -config['delta_crash'] * crash_p
    )


    return reward,close_target_r
