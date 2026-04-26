import torch 


def compute_reward(obs, previous_obs, current_actions, previous_actions, config):
    height = obs[:, -1]
    ang_vel = obs[:, 12:15]
    lin_vel = obs[:, 9:12]
    curr_waypoints_dist = obs[:, 15:18]
    prev_waypoints_dist = previous_obs[:, 15:18]
    
    R = obs[:, :9].reshape(-1, 3, 3)
    yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0])
    yaw_bonus = torch.exp(-torch.square(yaw) / (config['yaw_delta'] ** 2))
    
    progression = torch.sum(torch.square(prev_waypoints_dist), dim=-1) \
                - torch.sum(torch.square(curr_waypoints_dist), dim=-1)
    
    batched_vmax = torch.ones_like(progression) * (config['v_max'] * config['dt'])
    actions_diff = torch.sum(torch.square(current_actions - previous_actions), dim=-1)
    crash_p = torch.where(height < 0.1, 1, 0)
    dist_scalar = torch.linalg.norm(curr_waypoints_dist, dim=-1)
    close_target_r = torch.where(dist_scalar < 1., 1, 0)
    
    reward = (
        config['delta_prog'] * torch.minimum(progression, batched_vmax)
        + config['delta_closetarget'] * close_target_r
        + config['delta_yaw'] * yaw_bonus                               # ← AJOUT
        - config['delta_linvel'] * torch.square(torch.linalg.norm(lin_vel, dim=-1))
        - config['delta_actions'] * torch.maximum(actions_diff, torch.zeros_like(actions_diff))
        - config['delta_crash'] * crash_p
        - config['delta_angvel'] * torch.square(torch.linalg.norm(ang_vel, dim=-1))
     
    )
    
    #- config['delta_angvel'] * torch.square(torch.linalg.norm(ang_vel, dim=-1))
        
    return reward, close_target_r