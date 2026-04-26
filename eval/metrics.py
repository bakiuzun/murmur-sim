import torch


def compute_metrics(obs, 
                    actions,
                    reward,
                    sucess_counter):


    lin_vel = obs[:,9:9+3]
    ang_vel = obs[:,9+3:9+3+3]
    vz = lin_vel[:,2]
    vxy = lin_vel[:,:2]

    

    # maybe an episode has been finished so the previous is not accurate anymore but 
    # for now it's all good
    prev_action = actions[:-1]
    actions = actions[1:]

    return {
        "vxyz": torch.linalg.norm(lin_vel,axis=-1).mean(),
        "gyro_norm": torch.linalg.norm(ang_vel,axis=-1).mean(),
        "action_mean": actions.mean(),
        "action_jerk": torch.sum((actions - prev_action)**2,axis=-1).mean(),
        "reward_mean": reward.mean(),
        'success_counter': sucess_counter.mean()
    }


