import jax 
import jax.numpy as jnp 

def compute_metrics(mjx_data, 
                    obs, 
                    actions,
                    reward,
                    target_height=1.0):
    
    height = obs[:,10]
    vz = obs[:,11]
    quat = obs[:,0:4]     # w, x, y, z
    gyro = obs[:,7:10]    # roll rate, pitch rate, yaw rate
    accel = obs[4:7]
    x = mjx_data.qpos[0]
    y = mjx_data.qpos[1]

    prev_action = actions[:-1]
    actions = actions[1:]

    return {
        # Position — how close to target
        "height_error": jnp.abs(height - target_height).mean(),
        "xy_drift": jnp.sqrt(x**2 + y**2).mean(),

        # Velocity — should be zero for hover
        "vz": jnp.abs(vz).mean(),
        "tilt": (1.0 - quat[0]).mean() ,
        "quat_x": jnp.abs(quat[1]).mean(),
        "quat_y": jnp.abs(quat[2]).mean(),

        # Rotation — should be zero for hover
        "gyro_norm": jnp.linalg.norm(gyro).mean(),
        "gyro_roll": jnp.abs(gyro[0]).mean(),
        "gyro_pitch": jnp.abs(gyro[1]).mean(),
        "gyro_yaw": jnp.abs(gyro[2]).mean(),

        # Control quality
        "action_mean": actions.mean(),
        "action_jerk": jnp.sum((actions - prev_action)**2),

        # Acceleration
        "accel_norm": jnp.linalg.norm(accel).mean(),

        "reward_mean": reward.mean()
    }


