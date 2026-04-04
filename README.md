# murmur-sim v2 
V1 (hovering + domain randomization)
V2 (waypoints following) 

**murmur-sim** is an autonomous UAV (drone) simulation project. The final goal is to have an agent able to understand the world and do the requested task given by a human. First version was hovering, the second version was waypoints following and the v3 is camera integration + LLM/VLM/VLN (whatever that is needed to make it think). The RL algorithm used is PPO for the first 2 version but this can be changed anytime!. I use JAX/Optax which DRASTICALLY increases computation task.
I did inlcude domain randomization even for the first version just to make it ready for real world deployment.

## ⚙️ Physics & Hardware Specifications
* **Total Weight:** 1.325 kg
* **Gravity:** 9.81 m/s²
* **Minimal Hover Thrust (Total):** 12.998 N (calculated as 9.81 * 1.325)
* **Minimal Hover Thrust (Per Motor):** ~3.24 N

---

## 🌍 Environment Setup (JAX & MuJoCo MJX)
* **Simulation Frequencies:** The physics engine runs at 500 Hz (dt = 0.002). The drone executes actions every 5 physics steps, resulting in a 100 Hz control loop for the agent.
* **Action Space:** 4 continuous values corresponding to the 4 motors. The network outputs values that are unbounded, which are then multiplied by 13.0 for simulation control.
* **Observation Space (Size: 21):**
  * Rotation Matrix (9)
  * Accelerometer (3)
  * Gyroscope (3)
  * Z-position (1)
  * X, Y, Z velocities (3)
  * Position of the waypoints relative to the drone (3)
---

## 🧠 RL Training & Reward Shaping History
Training is handled via PPO. The entire environment and training loop are fully vectorized using `jax.lax.scan` and `jax.vmap` for massive parallelization across 2048 environments. Normalizing the advantage proved to be a massive improvement for overall stability.


## 🚀 Current Progress & Milestones

### Waypoint Following & Reward Shaping
A major challenge was the agent learning to fly at full speed toward a single waypoint without preparing for the next one, leading to unstable flight and crashes. After testing multiple configurations, we found the optimal balance. 

**The Solution:** Massively rewarding progression (`delta_prog: 10.0`) and target reaching (`delta_closetarget: 10.0`) while keeping physics penalties (`delta_linvel`, `delta_angvel`) low enough.

#### 🏆 Best Reward Configuration
```python
reward_presets = {
    'best_waypoint_following': {
        'target_height': 5.0, 
        'delta_angvel': 0.001, 
        'delta_linvel': 0.001,
        'delta_prog': 10.0,
        'delta_actions': 0.005,
        'action_threshold': 0.0,
        'delta_lateral_vel': 0.00,
        'delta_crash': 1.0,
        'delta_hover': 0.01,
        'delta_closetarget': 10.0,
        'v_max': 10.0,
        'dt': 0.002 # Mujoco env
    },  
}  

DR_config = {
    'randomize_height': False,
    'randomize_quat': False,
    'randomize_linvel': False,
    'randomize_angvel': False,
    'randomize_thrust': False,
    'randomize_motor_constant': False,
    'randomize_waypoints': True,
    'waypoints_x': (-5,5),
    'waypoints_y': (-5,5),
    'waypoints_z': (1,5), 
    'nominal_thrust': 13.0,
    'thrust_variation': 0.0, 
    'motor_tau': 0.0,
    'motor_tau_variation': 0.0, 
    'quat_angle': 0.3, 
    'angvel_val': 0.05,
    'linvel_val': 0.5
}

```python
import jax.numpy as jnp

def compute_reward(obs, previous_obs, current_actions, previous_actions, config):
    height = obs[15]
    prev_height = previous_obs[15]
    
    gyro = obs[12:15]
    lin_vel = obs[16:19]
    curr_waypoints_dist = obs[19:22]
    prev_waypoints_dist = previous_obs[19:22]
     
    progression = jnp.linalg.norm(prev_waypoints_dist) - jnp.linalg.norm(curr_waypoints_dist)
    
    # Bounded max velocity progression
    batched_vmax = jnp.ones_like(progression) * (config['v_max'] * config['dt'])
    
    actions = jnp.abs(current_actions - previous_actions)
    actions = jnp.sum(actions - config['action_threshold'], axis=-1)
    
    crash_p = jnp.where(height < 0.1, 1, 0)

    dist_scalar = jnp.linalg.norm(curr_waypoints_dist)
    close_target_r = jnp.where(dist_scalar < 1., 1, 0)  

    reward = (
        config['delta_prog'] * jnp.minimum(progression, batched_vmax)
        + config['delta_closetarget'] * close_target_r
        - config['delta_linvel'] * jnp.square(jnp.linalg.norm(lin_vel, axis=-1)) 
        - config['delta_actions'] * jnp.maximum(actions, 0)
        - config['delta_angvel'] * jnp.square(jnp.linalg.norm(gyro, axis=-1))
        - config['delta_crash'] * crash_p
    )

    return reward, close_target_r
```


