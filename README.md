# 🚁 murmur-sim

**murmur-sim** is an autonomous UAV (drone) simulation project. The primary goal is to train a Reinforcement Learning agent using PPO (via JAX/Flax/Optax) to take off, reach a target altitude (e.g., 2m), and maintain a stable hover with minimal drift.

---

## ⚙️ Physics & Hardware Specifications
* **Total Weight:** 1.325 kg
* **Gravity:** 9.81 m/s²
* **Minimal Hover Thrust (Total):** 12.998 N (calculated as 9.81 * 1.325)
* **Minimal Hover Thrust (Per Motor):** ~3.24 N
* **Max Force Per Motor:** 13.0 N (scalable up to 51 N)

---

## 🌍 Environment Setup (JAX & MuJoCo MJX)
* **Simulation Frequencies:** The physics engine runs at 500 Hz (dt = 0.002). The drone executes actions every 5 physics steps, resulting in a 100 Hz control loop for the agent.
* **Action Space:** 4 continuous values corresponding to the 4 motors. The network outputs values that are unbounded, which are then multiplied by 13.0 for simulation control.
* **Observation Space (Size: 19):**
  * Rotation Matrix (9)
  * Accelerometer (3)
  * Gyroscope (3)
  * Z-position (1)
  * X, Y, Z velocities (3)

---

## 🛑 Termination Conditions (Done Flag)
To prevent the agent from learning "hacks" or wasting time in unrecoverable states, episodes terminate if any of the following occur:
* Episode length exceeds 1500 steps.
* X or Y drift exceeds 15.0m.
* Z drift exceeds 20.0m.
* **Inclination limits:** The episode terminates if the drone's Z-axis rotation matrix value drops below 0.0 (indicating an extreme tilt/flip past 90 degrees).

---

## 🧠 RL Training & Reward Shaping History
Training is handled via PPO. The entire environment and training loop are fully vectorized using `jax.lax.scan` and `jax.vmap` for massive parallelization across 2048 environments. Normalizing the advantage proved to be a massive improvement for overall stability.

### 🏆 Current Best Reward Function (Progress + Hover Maintenance)
The current approach rewards the drone based on its frame-by-frame progress toward the target height to encourage smooth ascent. Once it enters a 25cm "hover zone", a Gaussian maintenance reward activates to encourage zero-velocity stabilization.

> **Reference:** The progress-based reward structure used in this project is inspired by the paper *MonoRace: Winning Champion-Level Drone Racing with Robust Monocular AI*.

#### Key Design Decisions

1. **Progress reward:** The drone is rewarded for reducing its distance to the target height frame-by-frame. The initial problem was that once the target was reached, no further reward was available, causing the drone to crash. A dedicated hover reward term was added to solve this.

2. **Lateral drift fix:** A small lateral velocity penalty (`1 - ||v_xy||`) effectively eliminated the X/Y drift the agent was exploiting.

3. **Action smoothness:** An action jerk penalty (`||a_t - a_{t-1}||²`) reduces abrupt motor changes. Current jerk values range from ~0.04 to ~0.4 — acceptable for simulation but still needs refinement for real-world deployment.

```python
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
    batched_vmax = jnp.ones_like(curr_dist) * (config['v_max'] * config['dt'])
    progression = (prev_dist - curr_dist)

    # If close to target, reward minimal movement
    hover = 1 - progression
    
    xy_vel = lin_vel[0:2]
    lateral_vel = 1 - jnp.linalg.norm(xy_vel, axis=-1)

    actions = jnp.linalg.norm(current_actions - previous_actions, axis=-1)

    # Squared norm: small velocities get small penalties, large velocities get large penalties
    reward = (
        config['delta_prog'] * jnp.minimum(progression, batched_vmax)
        + config['delta_hover'] * hover
        + config['delta_lateral_vel'] * lateral_vel
        - config['delta_linvel'] * jnp.square(jnp.linalg.norm(lin_vel, axis=-1)) 
        - config['delta_actions'] * jnp.square(actions)
        - config['delta_angvel'] * jnp.square(jnp.linalg.norm(gyro, axis=-1))
    )

    return reward
```

#### Best Configuration

| Parameter | Value |
|---|---|
| `target_height` | 5.0 |
| `delta_prog` | 1.0 |
| `delta_hover` | 0.01 |
| `delta_lateral_vel` | 0.01 |
| `delta_linvel` | 0.001 |
| `delta_angvel` | 0.000 |
| `delta_actions` | 0.002 |
| `delta_crash` | 10.0 |
| `v_max` | 10.0 |
| `dt` | 0.002 |


## 🚀 Current Progress & Milestones

### 1. Domain Randomization (Completed ✅)
Domain randomization has been fully implemented, tested, and is working without issues. It successfully introduces necessary perturbations to improve robustness and sim-to-real transfer. 
Current randomized parameters include:
* Waypoints (X, Y, Z dynamic generation)
* Motor Tau (delay) and Thrust coefficients
* Height, quaternion angles, linear, and angular velocities

### 2. Waypoint Following & Reward Shaping
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


Next Step is camera integration and LLM/VLA/VLN whatever it is to make some thinking :)
