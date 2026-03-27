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
* **Action Space:** 4 continuous values corresponding to the 4 motors. The network outputs values between (0, 1), which are then multiplied by 13.0 for simulation control.
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

The current approach rewards the drone based on its frame-by-frame progress toward the target height to encourage smooth ascent. The problem is that after reaching the target height there is basically no more reward to get so the drone crashes... I am adding a reward hover and experimenting this to see give him reward just to maintain it's positions. Another thing that could be added is the actions penalty to minimize the action jerk

```python
def compute_reward(obs, previous_obs, config):
    height = obs[15]
    gyro = obs[12:15]
    prev_height = previous_obs[15]
    
    TARGET_HEIGHT = config['target_height']
    
    prev_dist = jnp.abs(prev_height - TARGET_HEIGHT)
    curr_dist = jnp.abs(height - TARGET_HEIGHT)
    
    v_max = 10 # 10 meter per second
    dt = 0.002
    batched_vmax = jnp.ones_like(curr_dist) * (v_max*dt)
    progression = (prev_dist - curr_dist)
    r_progress = config['delta_prog']*jnp.minimum(progression,batched_vmax)

    p_angularvel = config['delta_angvel'] * jnp.linalg.norm(gyro)

    p_crash = jnp.where(height < 0.1,config['delta_crash'],0)
    
    reward = r_progress - p_angularvel - p_crash   
    return reward
    
   

THE BEST REWARD SO FAR IS 
```python
def compute_reward(obs, previous_obs, config):
    rotation_obs = obs[0:9]
    height = obs[15]
    vz = obs[18]
    gyro = obs[12:15]
    lin_vel = obs[16:19]
    prev_height = previous_obs[15]
    
    TARGET_HEIGHT = config['target_height']
    
    prev_dist = jnp.abs(prev_height - TARGET_HEIGHT)
    curr_dist = jnp.abs(height - TARGET_HEIGHT)
    
    v_max = 10 # 10 meter per second
    dt = 0.002
    batched_vmax = jnp.ones_like(curr_dist) * (v_max*dt)
    progression = (prev_dist - curr_dist)
    
    prev_linvel = jnp.linalg.norm(previous_obs[16:19])
    curr_linvel = jnp.linalg.norm(lin_vel)
    
    
    r_hover = jnp.where(progression < 0.5,1 - progression,0.0) 
    
    reward = (
        config['delta_prog'] * jnp.minimum(progression,batched_vmax)
        + config['delta_hover'] * r_hover 
        -config['delta_linvel'] * jnp.sum(jnp.square(curr_linvel))
        -config['delta_angvel'] * jnp.sum(jnp.square(gyro))
        #-0.01 * jnp.sum(jnp.square(action - hover_action))
    )
    return reward
```
