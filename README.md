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

---

## 🔜 Next Steps

1. **Domain Randomization:** Introduce motor lag, random initial positions, and other perturbations to improve robustness and sim-to-real transfer.
2. **Camera Integration:** Add visual input to the observation space, enabling the agent to perceive and interact with the real world (obstacle avoidance, navigation, scene understanding).
