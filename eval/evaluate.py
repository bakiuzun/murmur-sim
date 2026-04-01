import os
import json
import time
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
from flax import nnx
from models import ActorCritic
from structs import ModelSpec
from envs.utils import quat_to_rotmat
from utils import load_model
from typing import NamedTuple


# ─── Config ─────────────────────────────────────────────────────────────────

DRONE_XML = 'drone_models/skydio_x2/scene.xml'
CHECKPOINT_DIR = 'checkpoints'
TARGET_HEIGHT = 5.0
NUM_EVAL_STEPS = 3000          # ~6 seconds at 500Hz
TAU = 0.025                    # actuator time constant (seconds)
DRONE_HZ = 100                 # policy frequency
NUM_SEEDS = 5                  # run each model N times for statistics
RENDER = False                 # set True to visualize best model after


# ─── Data structures ────────────────────────────────────────────────────────

class EvalMetrics(NamedTuple):
    height_errors: list
    xy_drifts: list
    velocities: list       # full linear velocity norm
    vz_values: list
    tilts: list            # 1 - quat_w (0 = upright)
    gyro_norms: list
    gyro_rolls: list
    gyro_pitches: list
    gyro_yaws: list
    action_means: list
    action_jerks: list
    accel_norms: list
    rewards: list


# ─── Helpers ────────────────────────────────────────────────────────────────

def setup_sensors(mj_model):
    """Get sensor slices — computed once."""
    quat_id = mj_model.sensor('body_quat').id
    accel_id = mj_model.sensor('body_linacc').id
    gyro_id = mj_model.sensor('body_gyro').id
    adr = mj_model.sensor_adr
    dim = mj_model.sensor_dim
    return {
        'quat': slice(adr[quat_id], adr[quat_id] + dim[quat_id]),
        'accel': slice(adr[accel_id], adr[accel_id] + dim[accel_id]),
        'gyro': slice(adr[gyro_id], adr[gyro_id] + dim[gyro_id]),
    }


def get_obs(mj_data, sensors):
    """Extract observation from MuJoCo data (CPU, not MJX)."""
    sd = mj_data.sensordata
    quat = sd[sensors['quat']]
    # Rotation matrix from quaternion
    w, x, y, z = quat
    rot = [
        1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ]
    return jnp.concatenate([
        jnp.array(rot),                          # [0:9]
        jnp.array(sd[sensors['accel']]),          # [9:12]
        jnp.array(sd[sensors['gyro']]),           # [12:15]
        jnp.array([mj_data.qpos[2]]),             # [15] height
        jnp.array(mj_data.qvel[0:3]),             # [16:19] velocity
    ])


def simple_reward(obs, prev_obs, target_height):
    """Lightweight reward for evaluation (no action penalty)."""
    height = float(obs[15])
    prev_height = float(prev_obs[15])
    prev_dist = abs(prev_height - target_height)
    curr_dist = abs(height - target_height)
    progression = prev_dist - curr_dist
    vel_penalty = float(jnp.sum(jnp.square(obs[16:19])))
    return progression - 0.001 * vel_penalty


# ─── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, mj_model, sensors, seed=0):
    """Run one evaluation episode, return per-step metrics."""
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, mj_data)

    timestep = mj_model.opt.timestep
    n_substeps = int((1 / timestep) / DRONE_HZ)
    alpha = timestep / (TAU + timestep)

    rng = jax.random.PRNGKey(seed)
    obs = get_obs(mj_data, sensors)
    prev_obs = obs
    prev_action = jnp.zeros(4)
    smoothed_ctrl = jnp.zeros(4)

    # Storage
    height_errors = []
    xy_drifts = []
    velocities = []
    vz_values = []
    tilts = []
    gyro_norms = []
    gyro_rolls = []
    gyro_pitches = []
    gyro_yaws = []
    action_means = []
    action_jerks = []
    accel_norms = []
    rewards = []

    for step in range(NUM_EVAL_STEPS):
        rng, k1 = jax.random.split(rng)
        action, _, _ = model(obs, k1)

        # Actuator smoothing with tau
        raw_ctrl = action * 13.0
        smoothed_ctrl = alpha * raw_ctrl + (1 - alpha) * smoothed_ctrl
        mj_data.ctrl[:] = smoothed_ctrl

        # Physics substeps
        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        obs = get_obs(mj_data, sensors)

        # ── Metrics ──
        height = float(obs[15])
        gyro = obs[12:15]
        accel = obs[9:12]
        lin_vel = obs[16:19]
        quat = mj_data.sensordata[sensors['quat']]

        height_errors.append(abs(height - TARGET_HEIGHT))
        xy_drifts.append(float(jnp.sqrt(mj_data.qpos[0]**2 + mj_data.qpos[1]**2)))
        velocities.append(float(jnp.linalg.norm(lin_vel)))
        vz_values.append(abs(float(lin_vel[2])))
        tilts.append(float(1.0 - quat[0]))
        gyro_norms.append(float(jnp.linalg.norm(gyro)))
        gyro_rolls.append(abs(float(gyro[0])))
        gyro_pitches.append(abs(float(gyro[1])))
        gyro_yaws.append(abs(float(gyro[2])))
        action_means.append(float(jnp.mean(action)))
        accel_norms.append(float(jnp.linalg.norm(accel)))
        rewards.append(simple_reward(obs, prev_obs, TARGET_HEIGHT))

        # Action jerk
        if step > 0:
            jerk = float(jnp.sum(jnp.square(action - prev_action)))
            action_jerks.append(jerk)
        else:
            action_jerks.append(0.0)

        prev_obs = obs
        prev_action = action

        # Early termination check
        rotation_z = float(obs[8])
        if (abs(mj_data.qpos[0]) >= 10.0 or
            abs(mj_data.qpos[1]) >= 10.0 or
            abs(mj_data.qpos[2]) >= 15.0 or
            rotation_z < 0.0):
            break

    return EvalMetrics(
        height_errors=height_errors,
        xy_drifts=xy_drifts,
        velocities=velocities,
        vz_values=vz_values,
        tilts=tilts,
        gyro_norms=gyro_norms,
        gyro_rolls=gyro_rolls,
        gyro_pitches=gyro_pitches,
        gyro_yaws=gyro_yaws,
        action_means=action_means,
        action_jerks=action_jerks,
        accel_norms=accel_norms,
        rewards=rewards,
    )


def summarize(metrics: EvalMetrics):
    """Compute summary stats from per-step metrics."""
    import numpy as np

    def stats(vals):
        a = np.array(vals)
        return {"mean": float(a.mean()), "std": float(a.std()), "max": float(a.max())}

    # Split into phases: ascent (first 500 steps) and hover (rest)
    split = min(500, len(metrics.height_errors))

    return {
        "steps_survived": len(metrics.height_errors),
        "height_error": stats(metrics.height_errors),
        "height_error_hover": stats(metrics.height_errors[split:]) if len(metrics.height_errors) > split else None,
        "xy_drift": stats(metrics.xy_drifts),
        "xy_drift_final": float(metrics.xy_drifts[-1]) if metrics.xy_drifts else 0,
        "velocity": stats(metrics.velocities),
        "vz": stats(metrics.vz_values),
        "tilt": stats(metrics.tilts),
        "gyro_norm": stats(metrics.gyro_norms),
        "gyro_roll": stats(metrics.gyro_rolls),
        "gyro_pitch": stats(metrics.gyro_pitches),
        "gyro_yaw": stats(metrics.gyro_yaws),
        "action_mean": stats(metrics.action_means),
        "action_jerk": stats(metrics.action_jerks),
        "accel_norm": stats(metrics.accel_norms),
        "reward": stats(metrics.rewards),
        "reward_total": float(sum(metrics.rewards)),
    }


# ─── Scoring ────────────────────────────────────────────────────────────────

def compute_score(summary):
    """Single score combining all metrics. Lower is better.
    
    Weights reflect what matters for real deployment:
    - Height accuracy: most important
    - Stability (tilt, gyro): very important
    - Action smoothness: important for hardware
    - XY drift: moderate importance
    """
    if summary['steps_survived'] < NUM_EVAL_STEPS * 0.8:
        return float('inf')  # crashed = worst score

    hover = summary.get('height_error_hover')
    if hover is None:
        hover = summary['height_error']

    score = (
        10.0 * hover['mean']                    # height accuracy
        + 5.0 * summary['tilt']['mean']          # attitude stability
        + 3.0 * summary['gyro_norm']['mean']     # rotation smoothness
        + 3.0 * summary['action_jerk']['mean']   # motor smoothness
        + 2.0 * summary['xy_drift']['mean']      # lateral stability
        + 1.0 * summary['velocity']['mean']      # should be near zero
    )
    return score


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print(f"=" * 70)
    print(f"DRONE POLICY BENCHMARK")
    print(f"tau={TAU}s | target={TARGET_HEIGHT}m | steps={NUM_EVAL_STEPS} | seeds={NUM_SEEDS}")
    print(f"=" * 70)

    # Setup
    mj_model = mujoco.MjModel.from_xml_path(DRONE_XML)
    sensors = setup_sensors(mj_model)

    actorSpec = ModelSpec(
        hidden_sizes=jnp.array([64, 64, 4]),
        hidden_activation=nnx.relu,
        last_activation=None,
    )
    criticSpec = ModelSpec(
        hidden_sizes=jnp.array([64, 64, 1]),
        hidden_activation=nnx.relu,
        last_activation=None,
    )
    model = ActorCritic(19, 4, actorSpec, criticSpec, nnx.Rngs(0))
    graphdef, _, _ = nnx.split(model, nnx.Param, ...)

    # Find checkpoints
    model_paths = sorted([
        os.path.join(CHECKPOINT_DIR, f)
        for f in os.listdir(CHECKPOINT_DIR)
        if f.endswith('.pt')
    ])

    if not model_paths:
        print(f"No checkpoints found in {CHECKPOINT_DIR}/")
        return

    print(f"\nFound {len(model_paths)} checkpoints\n")

    # Evaluate each
    results = {}
    for path in model_paths:
        name = os.path.basename(path).replace('.pt', '')
        print(f"{'─' * 50}")
        print(f"Evaluating: {name}")

        model = load_model(path, graphdef)

        seed_summaries = []
        for seed in range(NUM_SEEDS):
            metrics = evaluate_model(model, mj_model, sensors, seed=seed)
            summary = summarize(metrics)
            seed_summaries.append(summary)

        # Average across seeds
        avg_summary = {}
        for key in seed_summaries[0]:
            vals = [s[key] for s in seed_summaries]
            if isinstance(vals[0], dict):
                avg_summary[key] = {
                    k: float(sum(v[k] for v in vals) / len(vals))
                    for k in vals[0]
                }
            elif isinstance(vals[0], (int, float)):
                avg_summary[key] = float(sum(vals) / len(vals))
            else:
                avg_summary[key] = vals[0]

        score = compute_score(avg_summary)
        avg_summary['score'] = score
        results[name] = avg_summary

        # Print key metrics
        print(f"  Steps survived:  {avg_summary['steps_survived']:.0f} / {NUM_EVAL_STEPS}")
        hover_err = avg_summary.get('height_error_hover') or avg_summary['height_error']
        print(f"  Height error:    {hover_err['mean']:.4f}m (hover phase)")
        print(f"  XY drift:        {avg_summary['xy_drift']['mean']:.4f}m")
        print(f"  Tilt:            {avg_summary['tilt']['mean']:.6f}")
        print(f"  Gyro norm:       {avg_summary['gyro_norm']['mean']:.4f} rad/s")
        print(f"  Action jerk:     {avg_summary['action_jerk']['mean']:.4f}")
        print(f"  Velocity:        {avg_summary['velocity']['mean']:.4f} m/s")
        print(f"  Reward total:    {avg_summary['reward_total']:.2f}")
        print(f"  SCORE:           {score:.4f} (lower is better)")

    # ─── Ranking ────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL RANKING (lower score = better)")
    print(f"{'=' * 70}")

    ranked = sorted(results.items(), key=lambda x: x[1]['score'])
    for rank, (name, summary) in enumerate(ranked, 1):
        marker = " ★" if rank == 1 else ""
        print(f"  {rank}. {name}")
        print(f"     Score: {summary['score']:.4f}{marker}")
        hover_err = summary.get('height_error_hover') or summary['height_error']
        print(f"     Height: {hover_err['mean']:.4f}m | "
              f"Drift: {summary['xy_drift']['mean']:.4f}m | "
              f"Jerk: {summary['action_jerk']['mean']:.4f} | "
              f"Tilt: {summary['tilt']['mean']:.6f}")
        print()

    # Save results
    os.makedirs('eval_results', exist_ok=True)
    output_path = f"eval_results/benchmark_tau{TAU}.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")

    # Return best model name for optional rendering
    best_name = ranked[0][0]
    print(f"\nBest model: {best_name}")
    return best_name


if __name__ == "__main__":
    main()
