import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
import time
from ppo import ActorCritic, load_model
from flax import nnx

# Load model
model = ActorCritic(10, 4, nnx.Rngs(0))
graphdef, params, non_params = nnx.split(model, nnx.Param, ...)
model = load_model('checkpoints/ppo_uav2', graphdef)

# Setup MuJoCo (pas MJX, le vrai renderer)
mj_model = mujoco.MjModel.from_xml_path('skydio_x2/scene.xml')
mj_data = mujoco.MjData(mj_model)

# Sensor slices
quat_id = mj_model.sensor('body_quat').id
accel_id = mj_model.sensor('body_linacc').id
gyro_id = mj_model.sensor('body_gyro').id
adr = mj_model.sensor_adr
dim = mj_model.sensor_dim
quat_slice = slice(adr[quat_id], adr[quat_id] + dim[quat_id])
accel_slice = slice(adr[accel_id], adr[accel_id] + dim[accel_id])
gyro_slice = slice(adr[gyro_id], adr[gyro_id] + dim[gyro_id])

rng = jax.random.PRNGKey(0)
i = 0
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    while viewer.is_running():
        # Get obs
        sd = mj_data.sensordata
        obs = jnp.concatenate([
            jnp.array(sd[quat_slice]),
            jnp.array(sd[accel_slice]),
            jnp.array(sd[gyro_slice]),
            mj_data.qpos[2:3]
        ])
         
        # Get action from trained model
        rng, k = jax.random.split(rng)
        action, _, _ = model(obs, k)

        mj_data.ctrl[:] = action
        mujoco.mj_step(mj_model, mj_data)

        viewer.sync()
        time.sleep(mj_model.opt.timestep)

        i += 1
        #if i == 2:break 
