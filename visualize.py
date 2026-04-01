import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
import time
from flax import nnx
import tkinter as tk 
from models import ActorCritic
from utils import * 
from structs import ModelSpec
from envs import rewards
from envs import utils as env_utils

# Load mode
actorSpec = ModelSpec(
        hidden_sizes=jnp.array([64,64,4]),
        hidden_activation=nnx.relu,
        last_activation=None
    )

criticSpec = ModelSpec(
       hidden_sizes=jnp.array([64,64,1]),
       hidden_activation=nnx.relu,
       last_activation=None
)

model = ActorCritic(17, 4,actorSpec,criticSpec,nnx.Rngs(0))
graphdef, params, non_params = nnx.split(model, nnx.Param, ...)

path = "rdmenv_quatrad01_lr0.0003_gm0.99_steps100000000.0.pt"

path = "mt.pt"
model = load_model(f'checkpoints/{path}', graphdef)
print(f"Model: {model.log_std} ")
# Setup MuJoCo (pas MJX, le vrai renderer)
mj_model = mujoco.MjModel.from_xml_path('drone_models/skydio_x2/scene.xml') 
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
base_action = 3.2495625
timestep = mj_model.opt.timestep # 0.002 Hz -> 1/
drone_hz = 100 
n_substeps = int( (1 / timestep) / drone_hz)
tau = 0.0
# if alpha = 1 -> we take directly the output 
alpha = timestep / (tau + timestep) # how fast we update 

prev_obs = jnp.zeros(19)
prev_act = None 
step_counter = 0 

current_ctrl = jnp.zeros(4)

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    while viewer.is_running():
        # Get obs
        sd = mj_data.sensordata
        rot_mat = env_utils.quat_to_rotmat(sd[quat_slice])
        
        obs = jnp.concatenate([
            rot_mat,
            jnp.array(sd[accel_slice]),
            jnp.array(sd[gyro_slice]),
            mj_data.qpos[2:3],
            mj_data.qvel[0:3]
        ])


        rng, k = jax.random.split(rng)
        action, _, _ = model(obs, k)

        if prev_act is None:
            prev_act = action 
        
        output_ctrl = action*13 
        for _ in range(n_substeps):

            current_ctrl = alpha * output_ctrl + (1 - alpha) * current_ctrl
            
            mj_data.ctrl[:] = current_ctrl

            mujoco.mj_step(mj_model, mj_data)
        
        viewer.sync() # je update le viewer pour quil se met exactement en 0.012 

        prev_obs = obs 


        time.sleep(n_substeps * mj_model.opt.timestep)
        i += 1
