from math import dist
import mujoco
import mujoco.viewer
import jax
import jax.numpy as jnp
import time
from flax import nnx
from models import ActorCritic
from utils import *
import cv2 
from structs import ModelSpec
from envs import rewards
from envs import utils as env_utils
import numpy as np
import warp as wp 

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

path = "baseline.pt"
path = "test_massreward_2.pt"


model = load_model(f'checkpoints/{path}', graphdef)
print(f"Model: {model.log_std} ")
# Setup MuJoCo (pas MJX, le vrai renderer)
mj_model = mujoco.MjModel.from_xml_path('drone_models/skydio_x2/env.xml') 
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

waypoints = jnp.array([-1,3.4,2.])

prev_obs = jnp.zeros(19)
prev_act = None 
step_counter = 0 

current_ctrl = jnp.zeros(4)

import time 
img_height = 64
img_width = 64
channels = 3
        
cpu_renderer = mujoco.Renderer(mj_model, height=img_height, width=img_width)


def new_waypoints(x,y,z):
  return jnp.array([x + 2,y+0.5,z])


def randomize_waypoints(key,x: tuple = (-5,5),y:tuple = (-5,5),z:tuple = (0.1,5)):
    x_min,x_max = x
    y_min,y_max = y 
    z_min,z_max = z 

    keys = jax.random.split(key,3)

    x_rand = jax.random.uniform(keys[0],shape=(),minval=x_min,maxval=x_max)
    y_rand = jax.random.uniform(keys[1],shape=(),minval=y_min,maxval=y_max)
    z_rand = jax.random.uniform(keys[2],shape=(),minval=z_min,maxval=z_max)
    
    return jnp.array([x_rand,y_rand,z_rand])

with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    while viewer.is_running():
        # Get obs
        viewer.user_scn.ngeom = 0 
        geom_idx = 0 


        mujoco.mjv_initGeom(
          viewer.user_scn.geoms[geom_idx],
          type=mujoco.mjtGeom.mjGEOM_SPHERE,
          size=[0.15, 0, 0],
          pos=np.array(waypoints, dtype=np.float64),
          mat=np.eye(3).flatten(),
          rgba=np.array([1, 0, 0, 0.8], dtype=np.float32),
        )
        
        geom_idx += 1
        
        viewer.user_scn.ngeom = geom_idx
        sd = mj_data.sensordata
        rot_mat = env_utils.quat_to_rotmat(sd[quat_slice])
      
        distt = waypoints - mj_data.qpos[:3]
      
        if jnp.linalg.norm(distt) < 1:
          key = jax.random.fold_in(rng, int(time.time()))
        
          waypoints = randomize_waypoints(key,z=(1,5))
          
        body_frame = rot_mat.reshape((3,3)).T @ distt 
        obs = jnp.concatenate([
            rot_mat,
            jnp.array(sd[accel_slice]),
            jnp.array(sd[gyro_slice]),
            mj_data.qpos[2:3],
            mj_data.qvel[0:3],
            body_frame
        ])

        cpu_renderer.update_scene(mj_data, camera="fpv")
        img = cpu_renderer.render()

        
        img_bgr = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

        cv2.imshow('drone fpv',img_bgr)
        cv2.waitKey(1)

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
