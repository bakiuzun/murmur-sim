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



path = 'baseline.pt' 
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


reward_config =   {
        "height_scale": 1.0, "sigma_height": 0.5,
        "vz_scale": 0.1, "steady_scale": 0.1,
        "quat_w_scale": 0.0, "quat_w_sigma": 0.5,
        "gyro_scale": 0.0,'target_height': 1.0,
    } 



root = tk.Tk()
root.title("Drone Dashboard")
root.geometry("350x300")
root.attributes('-topmost', True) # Keeps the GUI on top of the MuJoCo window

# Labels for Obs
tk.Label(root, text="Observation Values:", font=("Arial", 18, "bold")).pack(pady=5)
obs_var = tk.StringVar()
obs_label = tk.Label(root, textvariable=obs_var, justify="left", font=("Courier", 18))
obs_label.pack(pady=5)

# Slider for Multiplier
tk.Label(root, text="Action Multiplier:", font=("Arial", 18, "bold")).pack(pady=10)
slider = tk.Scale(root, from_=0.0, to=5.0, resolution=1., orient="horizontal", length=250)
slider.set(1.0) # Default to 1x
slider.pack()

base_action = 3.2495625

timestep = mj_model.opt.timestep # 0.002 Hz -> 1/
drone_hz = 90 
n_substeps = int( (1 / timestep) / drone_hz) 



prev_obs = jnp.zeros(19)
import time 

prev_act = None 

step_counter = 0 

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
        """
        obs_text = (
            f"Reward: {rewards.compute_reward(obs):.4f}\n\n"
            f"Quat (wxyz): {obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}, {obs[3]:.2f}\n"
            f"Accel (xyz) : {obs[4]:.2f}, {obs[5]:.2f}, {obs[6]:.2f}\n"
            f"Gyro (xyz)  : {obs[7]:.2f}, {obs[8]:.2f}, {obs[9]:.2f}\n"
            f"Height (z)  : {obs[10]:.2f}"
        )
        """

        perfect_rotation = jnp.array([1,0,0,0,1,0,0,0,1])
        p_rotation = 0.01 * jnp.linalg.norm(obs[0:9] - perfect_rotation)
        target_height = 5.0 
        height = obs[-4]
        dif = jnp.abs((height - target_height))
        
        prev_linvel = jnp.linalg.norm(prev_obs[16:19])
        curr_linvel = jnp.linalg.norm(obs[16:19])
    
        slider.get()

        # Get action from trained model
        rng, k = jax.random.split(rng)
        action, _, _ = model(obs, k)
        #print("Action: ",action)

        if prev_act is None:
            prev_act = action 


        action_jerk = jnp.sum(jnp.square(action - prev_act))
        print(f"Action Jerk: {action_jerk} ")


        prev_act = action 

        mj_data.ctrl[:] = action*13
        #print(obs[-3:])        
         
        # dans le monde de mujoco je suis a 6 * 0.002 0.012 en step 
        for _ in range(n_substeps):
            step_counter += 1 
            mujoco.mj_step(mj_model, mj_data)
        
        viewer.sync() # je update le viewer pour quil se met exactement en 0.012 

        prev_obs = obs 
        try:
            root.update()
        except tk.TclError:
            print("Error TK")

        start = time.time() 
        # je vois pas pk il a besoin de dormir 
        time.sleep(n_substeps * mj_model.opt.timestep)
        #time.sleep(0.5)
        i += 1
        #if i == 2:break 
