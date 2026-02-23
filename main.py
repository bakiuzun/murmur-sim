import mujoco 
import mujoco.viewer
import time
model = mujoco.MjModel.from_xml_path('skydio_x2/scene.xml')
data = mujoco.MjData(model)


sensor_accelerometer = data.sensor('body_linacc')
sensor_gyro = data.sensor('body_gyro')
sensor_quat = data.sensor('body_quat')

with mujoco.viewer.launch_passive(model,data) as viewer:
    while viewer.is_running():
        step_start = time.time()

        mujoco.mj_step(model,data)
        viewer.sync()


        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


