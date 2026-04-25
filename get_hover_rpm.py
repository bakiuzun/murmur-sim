import genesis as gs
import torch


# NOT WORKING RIGHT NOW

gs.init()

# 1. Setup a basic scene
scene = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01),
                show_viewer=True)
plane = scene.add_entity(gs.morphs.Plane())

# Load YOUR drone here
drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0, 0, 1.0)))

scene.build()


#14468.429183500699
#Target Hover RPM for this URDF is: 45776.94

low_rpm = 0.0
high_rpm = 50000.0
current_rpm = 50000.0

print("Calibrating Hover RPM...")

# 3. Search Loop
for i in range(500): # 50 tries is enough to get perfect precision
    # Reset drone to 1 meter high, zero velocity
    drone.set_pos(torch.tensor([0.0, 0.0, 1.0], device=gs.device))
    drone.zero_all_dofs_velocity()
    # Spin motors at our current guess
    drone.set_propellers_rpm(torch.tensor([current_rpm, current_rpm, current_rpm, current_rpm], device=gs.device))
    
    # Step physics a few times to let velocity build up
    for _ in range(50):
        scene.step()

    # Check if we are falling or rising
    z_vel = drone.get_vel()[-1].item()
    
    if z_vel > 0:
        # We are going UP. RPM is too high.
        high_rpm = current_rpm
    else:
        # We are going DOWN. RPM is too low.
        low_rpm = current_rpm
        
    # Guess the exact middle for the next try
    current_rpm = (high_rpm + low_rpm) / 2.0

print(f"✅ Calibration Complete!")
print(f"Target Hover RPM for this URDF is: {current_rpm:.2f}")