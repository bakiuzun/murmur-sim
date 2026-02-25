# murmur-sim
autonome uav

max force per motor -> 13 Newtons -> 51 Newtons 
total weights -> 1.325 KG 
Gravity -> 9.81 

Minimal thrust to apply just to hover -> weights =  9.81 * 1.325 = 12.998 Newtons
Each motor should apply 12.998/4 ~3.24 to make the drone hover 

Environement INFO:
The actions shape is 4 (4 motor)
the output of the network will be between (0,1) BUT multiplied by 13 for the simulation 

For the observation we have gyro (3 values),accel (3 values), quat (4 values)
Should be enough to make it JUST stable


First goal:
Should just start and go up at 2m of altitude then stay there without moving too much that's it! 

RL algo -> PPO (can be changed thoughh)
