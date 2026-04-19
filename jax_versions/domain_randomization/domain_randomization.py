import jax.numpy as jnp 
import jax 

def randomize_height(key):
    return jax.random.uniform(key, shape=(), minval=0.5, maxval=4.9)

def randomize_thrust(thrust_output):
    return thrust_output + jnp.randn_like(thrust_output)

def randomize_orientation(key, angle=0.3):
    """Small random rotation as a quaternion. max_angle in radians (~17 degrees)."""
    k1, k2, k3 = jax.random.split(key, 3)
    roll  = jax.random.uniform(k1, shape=(), minval=-angle, maxval=angle)
    pitch = jax.random.uniform(k2, shape=(), minval=-angle, maxval=angle)
    yaw   = jax.random.uniform(k3, shape=(), minval=-angle, maxval=angle)

    # Euler to quaternion
    cr, sr = jnp.cos(roll/2),  jnp.sin(roll/2)
    cp, sp = jnp.cos(pitch/2), jnp.sin(pitch/2)
    cy, sy = jnp.cos(yaw/2),   jnp.sin(yaw/2)

    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy

    return jnp.array([qw, qx, qy, qz])

def randomize_linvel(key,val=0.5):

    k1,k2,k3 = jax.random.split(key,3)

    x_vel = jax.random.uniform(k1,shape=(),minval=-val,maxval=val)
    y_vel = jax.random.uniform(k2,shape=(),minval=-val,maxval=val)
    z_vel = jax.random.uniform(k3,shape=(),minval=-val,maxval=val)

    return jnp.array([x_vel,y_vel,z_vel])

def randomize_angvel(key,val=0.05):

    k1,k2,k3 = jax.random.split(key,3)

    x_angvel = jax.random.uniform(k1,shape=(),minval=-val,maxval=val)
    y_angvel = jax.random.uniform(k2,shape=(),minval=-val,maxval=val)
    z_angvel = jax.random.uniform(k3,shape=(),minval=-val,maxval=val)

    return jnp.array([x_angvel,y_angvel,z_angvel])

def randomize_thrust(key,nominal=13.0,variation=0.1):
    max_value = nominal + variation*nominal
    min_value = nominal - variation*nominal

    thrust = jax.random.uniform(key,shape=(),minval=min_value,maxval=max_value)

    return thrust

def randomize_motor_t_constant(key,base_val=0.025,variation=0.3):
    t_min = base_val * (1.0 - variation) 
    t_max = base_val * (1.0 + variation)
    return jax.random.uniform(key,shape=(),minval=t_min,maxval=t_max)


def randomize_waypoints(key,x: tuple = (-5,5),y:tuple = (-5,5),z:tuple = (0.1,5)):
    x_min,x_max = x
    y_min,y_max = y 
    z_min,z_max = z 

    keys = jax.random.split(key,3)

    x_rand = jax.random.uniform(keys[0],shape=(),minval=x_min,maxval=x_max)
    y_rand = jax.random.uniform(keys[1],shape=(),minval=y_min,maxval=y_max)
    z_rand = jax.random.uniform(keys[2],shape=(),minval=z_min,maxval=z_max)
    
    return jnp.array([x_rand,y_rand,z_rand])

def randomize(mjx_data,config,key):
    ret = {
        'mjx_data':mjx_data,
        'thrust_coeff': config['nominal_thrust'],
        'motor_tau': config['motor_tau'],
        'waypoints': jnp.zeros(shape=(3,)) # x y z 
    }

    keys = jax.random.split(key,7)

    if config['randomize_height']:
        height = randomize_height(key=keys[0])
        new_qpos = mjx_data.qpos.at[2].set(height)
        mjx_data = mjx_data.replace(qpos=new_qpos)
        
    if config['randomize_quat']:
        quat = randomize_orientation(keys[1],
                                     angle=config['quat_angle'])
        new_qpos = mjx_data.qpos.at[3:7].set(quat)
        mjx_data = mjx_data.replace(qpos=new_qpos)
    
    if config['randomize_linvel']:
        linvel = randomize_linvel(keys[2],
                                  val=config['linvel_val'])
        
        new_qvel = mjx_data.qvel.at[0:3].set(linvel)
        mjx_data = mjx_data.replace(qvel=new_qvel)
    
    if config['randomize_angvel']:
        linvel = randomize_angvel(keys[3],
                                 val=config['angvel_val'])
        
        new_qvel = mjx_data.qvel.at[3:6].set(linvel)
        mjx_data = mjx_data.replace(qvel=new_qvel)

    if config['randomize_thrust']:
        new_thrust = randomize_thrust(keys[4],
                                      nominal=config['nominal_thrust'],
                                      variation=config['thrust_variation'])
        
        ret['thrust_coeff'] = new_thrust
    
    if config['randomize_motor_constant']:
        new_tau = randomize_motor_t_constant(keys[5],
                                             base_val=config['motor_tau'],
                                             variation=config['motor_tau_variation'],
                                             )
    
        ret['motor_tau'] = new_tau

    if config['randomize_waypoints']:
        new_waypoints = randomize_waypoints(keys[6],
                                            x=config['waypoints_x'],
                                            y=config['waypoints_y'],
                                            z=config['waypoints_z'])
        ret['waypoints'] = new_waypoints
    

    return ret

