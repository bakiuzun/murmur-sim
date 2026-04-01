import jax.numpy as jnp 


def quat_to_rotmat(q):
    w, x, y, z = q[0], q[1], q[2], q[3]
    r00 = 1 - 2*(y**2 + z**2)
    r01 = 2*(x*y - w*z)
    r02 = 2*(x*z + w*y)
    r10 = 2*(x*y + w*z)
    r11 = 1 - 2*(x**2 + z**2)
    r12 = 2*(y*z - w*x)
    r20 = 2*(x*z - w*y)
    r21 = 2*(y*z + w*x)
    r22 = 1 - 2*(x**2 + y**2)
    return jnp.array([r00, r01, r02, r10, r11, r12, r20, r21, r22])





