import numpy as np
from scipy.spatial.transform import Rotation as R

#QUATERNIONS

def q_update(q_curr, q_delta):
    w1, x1, y1, z1 = q_curr
    w2, x2, y2, z2 = q_delta

    q_new = np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

    return q_new


def q_dot(q_curr, omega):
    omega_quat = np.array([
        0.0,
        omega[0],
        omega[1],
        omega[2]
    ])
    q_dot = 0.5 * q_update(q_curr, omega_quat)
    return q_dot


#MECH DYNAMICS

#p_dot = v
def position_derivative(velocity):
    return velocity

#v_dot = R(q)Fb/m + g

def v_dot(q, Fb, m, g):
    #Define Quaternion to rotation matrix
    rotation = R.from_quat(q[[1,2,3,0]])
    rotation_matrix = rotation.as_matrix()

    v_dot = (rotation_matrix @ Fb) / m + g
    return v_dot

#omega_dot = J^-1(tau - omega x J * omega)

def omega_dot(J, tau, omega):
    omega = np.asarray(omega, dtype=float)
    tau = np.asarray(tau, dtype=float)

    gyroscopic_term = np.cross(omega, J @ omega)

    omega_derivative = np.linalg.solve(
        J,
        tau - gyroscopic_term
    )

    return omega_derivative