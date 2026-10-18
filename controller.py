
import numpy as np

# CONTROLLER GAINS ============================

Kp = np.diag([1.0, 1.0, 4.0])
Kv = np.diag([2.0, 2.0, 3.0])


# POSITION CONTROLLER =========================

def position_controller(p, v, p_des, v_des, a_des):

    e_p = p_des - p

    e_v = v_des - v

    #Commanded acceleration
    a_cmd = a_des + Kp @ e_p + Kv @ e_v

    return a_cmd


def thrust_attitude_controller(a_cmd, yaw_des, m, g):

    # Desired force in world frame
    F_des = m * (a_cmd - g)

    # Desired thrust magnitude
    T_des = np.linalg.norm(F_des)

    if T_des < 1e-9:
        raise ValueError("Desired thrust direction undefined")

    b3_des = F_des / T_des

    b1_ref = np.array([
        np.cos(yaw_des),
        np.sin(yaw_des),
        0.0
    ])

    b2_cross = np.cross(b3_des, b1_ref)
    b2_norm = np.linalg.norm(b2_cross)

    if b2_norm < 1e-9:
        raise ValueError("Desired heading is parallel to thrust")

    b2_des = b2_cross / b2_norm

    #Complete orthonormal frame
    b1_des = np.cross(b2_des, b3_des)

    R_des = np.column_stack([
        b1_des,
        b2_des,
        b3_des
    ])

    return T_des, R_des


from scipy.spatial.transform import Rotation as Rot

#ATTITUDE GAINS
KR = np.diag([0.08, 0.08, 0.04])
KW = np.diag([0.08, 0.08, 0.02])


def attitude_controller(q, omega, R_des, J):

    # Current rotation matrix
    # Quaternion stored as [w, x, y, z]
    R_actual = Rot.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()

    error_matrix = 0.5 * (
        R_des.T @ R_actual
        - R_actual.T @ R_des
    )

    #Vee operator
    e_R = np.array([
        error_matrix[2, 1],
        error_matrix[0, 2],
        error_matrix[1, 0]
    ])

    e_omega = omega

    #Geometric control law
    tau = (
        -KR @ e_R
        -KW @ e_omega
        +np.cross(omega, J @ omega)
    )

    return tau
