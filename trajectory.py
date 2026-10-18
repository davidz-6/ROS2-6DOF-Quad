
import numpy as np

# TRAJECTORY PARAMETERS =====================

hover_position = np.array([0.0, 0.0, 1.0])
yaw_desired = 0.0


# HOVER TRAJECTORY ==========================

def hover_trajectory(t):

    p_des = hover_position.copy()
    v_des = np.zeros(3)
    a_des = np.zeros(3)
    yaw_des = yaw_desired

    return p_des, v_des, a_des, yaw_des


# STEP TRAJECTORY ===========================

def step_trajectory(t):

    if t < 3.0:
        p_des = np.array([0.0, 0.0, 1.0])
    else:
        p_des = np.array([1.0, 0.0, 1.0])

    v_des = np.zeros(3)
    a_des = np.zeros(3)
    yaw_des = 0.0

    return p_des, v_des, a_des, yaw_des


# SMOOTH TRAJECTORY =========================

def smooth_trajectory(t):

    t0 = 2.0
    tf = 5.0

    p0 = np.array([0.0, 0.0, 1.0])
    pf = np.array([1.0, 0.0, 1.0])

    T = tf - t0

    if t <= t0:
        p_des = p0.copy()
        v_des = np.zeros(3)
        a_des = np.zeros(3)

    elif t >= tf:
        p_des = pf.copy()
        v_des = np.zeros(3)
        a_des = np.zeros(3)

    else:
        s = (t - t0) / T

        h = 3*s**2 - 2*s**3
        h_dot = (6*s - 6*s**2) / T
        h_ddot = (6 - 12*s) / T**2

        p_des = p0 + (pf - p0)*h
        v_des = (pf - p0)*h_dot
        a_des = (pf - p0)*h_ddot

    yaw_des = 0.0

    return p_des, v_des, a_des, yaw_des

def figure_eight_trajectory(t):
    A = 0.5       # Amplitude (meters)
    w = 0.2       # Angular frequency (rad/s)
    z = 1.0       # Altitude (meters)

    p_des = np.array([
        A * np.sin(w*t),
        A * np.sin(2*w*t) / 2,
        z
    ])

    v_des = np.array([
        A*w*np.cos(w*t),
        A*w*np.cos(2*w*t),
        0.0
    ])


    a_des = np.array([
        -A*w**2*np.sin(w*t),
        -2*A*w**2*np.sin(2*w*t),
        0.0
    ])

    yaw_des = 0.0

    return p_des, v_des, a_des, yaw_des
