
import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as Rot

from quadrotor_dynamics.nodes.dynamics_node import (
    position_derivative,
    v_dot,
    q_dot,
    omega_dot
)

from quadrotor_dynamics.trajectories.trajectory import smooth_trajectory

from quadrotor_dynamics.control.controller import (
    position_controller,
    thrust_attitude_controller,
    attitude_controller
)


# PHYSICAL PARAMETERS ========================

m = 1.0
g = np.array([0.0, 0.0, -9.81])
J = np.diag([0.01, 0.01, 0.02])


# INITIAL CONDITIONS =========================

p0 = np.array([0.0, 0.0, 0.0])
v0 = np.zeros(3)

# Quaternion format: [w, x, y, z]
q0 = np.array([1.0, 0.0, 0.0, 0.0])

omega0 = np.zeros(3)

state0 = np.concatenate([p0, v0, q0, omega0])


# CLOSED-LOOP DYNAMICS ========================

def dynamics(t, state):

    # Current state
    p = state[0:3]
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]

    # Normalize quaternion for rotation calculations
    q = q / np.linalg.norm(q)

    p_des, v_des, a_des, yaw_des = smooth_trajectory(t)

    a_cmd = position_controller(
        p, v, p_des, v_des, a_des
    )

    T_des, R_des = thrust_attitude_controller(
        a_cmd, yaw_des, m, g
    )

    tau = attitude_controller(
        q, omega, R_des, J
    )

    R_actual = Rot.from_quat(
        q[[1, 2, 3, 0]]
    ).as_matrix()

    b3_actual = R_actual[:, 2]
    b3_des = R_des[:, 2]

    T = T_des * np.dot(b3_des, b3_actual)

    # Conventional quadcopter: no negative thrust
    T = max(0.0, T)

    Fb = np.array([0.0, 0.0, T])

    p_derivative = position_derivative(v)
    v_derivative = v_dot(q, Fb, m, g)
    q_derivative = q_dot(q, omega)
    omega_derivative = omega_dot(J, tau, omega)

    return np.concatenate([
        p_derivative,
        v_derivative,
        q_derivative,
        omega_derivative
    ])


# SIMULATION ==================================

t_start = 0.0
t_end = 12.0

t_eval = np.linspace(t_start, t_end, 1001)

solution = solve_ivp(
    dynamics,
    (t_start, t_end),
    state0,
    t_eval=t_eval,
    rtol=1e-8,
    atol=1e-10
)

if not solution.success:
    raise RuntimeError(solution.message)

time = solution.t
states = solution.y.T

positions = states[:, 0:3]
velocities = states[:, 3:6]
quaternions = states[:, 6:10]
angular_velocities = states[:, 10:13]


# DESIRED TRAJECTORY ==========================

desired_positions = np.array([
    smooth_trajectory(t)[0] for t in time
])


# TRACKING ERROR ==============================

position_errors = desired_positions - positions

error_norm = np.linalg.norm(
    position_errors, axis=1
)

rms_error = np.sqrt(
    np.mean(error_norm**2)
)

print("Final position:", positions[-1])
print("Desired position:", desired_positions[-1])
print("Final position error:", error_norm[-1])
print("RMS position error:", rms_error)

q_norms = np.linalg.norm(quaternions, axis=1)

print(
    "Maximum quaternion norm error:",
    np.max(np.abs(q_norms - 1.0))
)


# PLOTS =======================================

plt.figure(figsize=(10, 9))

plt.subplot(3, 1, 1)

plt.plot(time, positions[:, 2], label="Actual Z")
plt.plot(
    time,
    desired_positions[:, 2],
    "--",
    label="Desired Z"
)

plt.ylabel("Altitude (m)")
plt.title("Closed-Loop Altitude Tracking")
plt.legend()
plt.grid()


plt.subplot(3, 1, 2)

plt.plot(time, error_norm)

plt.ylabel("Position Error (m)")
plt.title("Position Tracking Error")
plt.grid()


plt.subplot(3, 1, 3)

plt.plot(
    time,
    angular_velocities[:, 0],
    label="Roll Rate"
)

plt.plot(
    time,
    angular_velocities[:, 1],
    label="Pitch Rate"
)

plt.plot(
    time,
    angular_velocities[:, 2],
    label="Yaw Rate"
)

plt.xlabel("Time (s)")
plt.ylabel("Angular Velocity (rad/s)")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
