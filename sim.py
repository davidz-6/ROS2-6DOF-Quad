import numpy as np
from scipy.spatial.transform import Rotation as R
from quadrotor_dynamics.nodes.dynamics_node import position_derivative, v_dot, q_dot, omega_dot
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# PHYSICS PARAMS ============================================
m = 1.0
g = np.array([0.0, 0.0, -9.81])
J = np.diag([0.01, 0.01, 0.02])

# INITIAL CONDITIONS ========================================
p = np.array([0.0, 0.0, 0.0])
v = np.array([0.0, 0.0, 0.0])
q = np.array([1.0, 0.0, 0.0, 0.0])
omega = np.array([0.0, 0.0, 0.0])

# TEST INPUTS ===============================================
Fb = np.array([0.0, 0.0, m * 9.81])
tau = np.array([0.01, 0.0, 0.0])

def print_state(state):
    print("\n=== QUADCOPTER STATE ===")
    print("Position (m):         ", state[0:3])
    print("Velocity (m/s):       ", state[3:6])
    print("Quaternion [w,x,y,z]: ", state[6:10])
    print("Angular velocity:     ", state[10:13])
    print("===============================")

state0 = np.concatenate([p, v, q, omega])

def dynamics(t, state):

    # Extract current state
    p = state[0:3]
    v = state[3:6]
    q = state[6:10]
    omega = state[10:13]

    # Calculate derivatives
    p_derivative = position_derivative(v)
    v_derivative = v_dot(q, Fb, m, g)
    q_derivative = q_dot(q, omega)
    omega_derivative = omega_dot(J, tau, omega)

    # Combine derivatives into one state vector
    state_derivative = np.concatenate([
        p_derivative,
        v_derivative,
        q_derivative,
        omega_derivative
    ])

    return state_derivative

# SIMULATION ========================================

t_start = 0.0
t_end = 10.0

t_eval = np.linspace(t_start, t_end, 1001)

solution = solve_ivp(
    dynamics,
    (t_start, t_end),
    state0,
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-11
)

if not solution.success:
    raise RuntimeError(solution.message)

time = solution.t
states = solution.y.T

positions = states[:, 0:3]
velocities = states[:, 3:6]
quaternions = states[:, 6:10]
angular_velocities = states[:, 10:13]

print("\n=== FINAL STATE ===")
print_state(states[-1])

# QUATERNION VALIDATION (||q|| must remain 1)

q_norms = np.linalg.norm(quaternions, axis=1)
max_error = np.max(np.abs(q_norms - 1.0))
print("\nQuaternion maximum norm error:", max_error)
assert max_error < 1e-6, "Quaternion norm drift too large"

# POSITION AND VELOCITY PLOTS

plt.figure(figsize=(10, 7))

plt.subplot(2, 1, 1)
plt.plot(time, positions[:, 0], label="X")
plt.plot(time, positions[:, 1], label="Y")
plt.plot(time, positions[:, 2], label="Z")
plt.ylabel("Position (m)")
plt.title("Quadcopter Position")
plt.legend()
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, velocities[:, 0], label="Vx")
plt.plot(time, velocities[:, 1], label="Vy")
plt.plot(time, velocities[:, 2], label="Vz")
plt.xlabel("Time (s)")
plt.ylabel("Velocity (m/s)")
plt.title("Quadcopter Velocity")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()