# Quadcopter Dynamics Simulation

A simple Python simulation of quadcopter motion using Newton-Euler equations and quaternions to represent orientation.

## Features
- Simulates position, velocity, orientation, and angular velocity.
- Uses `solve_ivp` to simulate 10 seconds of motion.
- Applies constant thrust and torque.
- Validates quaternion normalization.
- Plots position and velocity over time.

## Requirements
- Python 3
- NumPy
- SciPy
- Matplotlib
- `quadrotor_dynamics` package (local dynamics functions)

##Running
Can run simple sim using sim.py, or create visualization file and testing using RViz and Gazebo

## Reference
Cariño, J., Abaunza, H., & Castillo, P. (2022). *A fully-actuated quadcopter representation using quaternions*. International Journal of Control.

https://doi.org/10.1080/00207179.2022.2129789

The simulation uses the quaternion-based quadcopter dynamics described in this paper (Equations 5–6).
