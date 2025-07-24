import numpy as np
import pandas as pd
from scipy.integrate import ode
from scipy.interpolate import interp1d
import dragcoeff as dc


# Default definitions for Q, R, P
accelerometer_noise = 0.3
barometer_noise     = 1.0 # Should be lower than accelerometer, noise needs fixing.
P = np.eye(3) * 0.01
Q = np.eye(3) * (accelerometer_noise**2)
R = np.eye(1) * (barometer_noise**2)

constants = [9.81, 1.225, 20, 0.0186] # g, rho, m_dry, A_body

# A matrix
def get_A(dt):
    return np.array([
        [1, dt, 0],
        [0, 1,  0],
        [0, 0,  0]
    ])

# B matrix
def get_B(dt):
    return np.array([
        [0,       0, 0.5*(dt**2)],
        [0,       0, dt],
        [0,       0, 1]
    ])

# Prediction step KF
def predict(x, P, Q, u, dt):
    g = 9.81
    az = -u[2] - g  # net acceleration (subtract gravity)
    A  = get_A(dt)
    B  = get_B(dt)
    x  = A @ x + B @ np.array([[az]])
    P  = A @ P @ A.T + Q
    return x, P

# h: Predicts pressure from altitude using the barometric formula.
#    This function serves as the nonlinear measurement function in the Kalman filter,
#    mapping the state (altitude) to a predicted pressure value.
def h(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6:
        base = 1e-6
    return P0 * (base ** (g_const / (R_const * L)))

# H: Computes the Jacobian of the measurement function h with respect to altitude.
# Compute the Jacobian (measurement matrix) for the nonlinear measurement function.  
# The barometric measurement function that predicts pressure from altitude is given by: 
#     p(h) = P0 * (1 - (L * h) / T0)^(g / (R * L))
def H(h_est):
    P0 = 101325.0
    T0 = 288.15
    L  = 0.0065
    g_const = 9.81
    R_const = 287.05
    base = 1 - (L * h_est / T0)
    if base < 1e-6:
        base = 1e-6
    exponent = (g_const / (R_const * L)) - 1.0
    dpdh = -P0 * (g_const / (R_const * T0)) * (base ** exponent)
    return np.array([[dpdh, 0, 0]])

# correction step
def correct(x, P, R, pbaro):
    z = np.array([[pbaro]])  # raw pressure directly
    h_est = x[0,0]
    z_pred = h(h_est)
    H_jac = H(h_est)
    
    y = z - np.array([[z_pred]])
    S = H_jac @ P @ H_jac.T + R
    K = P @ H_jac.T @ np.linalg.inv(S)
    
    x = x + K @ y
    P = (np.eye(3) - K @ H_jac) @ P
    return x, P

# Apogee estimator
def ode_ballistic(t, state, constants, deployment_level):
    # constants = [g, rho0, m_dry, A_total]
    g, rho0, m_dry, A_total = constants

    y, z, vy, vz = state
    v = np.hypot(vy, vz)

    # Atmospheric model to calculate speed of sound
    T0 = 288.15
    L  = 0.0065
    T  = T0 - L * z if (T0 - L * z) > 0 else 1.0
    R  = 287.05
    gamma = 1.4
    a_sound = np.sqrt(gamma * R * T)
    mach = v / a_sound if a_sound > 0 else 0

    # Get total Cd from the dragcoeff module
    deployment_mm = np.clip(deployment_level, 0.0, 1.0) * 26.0
    #Cd_val = dc.get_total_cd(mach, deployment_mm)
    Cd_val = dc.get_total_cd_for_override(mach, deployment_mm)
    
    drag_accel_z = (0.5 * rho0 * (vz * abs(vz)) * Cd_val * A_total) / m_dry

    dydt = vy
    dzdt = vz
    dvydt = 0 
    dvzdt = -g - drag_accel_z
    
    return [dydt, dzdt, dvydt, dvzdt]


def integrate_ballistic(initial_time, initial_state, constants,
                        duration=30, dt=0.1, deployment_level=0.0):
    solver = ode(lambda t, y: ode_ballistic(t, y, constants, deployment_level))
    solver.set_integrator('dopri5')
    solver.set_initial_value(initial_state, initial_time)
    
    apogee_value = None
    # Run until the absolute time reaches initial_time + duration.
    while solver.successful() and solver.t < (initial_time + duration):
        if solver.t > initial_time and solver.y[3] < 0:
            apogee_value = solver.y[1]
            break
        solver.integrate(solver.t + dt)
    
    if apogee_value is None:
        apogee_value = solver.y[1]
    return apogee_value



# Needs input of Q, R and initial P
def kf_runner(x, P, Q, R, u, pbaro, t, dt):
    x, P = predict(x, P, Q, u, dt)
    x, P = correct(x, P, R, pbaro)
    
    current_height = x[0].item()
    current_velocity = x[1].item()
    current_acceleration = x[2].item()

    initial_state = [0, current_height, 0, current_velocity] # y, z, vy, vz
    apogee = integrate_ballistic(t, initial_state, constants, duration=30, dt=0.1, deployment_level=0.0)

    return x, P, apogee