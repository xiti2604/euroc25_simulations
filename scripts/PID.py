import numpy as np
import math
import dragcoeff as dc

ctrl_time_hist = []
ctrl_pred_hist = []
ctrl_error_hist = []
ctrl_deploy_hist = []

class PIDController:
    """
    A standard PID controller class.
    """
    def __init__(self, Kp, Ki, Kd, setpoint, output_limits=(0, 1)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.output_limits = output_limits
        self.integral = 0
        self.previous_error = 0
        self.previous_time = None

    def update(self, process_variable, current_time):
        error = process_variable - self.setpoint  
        if self.previous_time is None:
            self.previous_time = current_time
            self.previous_error = error
            derivative_term = 0
        else:
            dt = current_time - self.previous_time
            if dt == 0:
                derivative_term = 0
            else:
                self.integral += error * dt
                self.integral = np.clip(self.integral, -500, 500)
                derivative_term = (error - self.previous_error) / dt

        proportional_term = self.Kp * error
        output = proportional_term + self.Ki * self.integral + self.Kd * derivative_term
        output = np.clip(output, *self.output_limits)
        self.previous_error = error
        self.previous_time = current_time
        # print(f"..., P={self.Kp*error:.4f}, I={self.Ki*self.integral:.4f}, D={self.Kd*derivative_term:.4f}") # This line is removed
        return output

    def reset(self):
        self.integral = 0
        self.previous_error = 0
        self.previous_time = None

def run_pid_controller(time, sampling_rate, state, air_brakes_object_ref,
                       env, fafnir, freya, apogee_predictor_module,
                       pid_apogee_controller, TARGET_APOGEE,
                       RATE_LIMIT_PER_SECOND, BRAKES_DISABLED=False):
    """
    This part has PID logic and gets called from the rocketoy(new)_algorithm_tester.py
    """
    

    altitude_ASL = state[2]
    altitude_AGL = altitude_ASL - env.elevation
    
    motor_burn_end_time = fafnir.burn_out_time
    
    if  altitude_AGL > 1500:
        # State for apogee predictor: [y, z, vy, vz]
        # y, vy are horizontal (0 for 1D prediction)
        # z is altitude AGL, vz is vertical velocity
        sim_state_for_prediction = [
            0,              # y (horizontal position)
            altitude_AGL,   # z (vertical position)
            0,              # vy (horizontal velocity)
            state[5]        # vz (vertical velocity)
        ]
        
        constants_for_prediction = [
            9.81, 
            env.density(altitude_ASL), 
            freya.total_mass(time), 
            dc.clean_area
        ]

        try:
            predicted_apogee_current_AGL = apogee_predictor_module.integrate_ballistic(
                time, 
                sim_state_for_prediction, 
                constants_for_prediction, 
                duration=60,
                dt=0.1,
                deployment_level=air_brakes_object_ref.deployment_level
            )
            predicted_apogee_current = predicted_apogee_current_AGL
            
            airbrake_command = pid_apogee_controller.update(predicted_apogee_current, time)
            
            max_rate_change = RATE_LIMIT_PER_SECOND / sampling_rate
            prev_deployment = air_brakes_object_ref.deployment_level
            limited_command = np.clip(
                airbrake_command,
                prev_deployment - max_rate_change,
                prev_deployment + max_rate_change
            )
            
            air_brakes_object_ref.deployment_level = limited_command
            
           
            
            ctrl_time_hist.append(time)
            ctrl_pred_hist.append(predicted_apogee_current)
            ctrl_error_hist.append(predicted_apogee_current - TARGET_APOGEE)
            ctrl_deploy_hist.append(air_brakes_object_ref.deployment_level)

        except Exception as e:
            print(f"Error in apogee prediction at t={time:.2f}s: {e}")
            pid_apogee_controller.reset()
            air_brakes_object_ref.deployment_level = 0.0 
    else:
        pid_apogee_controller.reset() 
        air_brakes_object_ref.deployment_level = 0.0

    current_airspeed = np.linalg.norm([state[3] - env.wind_velocity_x(altitude_ASL),
                                       state[4] - env.wind_velocity_y(altitude_ASL),
                                       state[5]])
    mach_number = current_airspeed / env.speed_of_sound(altitude_ASL) if env.speed_of_sound(altitude_ASL) != 0 else 0
    current_cd = air_brakes_object_ref.drag_coefficient(air_brakes_object_ref.deployment_level, mach_number)
    
    return (
        time,
        air_brakes_object_ref.deployment_level,
        current_cd
    )
