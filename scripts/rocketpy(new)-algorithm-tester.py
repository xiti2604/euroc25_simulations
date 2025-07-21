import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.integrate import ode, simpson
from scipy.interpolate import interp1d, RegularGridInterpolator
from rocketpy import Fluid, CylindricalTank, MassFlowRateBasedTank, HybridMotor
from rocketpy import Environment, Rocket, Flight
#import position_kf
#import ekf_barometer_apogee_predict
#import PID
from rocketpy.utilities import apogee_by_mass
import datetime



try:
    import dragcoeff as dc
except ImportError:
    print("ERROR: Could not import dragcoeff.py")
    dc = None 



AIRBRAKE_MAX_PHYSICAL_AREA = 0.0082 # m²  (four-pad frontal area at 26 mm)


# ====================================================
#Drag coefficient functions (for rocket body, not airbrakes)
# ====================================================

# Deprecated Function from before Airbrakes CFD data was available
def Cd_function_mach():
   
    simulation = pd.read_excel('cd_simulation.xlsx')
    velocity_vals = simulation['Mach'].tolist()
    CD_vals = simulation['CD'].tolist()
    return interp1d(velocity_vals, CD_vals, kind='linear', fill_value="extrapolate")

def airbrake_drag_from_model(deployment, mach):
    """
    A clean wrapper function that gets the incremental airbrake drag
    from the centralized model in dragcoeff.py. The deployment level
    here is a fraction from 0 to 1.
    """
    deployment_mm = deployment * 26.0
    return dc.get_airbrake_cd(mach, deployment_mm)

def get_body_drag_from_cfd():
    
    if dc is None:
        print("dragcoeff.py not loaded. Using a constant drag of 0.5 for the body.")
        # Return a function that always returns 0.5
        return lambda mach: 0.5

    # Access the Mach numbers and Clean Cd values from the imported module
    mach_points = dc.mach_numbers
    cd_points = dc.df['Clean'].values

    # Create and return the interpolator function
    return interp1d(mach_points, cd_points, kind='linear', fill_value="extrapolate")



Scale_Disabled = True
if Scale_Disabled:
    cd_scaled = get_body_drag_from_cfd()
else:
    Cd_mach = get_body_drag_from_cfd()
    alpha = 1.4
    cd_scaled = lambda M: alpha * Cd_mach(M)



# As a reference with fully deployed airbrakes for comparison between clean config and full deployment
def get_airbrake_drag_from_csv(mach_number, file_path="ab26mm_drag_curve.csv"):
   
    try:
        # Read the data from the CSV file
        df = pd.read_csv(file_path, header=None, names=['Mach', 'Cd'])
        
        # Create an interpolation function
        airbrake_cd_func = interp1d(df['Mach'], df['Cd'], kind='linear', fill_value="extrapolate")
        
        # Return the interpolated value
        return float(airbrake_cd_func(mach_number))
    
    except FileNotFoundError:
        print(f"Error: Could not find airbrake drag file: {file_path}")
        return 0.0
    except Exception as e:
        print(f"Error retrieving airbrake drag coefficient: {e}")
        return 0.0




# Define fluids for the hybrid motor (liquid and gaseous nitrous oxide)
# fluids at 20°C
liquid_nox = Fluid(name="lNOx", density=786.6) # Define liquid nitrous oxide (lNOx)
vapour_nox = Fluid(name="gNOx", density=159.4) # Define gaseous nitrous oxide (gNOx)

# Define tank geometry
tank_radius = 150 / 2000 # Tank radius in meters
tank_length = 0.7 # Tank length in meters
tank_shape = CylindricalTank(tank_radius, tank_length) # Create a cylindrical tank object

# Define tank properties and motor parameters
burn_time = 7 # Burn time in seconds
nox_mass = 7.84 # Total mass of nitrous oxide in kg
ullage_mass = nox_mass * 0.15 # Ullage mass (extra space in tank)
mass_flow = nox_mass / burn_time # Mass flow rate of nitrous oxide in kg/s
isp = 213 # Specific impulse of the motor in seconds
grain_length = 0.304 # Length of the fuel grain in meters
nozzle_length = 0.05185 # Length of the nozzle in meters
plumbing_length = 0.4859 # Length of plumbing including pre-combustion chamber and topcap/injector
post_cc_length = 0.0605 # Length of post-combustion chamber in meters
pre_cc_length = 0.039 # Length of pre-combustion chamber in meters

# Create an oxidizer tank object
oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer tank", # Name of the tank
    geometry=tank_shape, # Tank geometry (cylindrical)
    flux_time=burn_time - 0.01, # Time during which mass flow occurs
    initial_liquid_mass=nox_mass, # Initial mass of liquid nitrous oxide
    initial_gas_mass=0, # Initial mass of gaseous nitrous oxide
    liquid_mass_flow_rate_in=0, # Rate of liquid nitrous oxide flowing in
    liquid_mass_flow_rate_out= mass_flow, # Rate of liquid nitrous oxide flowing out
    gas_mass_flow_rate_in=0, # Rate of gaseous nitrous oxide flowing in
    gas_mass_flow_rate_out=0, # Rate of gaseous nitrous oxide flowing out
    liquid=liquid_nox, # Fluid object for liquid nitrous oxide
    gas=vapour_nox, # Fluid object for gaseous nitrous oxide
)

Filling_tank = MassFlowRateBasedTank(
    name="Filling tank",
    geometry=tank_shape,
    flux_time=burn_time - 0.01,
    initial_liquid_mass=nox_mass / 2,  # Split initial mass between two tanks
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=0,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=liquid_nox,
    gas=vapour_nox,
)


# Create a hybrid motor object named "fafnir"
fafnir = HybridMotor(
    thrust_source = isp * 9.8 * mass_flow, # Thrust source (calculated from isp, mass flow, and gravity)
    dry_mass = 27.243, # Dry mass of the motor (excluding fuel grain...)
    dry_inertia = (7.65, 7.65, 0.0845), # Dry inertia of the motor
    nozzle_radius = 70 / 2000, # Radius of the nozzle
    grain_number = 1, # Number of fuel grains
    grain_separation = 0, # Separation between fuel grains
    grain_outer_radius = 54.875 / 1000, # Outer radius of the fuel grain
    grain_initial_inner_radius = 38.90 / 2000, # Initial inner radius of the fuel grain
    grain_initial_height = 0.304, # Initial height of the fuel grain
    grain_density = 1.1, # Density of the fuel grain
    grains_center_of_mass_position = grain_length / 2 + nozzle_length+post_cc_length, # Center of mass position of the fuel grains
    center_of_dry_mass_position = 0.9956, # Center of dry mass position of the motor
    nozzle_position = 0, # Position of the nozzle
    burn_time = burn_time, # Burn time of the motor
    throat_radius = 15.875 / 2000, # Radius of the throat
)

# Add the oxidizer tank to the motor
fafnir.add_tank(
    tank=oxidizer_tank,
    position=post_cc_length+plumbing_length + grain_length + nozzle_length + tank_length / 2)

# Add the second oxidizer tank after the first
fafnir.add_tank(
    tank=Filling_tank,
    position=post_cc_length + plumbing_length + grain_length + nozzle_length + tank_length * 1.5,  # Adjust position
)

# Draw a diagram of the motor
#fafnir.draw()
# Print all information about the motor
#fafnir.all_info()


############################################ Flight
# This section defines the flight environment and the rocket

# Define ground level
ground_level = 95 # Ground level in meters
tomorrow = datetime.date.today() + datetime.timedelta(days=10)


# Create an environment object
env = Environment(
    latitude=38.72, # Latitude of the launch site
    longitude=-9.15, # Longitude of the launch site
    elevation=ground_level, # Elevation of the launch site
    #date=(2025, 10, 12, 12), # Date and time of the launch
    date=(tomorrow.year, tomorrow.month, tomorrow.day, 12), # Date and time of the launch
)

# Set the atmospheric model
#env.set_atmospheric_model("custom_atmosphere", wind_u=0, wind_v=-10) # Custom atmosphere with wind
env.set_atmospheric_model(type="Forecast", file="GFS") # Custom atmosphere with wind



# Create a rocket object named "freya"
freya = Rocket(
   
    radius=0.077, # Radius of the rocket
    mass= 11, # Mass of the rocket (without motor)
    inertia=(13, 13, 0.0506), # Inertia of the rocket
    power_off_drag="clean_drag_curve.csv", # Drag coefficient when the motor is off
    power_on_drag="clean_drag_curve.csv", # Drag coefficient when the motor is on
    center_of_mass_without_motor=2.605+0.7, # Center of mass position without motor
    coordinate_system_orientation="tail_to_nose" # Orientation of the coordinate system
)

# Add the motor to the rocket
freya.add_motor(fafnir, 0.002) 

# Add a nose cone to the rocket
freya.add_nose(
    length=0.26, # Length of the nose cone
    kind="Von Karman", # Shape of the nose cone
    position=3.856, # Position of the nose cone
)

# Add fins to the rocket
fins = freya.add_trapezoidal_fins(
    4, # Number of fins
    root_chord=0.2, # Root chord length of the fins
    tip_chord=0.1, # Tip chord length of the fins
    span=0.1, # Span of the fins
    position=0.3, # Position of the fins
    sweep_angle=25 # Sweep angle of the fins
)
# Define spill radius for parachutes
spill_radius = 0.5 / 2 # Radius of the spill hole for parachutes, 0.45

# Add a reefed parachute to the rocket
reefed_cd = 0.7 # Drag coefficient of the reefed parachute
reefed_radius = 1.5 / 2 # Radius of the reefed parachute
freya.add_parachute('Reefed', # Name of the parachute
                    cd_s=reefed_cd * math.pi * (reefed_radius ** 2 - spill_radius ** 2), # Effective drag area of the parachute
                    trigger="apogee", # Trigger for parachute deployment (apogee)
                    lag=3 # Delay in seconds after trigger before deployment
)

# Add a main parachute to the rocket
main_cd = 1.1 # Drag coefficient of the main parachute used 1.5 before
#main_radius = 3.8 / 2 # Radius of the main parachute
main_radius = 3 / 2 # Radius of the main parachute, 2.43
freya.add_parachute('Main', # Name of the parachute
                    cd_s=main_cd * math.pi * (main_radius ** 2 - spill_radius ** 2), # Effective drag area of the parachute
                    trigger=300 # Trigger for parachute deployment (altitude in meters)
)


BRAKES_DISABLED = False

def airbrakes_controller(
    time, sampling_rate, state, state_history, observed_variables, air_brakes_object_ref
):
        if BRAKES_DISABLED:
            air_brakes_object_ref.deployment_level = 0.0
        else:
            altitude_ASL = state[2]
            altitude_AGL = altitude_ASL - env.elevation
            motor_burn_end_time = fafnir.burn_out_time
            is_after_burnout = time > motor_burn_end_time

            if  altitude_AGL > 1500:
                air_brakes_object_ref.deployment_level = 1  # Fully deploy
            else:
                air_brakes_object_ref.deployment_level = 0  # Keep retracted

        # This part is needed for both controllers to log data correctly
        altitude_ASL = state[2]
        vx, vy, vz = state[3], state[4], state[5]
        wind_x, wind_y = env.wind_velocity_x(altitude_ASL), env.wind_velocity_y(altitude_ASL)
        free_stream_speed = ((vx - wind_x) ** 2 + (vy - wind_y) ** 2 + (vz) ** 2) ** 0.5
        mach_number = free_stream_speed / env.speed_of_sound(altitude_ASL)
        current_cd = air_brakes_object_ref.drag_coefficient(
            air_brakes_object_ref.deployment_level, mach_number
        )

        # Print the current time and calculated incremental drag coefficient
        #print(f"Time: {time:.2f}s | Incremental Airbrake Cd: {current_cd:.4f} |deployment: {air_brakes_object_ref.deployment_level:.2f} ")
        #print(f"Mach number: {mach_number:.2f}")
        

        return (
            time,
            air_brakes_object_ref.deployment_level,
            current_cd,
        )


FRONTAL_AREA = 0.01119008 * 2


air_brakes_system = freya.add_air_brakes(
        drag_coefficient_curve= "total_drag_override.csv",
        controller_function=airbrakes_controller,
        sampling_rate=100,
        reference_area=FRONTAL_AREA,
        clamp=True,
        initial_observed_variables=[0, 0, 0],
        override_rocket_drag=True,
        name="TableBasedAirBrakes",
        controller_name="AltitudeDeployController"
    )




# Draw a diagram of the rocket
#freya.draw()

# Create a flight object named "test_flight"
test_flight = Flight(
    rocket=freya, # Rocket object to be used in the simulation
    environment=env, # Environment object to be used in the simulation
    rail_length=12, # Length of the launch rail
    inclination=85, # Launch inclination angle in degrees
    heading=0, # Launch heading angle in degrees
    time_overshoot=False,

    terminate_on_apogee=True
)

# Print all information about the motor, rocket, and flight
"""fafnir.all_info()
freya.all_info()
test_flight.all_info()"""
#freya.all_info()
burn_time = fafnir.burn_time;
print(burn_time[1])
print(f"test_flight.altitude(burn_time): {test_flight.altitude(burn_time)[1]:.2f}")
print(f"velocity at the end of the rail: {test_flight.out_of_rail_velocity:.2f}")
Drag = test_flight.aerodynamic_drag(burn_time)[1]
print(f"Drag at the end of the rail: {Drag:.2f}")

for i in test_flight.time:
    print(f"time:{i:.2f} - Drag:{test_flight.aerodynamic_drag(i):.2f}")


position_z= test_flight.solution_array[:, 3]
velocity_z = test_flight.solution_array[:, 6]
time = test_flight.solution_array[:, 0]

plt.plot(time, position_z)
plt.plot(time, velocity_z)
plt.show()



angle_of_attack_at_burnout = test_flight.angle_of_attack(burn_time[1])
print(f"Angle of Attack at burnout: {angle_of_attack_at_burnout:.3f} degrees")

test_flight.prints.out_of_rail_conditions()
test_flight.prints.burn_out_conditions()
test_flight.prints.maximum_values()

"""for t, z in zip(time, position_z):
    print(f"time:{t:.2f} - position_z:{z:.2f}")"""


"""result = apogee_by_mass(
    flight=test_flight, min_mass=5, max_mass=20, points=10, plot=True
)
print(f"Type of apogee_by_mass result: {type(result)}")


target_mass = 15  # kg
predicted_apogee = result(target_mass)"""

#print(f"\nPredicted apogee for a mass of {target_mass} kg is: {predicted_apogee:.2f} m AGL")

""""
print(type(test_flight.mach_number(7)))
print(test_flight.mach_number(7))
for t, macho in zip(time, test_flight.mach_number(time)):
    print(f"time:{t:.2f} - mach:{macho:.2f}")
for t, macho, cd in zip(time, test_flight.mach_number(time), cd_scaled(test_flight.mach_number(time))):
    print(f"time:{t:.2f} - mach:{macho:.2f} - cd:{cd:.4f}")"""



def plot_drag_curves(clean_file="clean_drag_curve.csv", ab26_file="ab26mm_drag_curve.csv"):
    
    try:
        # Read the data from the CSV files, assigning column names
        df_clean = pd.read_csv(clean_file, header=None, names=['Mach', 'Cd'])
        df_ab26 = pd.read_csv(ab26_file, header=None, names=['Mach', 'Cd'])

        # Create the plot
        plt.figure(figsize=(12, 8))
        plt.plot(df_clean['Mach'], df_clean['Cd'], 'b-o', label='Clean Configuration', markersize=4)
        plt.plot(df_ab26['Mach'], df_ab26['Cd'], 'g-s', label='Airbrakes at 26mm', markersize=4)
        
        # Highlight the region from Mach 0 to Mach 0.8
        plt.axvspan(0, 0.8, alpha=0.2, color='yellow', label='Mach 0-0.8 Region')

        # Add plot details
        plt.title('Drag Coefficient (Cd) vs. Mach Number Comparison')
        plt.xlabel('Mach Number')
        plt.ylabel('Drag Coefficient (Cd)')
        plt.grid(True)
        plt.legend()
        print("\nDisplaying drag curve comparison plot...")
        plt.show()

    except FileNotFoundError as e:
        print(f"\nError: Could not find required file: {e.filename}")
        print("Please generate the drag curve CSV files first before plotting.")
    except Exception as e:
        print(f"An unexpected error occurred while plotting: {e}")







# Example usage

test_flight.prints.out_of_rail_conditions()

print(test_flight.altitude(29))
print(airbrake_drag_from_model(1, 0.27))
print(airbrake_drag_from_model(1, 0.60))




# Call the function to display the plot at the end of the script execution
plot_drag_curves()


test_flight.prints.apogee_conditions()
#air_brakes_system.all_info()
time_list, deployment_level_list, drag_coefficient_list = [], [], []

obs_vars = test_flight.get_controller_observed_variables()

for time, deployment_level, drag_coefficient in obs_vars:
    time_list.append(time)
    deployment_level_list.append(deployment_level)
    drag_coefficient_list.append(drag_coefficient)

# Plot deployment level by time
plt.plot(time_list, deployment_level_list)
plt.xlabel("Time (s)")
plt.ylabel("Deployment Level")
plt.title("Deployment Level by Time")
plt.grid()
plt.show()

# Plot drag coefficient by time
plt.plot(time_list, drag_coefficient_list)
plt.xlabel("Time (s)")
plt.ylabel("Drag Coefficient")
plt.title("Drag Coefficient by Time")
plt.grid()
plt.show()





test_flight.prints.out_of_rail_conditions()

test_flight.prints.maximum_values()
test_flight.prints.burn_out_conditions()

apogee_by_mass(
    flight=test_flight, min_mass=5, max_mass=20, points=10, plot=True
    )