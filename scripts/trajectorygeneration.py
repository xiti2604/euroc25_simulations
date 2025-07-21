import numpy as np
import datetime
from rocketpy import (
    Environment,
    Rocket,
    Flight,
    Fluid,
    CylindricalTank,
    MassFlowRateBasedTank,
    HybridMotor,
)


ground_level = 95 # Ground level in meters
tomorrow = datetime.date.today() + datetime.timedelta(days=1)


# Create an nvironment object
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




# Define fluids
liquid_nox = Fluid(name="lNOx", density=786.6)
vapour_nox = Fluid(name="gNOx", density=159.4)

# Define tank geometry
tank_radius = 150 / 2000
tank_length = 0.7
tank_shape = CylindricalTank(tank_radius, tank_length)

# Define motor parameters
burn_time = 7
nox_mass = 7.84
mass_flow = nox_mass / burn_time
isp = 213
grain_length = 0.304
nozzle_length = 0.05185
plumbing_length = 0.4859
post_cc_length = 0.0605

# Create oxidizer tank
oxidizer_tank = MassFlowRateBasedTank(
    name="oxidizer tank",
    geometry=tank_shape,
    flux_time=burn_time - 0.01,
    initial_liquid_mass=nox_mass,
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=mass_flow,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=liquid_nox,
    gas=vapour_nox,
)

# Create another tank
Filling_tank = MassFlowRateBasedTank(
    name="Filling tank",
    geometry=tank_shape,
    flux_time=burn_time - 0.01,
    initial_liquid_mass=nox_mass / 2,
    initial_gas_mass=0,
    liquid_mass_flow_rate_in=0,
    liquid_mass_flow_rate_out=0,
    gas_mass_flow_rate_in=0,
    gas_mass_flow_rate_out=0,
    liquid=liquid_nox,
    gas=vapour_nox,
)

# Create Hybrid Motor "fafnir"
fafnir = HybridMotor(
    thrust_source=isp * 9.8 * mass_flow,
    dry_mass=27.243,
    dry_inertia=(7.65, 7.65, 0.0845),
    nozzle_radius=70 / 2000,
    grain_number=1,
    grain_separation=0,
    grain_outer_radius=54.875 / 1000,
    grain_initial_inner_radius=38.90 / 2000,
    grain_initial_height=0.304,
    grain_density=1.1,
    grains_center_of_mass_position=grain_length / 2 + nozzle_length + post_cc_length,
    center_of_dry_mass_position=0.9956,
    nozzle_position=0,
    burn_time=burn_time,
    throat_radius=15.875 / 2000,
)
fafnir.add_tank(
    tank=oxidizer_tank,
    position=post_cc_length + plumbing_length + grain_length + nozzle_length + tank_length / 2,
)
fafnir.add_tank(
    tank=Filling_tank,
    position=post_cc_length + plumbing_length + grain_length + nozzle_length + tank_length * 1.5,
)

# ====================================================
# Airbrake System Definition
# ====================================================
BRAKES_DISABLED = False
FRONTAL_AREA = 0.01119008 * 2

def airbrakes_controller(
    time, sampling_rate, state, state_history, observed_variables, air_brakes_object_ref
):
   
    if BRAKES_DISABLED:
        air_brakes_object_ref.deployment_level = 0.0
    else:
        altitude_ASL = state[2]
        altitude_AGL = altitude_ASL - env.elevation

        if altitude_AGL > 1500:
            air_brakes_object_ref.deployment_level = 1  # Fully deploy
        else:
            air_brakes_object_ref.deployment_level = 0  # Keep retracted

    altitude_ASL = state[2]
    vx, vy, vz = state[3], state[4], state[5]
    wind_x, wind_y = env.wind_velocity_x(altitude_ASL), env.wind_velocity_y(
        altitude_ASL
    )
    free_stream_speed = ((vx - wind_x) ** 2 + (vy - wind_y) ** 2 + (vz) ** 2) ** 0.5
    mach_number = free_stream_speed / env.speed_of_sound(altitude_ASL)
    current_cd = air_brakes_object_ref.drag_coefficient(
        air_brakes_object_ref.deployment_level, mach_number
    )
    return (time, air_brakes_object_ref.deployment_level, current_cd)


# ====================================================
# Trajectory Generation Function
# ====================================================
def find_optimal_trajectories(
    mass_range,
    inclination_range,
    target_apogee,
    apogee_tolerance,
    min_rail_exit_velocity,
    min_rail_exit_stability,
):
    
    print("\nStarting parameter sweep to find optimal trajectories...")
    print("Criteria:")
    print(f"  - Apogee: {target_apogee} ± {apogee_tolerance} m")
    print(f"  - Min Rail Exit Velocity: {min_rail_exit_velocity} m/s")
    print(f"  - Min Rail Exit Stability: {min_rail_exit_stability} cals")
    print("-" * 30)

    successful_runs = []
    
    def create_rocket(mass_without_motor):
        rocket = Rocket(
            radius=0.077,
            mass=mass_without_motor,
            inertia=(13, 13, 0.0506),
            power_off_drag="clean_drag_curve.csv",
            power_on_drag="clean_drag_curve.csv",
            center_of_mass_without_motor=2.605 + 0.7,
            coordinate_system_orientation="tail_to_nose",
        )
        rocket.add_motor(fafnir, 0.002)
        rocket.add_nose(length=0.26, kind="Von Karman", position=3.856)
        rocket.add_trapezoidal_fins(
            4, root_chord=0.2, tip_chord=0.1, span=0.1, position=0.3, sweep_angle=25
        )
        rocket.add_air_brakes(
            drag_coefficient_curve="total_drag_override.csv",
            controller_function=airbrakes_controller,
            sampling_rate=100,
            reference_area=FRONTAL_AREA,
            clamp=True,
            initial_observed_variables=[0, 0, 0],
            override_rocket_drag=True,
            name="TableBasedAirBrakes",
            controller_name="AltitudeDeployController",
        )
        return rocket

    for mass in np.arange(mass_range[0], mass_range[1] + mass_range[2], mass_range[2]):
        for inclination in np.arange(inclination_range[0], inclination_range[1] + inclination_range[2], inclination_range[2]):
            print(f"Simulating with: Mass = {mass:.2f} kg, Inclination = {inclination}°")

            current_rocket = create_rocket(mass)
            flight = Flight(
                rocket=current_rocket,
                environment=env,
                rail_length=12,
                inclination=inclination,
                heading=0,
                terminate_on_apogee=True,
            )

            apogee = flight.apogee - env.elevation
            rail_exit_velocity = flight.out_of_rail_velocity
            time_at_rail_exit = flight.out_of_rail_time
            mach_at_rail_exit = flight.mach_number(time_at_rail_exit)
            stability_margin = current_rocket.stability_margin(mach_at_rail_exit, time_at_rail_exit)

            if (
                abs(apogee - target_apogee) <= apogee_tolerance
                and rail_exit_velocity >= min_rail_exit_velocity
                and stability_margin >= min_rail_exit_stability
            ):
                print(" Success! Trajectory meets all criteria.")
                successful_runs.append({
                    "mass": mass,
                    "inclination": inclination,
                    "apogee": apogee,
                    "rail_exit_velocity": rail_exit_velocity,
                    "stability_margin": stability_margin,
                })
            else:
                print(" Failed. Criteria not met.")

    print("\n--- Parameter change Results ---")
    if successful_runs:
        print(f"Found {len(successful_runs)} successful trajectory configurations:")
        for i, run in enumerate(successful_runs):
            print(f"\n--- Configuration {i+1} ---")
            print(f"  Rocket Mass (w/o motor): {run['mass']:.2f} kg")
            print(f"  Launch Inclination: {run['inclination']}°")
            print(f"  Apogee (AGL): {run['apogee']:.2f} m")
            print(f"  Rail Exit Velocity: {run['rail_exit_velocity']:.2f} m/s")
            print(f"  Stability Margin: {run['stability_margin']:.2f} cals")
    else:
        print("No simulation runs met all the specified criteria.")
    print("---------------------------------")
    return successful_runs


# Main execution block

def main():
    
    mass_range = (10, 14, 1)  
    inclination_range = (70, 85, 1)  

    
    find_optimal_trajectories(
        mass_range=mass_range,
        inclination_range=inclination_range,
        target_apogee=2800,
        apogee_tolerance=100,
        min_rail_exit_velocity=30,
        min_rail_exit_stability=1.5,
    )

if __name__ == "__main__":
    main()
