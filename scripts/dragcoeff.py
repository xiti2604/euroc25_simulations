import pandas as pd
import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d
import matplotlib.pyplot as plt

# This multiplier, N, allows us to scale the airbrake effectiveness
# for Monte Carlo simulations. N=1 is the baseline from CFD.

DRAG_MULTIPLIER = 1

def set_drag_multiplier(N: float):
    """
    Sets the global drag multiplier for uncertainty analysis. This function
    will be called by the main script before each simulation run.
    """
    global DRAG_MULTIPLIER
    DRAG_MULTIPLIER = N


data = {
    'Mach': [0.3, 0.6, 0.9, 1.2, 1.5],
    'Clean': [0.4183, 0.404, 0.4417, 0.6428, 0.5814],
    'AB=5mm': [0.4030, 0.3886, 0.4181, 0.6252, 0.5668],
    'AB=10mm': [0.4164, 0.4049, 0.4292, 0.6285, 0.5672],
    'AB=15mm': [0.4408, 0.4228, 0.4419, 0.6358, 0.5725],
    'AB=20mm': [0.4536, 0.4506, 0.4624, 0.6495, 0.5795],
    'AB=26mm': [0.4729, 0.4745, 0.4770, 0.6704, 0.5864]
}

df = pd.DataFrame(data)

# Use the full frontal area of the rocket. This is twice the half-body area.
clean_area = 0.01119008 * 2 

# Use the full frontal area for each deployment level.
area_map = { 0: 0.01119008 * 2,
             5: 0.01196928 * 2,
            10: 0.01274567 * 2,
            15: 0.01352337 * 2,
            20: 0.01430339 * 2,
            26: 0.01508393 * 2 }

area_deployments = np.array(list(area_map.keys()))
area_values = np.array(list(area_map.values()))
area_interpolator = interp1d(area_deployments, area_values, kind='linear')

A_BRAKE_MAX = area_map[26] - clean_area

df["AB=0mm"] = df["Clean"]
airbrake_columns = ["AB=0mm","AB=5mm","AB=10mm",
                    "AB=15mm","AB=20mm","AB=26mm"]

deployment_mm = np.array([0, 5, 10, 15, 20, 26])
mach_numbers  = df["Mach"].values

delta_grid = np.zeros((len(mach_numbers), len(deployment_mm)))

for i, M in enumerate(mach_numbers):
    Cd_clean = df.loc[i, "Clean"]
    for j, mm in enumerate(deployment_mm):
        Cd_d = df.loc[i, airbrake_columns[j]]
        A_d  = area_map[mm]
        delta_grid[i, j] = (Cd_d*A_d - Cd_clean*clean_area) / A_BRAKE_MAX

interp_inc = RegularGridInterpolator(
    (mach_numbers, deployment_mm),
    delta_grid,
    bounds_error=False,
    fill_value=None
)

def get_airbrake_cd_increment(mach_number: float,
                              deployment_frac: float) -> float:
    deploy_mm = np.clip(deployment_frac, 0.0, 1.0) * 26.0
    return float(interp_inc((mach_number, deploy_mm)))

total_grid = df[airbrake_columns].values
interp_tot = RegularGridInterpolator(
    (mach_numbers, deployment_mm),
    total_grid,
    bounds_error=False,
    fill_value=None
)

def get_total_cd(mach_number: float, deployment_mm: float) -> float:
    """
    Calculates the TOTAL drag coefficient of the rocket, applying
    the global DRAG_MULTIPLIER to the airbrakes' contribution.
    This is used by the apogee predictor.
    """
    deployment_mm = np.clip(deployment_mm, 0.0, 26.0)
    mach_number = max(mach_number, 0.0) # Prevent negative mach

    # Get the original coefficients from the base CFD model
    cd_total_orig = float(interp_tot((mach_number, deployment_mm)))
    cd_clean = get_clean_cd(mach_number)

    # Get the areas needed for the formula
    A_clean = clean_area
    A_total = float(area_interpolator(deployment_mm))

    if A_total < 1e-9:
        return cd_total_orig

    N = DRAG_MULTIPLIER
    area_ratio = A_clean / A_total
    
    cd_total_new = N * cd_total_orig + (1 - N) * cd_clean * area_ratio
    
    return cd_total_new

def get_total_cd_for_override(mach_number: float, deployment_mm: float) -> float:
    """
    Calculates the TOTAL drag coefficient of the rocket, NORMALIZED to the
    clean rocket's reference area. This is for use with a constant ref area.
    """
    A_clean = clean_area
    A_total = float(area_interpolator(deployment_mm))

    # Get the original (non-normalized) total Cd
    cd_original = get_total_cd(mach_number, deployment_mm)

    if A_clean < 1e-9:
        return cd_original # Avoid division by zero

    # Normalize the Cd to the clean reference area
    cd_normalized = cd_original * (A_total / A_clean)
    
    return cd_normalized

def get_clean_cd(mach_number):
    clean_interp = interp1d(mach_numbers, df['Clean'].values, 
                           kind='linear', fill_value='extrapolate')
    return float(clean_interp(mach_number))

def get_airbrake_cd(mach_number: float, deployment_mm: float) -> float:
    """
    Calculates the INCREMENTAL drag coefficient (ΔCd) of the airbrakes,
    scaled by the global DRAG_MULTIPLIER.
    This is the value RocketPy adds to the rocket's base drag.
    """
    deployment_mm = np.clip(deployment_mm, 0.0, 26.0)
    deployment_frac = deployment_mm / 26.0
    
    # Get the original incremental Cd from the base model
    original_incremental_cd = get_airbrake_cd_increment(mach_number, deployment_frac)
    
    # Apply the multiplier and return the scaled value
    return original_incremental_cd * DRAG_MULTIPLIER

def export_total_drag_for_override(filename="total_drag_override.csv"):
    """
    Generates a 3-column CSV file for the TOTAL rocket drag,
    formatted for use with RocketPy's override_rocket_drag=True feature.
    The Cd values are normalized to the rocket's clean reference area.
    """
    print(f"\nGenerating normalized total drag curve for RocketPy override: '{filename}'...")
    
    # The constant reference area we are normalizing to
    A_clean = clean_area 

    results = []
    # Iterate through all deployment levels provided in the original data
    for j, mm in enumerate(deployment_mm):
        deployment_level = mm / 26.0  # Convert mm to fractional (0-1)
        A_original = area_map[mm]

        # Iterate through all Mach numbers
        for i, mach in enumerate(mach_numbers):
            # Get the original Cd for the whole rocket from the dataframe
            Cd_original = df.loc[i, airbrake_columns[j]]
            
            # Normalize the Cd to the clean reference area
            Cd_normalized = Cd_original * (A_original / A_clean)
            
            results.append({
                'deployment_level': round(deployment_level, 4),
                'mach': round(mach, 2),
                'cd': round(Cd_normalized, 6)
            })

    df_export = pd.DataFrame(results, columns=['deployment_level', 'mach', 'cd'])
    
    try:
        df_export.to_csv(filename, index=False, sep=',')
        print(f"Successfully exported normalized total drag data to '{filename}'")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")

def plot_total_cd_at_26mm():

    mach_plot_range = np.linspace(min(mach_numbers), max(mach_numbers), 30)
    total_cd_values = [get_total_cd(mach, 26) for mach in mach_plot_range]

    plt.figure(figsize=(10, 6))
    plt.plot(mach_plot_range, total_cd_values, 'b-o', label='Interpolated Total Cd at 26mm (30 points)', linewidth=1.5, markersize=4)
    
    original_mach_points = df['Mach'].values
    original_cd_at_26mm = df['AB=26mm'].values
    plt.plot(original_mach_points, original_cd_at_26mm, 'ro', label='Original CFD Data Points (26mm)')

    plt.title('Total Drag Coefficient (Cd) vs. Mach Number (Airbrakes at 26mm)')
    plt.xlabel('Mach Number')
    plt.ylabel('Total Drag Coefficient (Cd)')
    plt.grid(True)
    plt.legend()
    print("Displaying plot...")
    plt.show()

def export_cd_to_csv(filename="cdfinalsimulation.csv"):
    print("\nGenerating data for CSV export...")
    deployment_levels = np.linspace(0, 1, 11)
    mach_grid = np.linspace(0.3, 1.5, 13)

    results = []

    for level in deployment_levels:
        for mach in mach_grid:
            deploy_mm = level * 26.0
            cd_inc = get_airbrake_cd(mach, deploy_mm)
            
            results.append({
                'deployment_level': round(level, 2),
                'mach': round(mach, 2),
                'cd': round(cd_inc, 5)
            })

    df_export = pd.DataFrame(results, columns=['deployment_level', 'mach', 'cd'])
    
    try:
        df_export.to_csv(filename, index=False, sep=',')
        print(f"Successfully exported drag coefficient data to '{filename}'")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def export_clean_drag_curve_to_csv(filename="clean_drag_curve.csv"):
    """
    Generates a 2-column CSV file for the clean configuration drag curve,
    formatted for use with RocketPy.
    Column 1: Mach number
    Column 2: Drag Coefficient (Cd)
    """
    print(f"\nGenerating clean configuration drag curve for RocketPy: '{filename}'...")
    # Generate a range of Mach numbers from 0.0 to 2.0
    mach_grid = np.linspace(0.0, 2.0, 41)

    results = []
    for mach in mach_grid:
        cd_clean = get_clean_cd(mach)
        # Format for 2-column CSV: [Mach, Cd]
        results.append([round(mach, 4), round(cd_clean, 6)])

    df_export = pd.DataFrame(results)

    try:
        df_export.to_csv(filename, header=False, index=False, sep=',')
        print(f"Successfully exported clean drag curve to '{filename}'")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def export_ab26mm_drag_curve_to_csv(filename="ab26mm_drag_curve.csv"):
    """
    Generates a 2-column CSV file for the drag curve with airbrakes fully
    deployed (26mm), formatted for use with RocketPy.
    Column 1: Mach number
    Column 2: Drag Coefficient (Cd)
    """
    # Generate a range of Mach numbers from 0.0 to 2.0
    mach_grid = np.linspace(0.0, 2.0, 41)

    results = []
    for mach in mach_grid:
        # Get total Cd with airbrakes fully deployed at 26mm
        cd_26mm = get_total_cd(mach, 26.0)
        # Format for 2-column CSV: [Mach, Cd]
        results.append([round(mach, 4), round(cd_26mm, 6)])

    df_export = pd.DataFrame(results)

    try:
        df_export.to_csv(filename, header=False, index=False, sep=',')
        print(f"Successfully exported 26mm airbrake drag curve to '{filename}'")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def export_normalized_ab26mm_drag_curve_to_csv(rocket_radius, filename="normalized_ab26mm_drag_curve.csv"):
    """
    Generates a 2-column CSV file for the 26mm drag curve, but with Cd
    values NORMALIZED to the rocket's radius-based reference area.
    This is the correct file to use in the Rocket.power_on/off_drag fields.
    """
    print(f"\nGenerating NORMALIZED 26mm drag curve for RocketPy: '{filename}'...")
    
    # The reference area that RocketPy will use for its calculation
    rocketpy_ref_area = np.pi * rocket_radius**2
    # The actual physical area of the rocket with brakes fully deployed
    ab26_physical_area = area_map[26]
    
    # The ratio we need to apply to every Cd value
    normalization_factor = ab26_physical_area / rocketpy_ref_area

    print(f"  RocketPy Reference Area (from radius): {rocketpy_ref_area:.6f} m²")
    print(f"  CFD Physical Area (at 26mm): {ab26_physical_area:.6f} m²")
    print(f"  Normalization Factor: {normalization_factor:.4f}")

    mach_grid = np.linspace(0.0, 2.0, 41)
    results = []
    for mach in mach_grid:
        # Get the original total Cd from the CFD data
        original_cd = get_total_cd(mach, 26.0)
        # Apply the normalization
        normalized_cd = original_cd * normalization_factor
        results.append([round(mach, 4), round(normalized_cd, 6)])

    df_export = pd.DataFrame(results)
    
    try:
        df_export.to_csv(filename, header=False, index=False, sep=',')
        print(f"Successfully exported NORMALIZED 26mm drag curve to '{filename}'")
    except Exception as e:
        print(f"Error exporting to CSV: {e}")


def main():
    while True:
        try:
            mach      = float(input("Enter Mach number: "))
            deploy_mm = float(input("Enter deployment level in mm (0-26): "))

            cd_inc = get_airbrake_cd(mach, deploy_mm)
            cd_tot = get_total_cd(mach, deploy_mm)

            print(f"Total drag coefficient  Cd  = {cd_tot:.3f}")
            print(f"Incremental drag coeff ΔCd = {cd_inc:.3f}")

            if input("Test another value? (yes/no): ").strip().lower() != "yes":
                break
        except ValueError:
            print("Invalid input. Please enter numeric values.")
    
    if input("\nExport incremental Cd table to CSV file? (yes/no): ").strip().lower() == "yes":
        export_cd_to_csv("cdfinalsimulation.csv")

    if input("\nExport clean config drag curve (for RocketPy)? (yes/no): ").strip().lower() == "yes":
        export_clean_drag_curve_to_csv("clean_drag_curve.csv")

    if input("\nExport 26mm airbrake drag curve (for RocketPy)? (yes/no): ").strip().lower() == "yes":
        export_ab26mm_drag_curve_to_csv("ab26mm_drag_curve.csv")

    if input("\nExport NORMALIZED TOTAL drag curve for override? (yes/no): ").strip().lower() == "yes":
        export_total_drag_for_override("total_drag_override.csv")

    if input("\nExport NORMALIZED 26mm drag curve (for RocketPy baseline)? (yes/no): ").strip().lower() == "yes":
        export_normalized_ab26mm_drag_curve_to_csv(rocket_radius=0.077)

    if input("\nGenerate plot for Total Cd at 26mm deployment? (yes/no): ").strip().lower() == "yes":
        plot_total_cd_at_26mm()
        
    print("Exiting.")

if __name__ == "__main__":
    main()

"""Time: 20.39s | Incremental Airbrake Cd: 0.6248 |deployment: 1.00
Mach number: 0.27

Time: 9.89s | Incremental Airbrake Cd: 0.6770 |deployment: 1.00
Mach number: 0.60

Time: 8.05s | Incremental Airbrake Cd: 0.6495 |deployment: 1.00
Mach number: 0.68


Time: 7.30s | Incremental Airbrake Cd: 0.6369 |deployment: 1.00
Mach number: 0.72

Time: 15.18s | Incremental Airbrake Cd: 0.6464 |deployment: 1.00
Mach number: 0.41

Time: 24.59s | Incremental Airbrake Cd: 0.6154 |deployment: 1.00
Mach number: 0.21
"""