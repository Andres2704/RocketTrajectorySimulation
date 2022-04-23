# ------------------------------------------------------------------------------
# Rocket Trajectory Simulation - Tau Rocket Team
# Author: Andres Benoit
# You can contact me at: andres.benoit@acad.ufsm.br
# ------------------------------------------------------------------------------
import numpy as np
from dof6_setup import *
# from dof3_setup import *
launch_vehicle = rocket()
# ------------------------------------------------------------------------------
# ----------------------------INITIAL CONDITIONS--------------------------------
# ------------------------------------------------------------------------------
launch_vehicle.initial(
    rail_length = 6,                    # Launch Rail Length [m]
    latitude    = -22.5,                # Initial Latitude [deg]
    longitude   = -50,                  # Initial Longitude [deg]
    azimut      = 45,                   # Azimute angle [deg]
    elevation   = 85,                   # Elevation angle [deg]
    phi0        = 0,                    # Roll angle [deg]
    u0          = 0.001,                # x-velocity in body-frame [m/s]
    v0          = 0,                    # y-velocity in body-frame [m/s]
    w0          = 0,                    # z-velocity in body-frame [m/s]
    Va          = 0.001,                # Aerodynamic velocity [m/s]
    p0          = 0,                    # x-angular velocity in body-frame [rad/s]
    q0          = 0,                    # y-angular velocity in body-frame [rad/s]
    r0          = 0,                    # z-angular velocity in body-frame [rad/s]
    x0          = 0,                    # x-position [m]
    y0          = 0,                    # y-position [m]
    z0          = 0,                    # z-position [m]
    alpha       = 0,                    # Angle of attack [deg]
    beta        = 0,                    # Side-slip angle [deg]
    delta       = 0,                    # Fin missalignment [deg]
    init_mass   = 28.497,               # Initial rocket mass (with propellant)
)

# ------------------------------------------------------------------------------
# ----------------------------SIMULATION SETUP----------------------------------
# ------------------------------------------------------------------------------
sim = simulation(
    step_time   = 0.0005,               # Step-time [s]
    tolerance   = 1E-6,                 # Tolerance
    add_wind    = True,                 # Add wind in the simulation
    add_miss    = True,                 # Add motor missalignment
    add_drogue  = True,                 # Add Drogue Chute
    add_main    = True,                 # Add Main Chute
    plot_3dtraj = True,                 # Plot 3D Trajectory
    plot_result = True,                 # Plot the results
    save_result = True,                 # Save the results in a sheet file
    save_name   = 'output.xlsx'         # Save name
)

# ------------------------------------------------------------------------------
# ------------------------MOTOR MISSALIGNMENT-----------------------------------
# ------------------------------------------------------------------------------
# Adding motor missalignment (if false at setup the values will not affect the simulation)
sim.motormissalign(
    eps         = 0,                    # lateral angle [deg]
    mu          = 0,                    # vertical angle [deg]
)

# ------------------------------------------------------------------------------
# -----------------------------DROGUE CHUTE-------------------------------------
# ------------------------------------------------------------------------------
# Adding drogue chute (if false at setup the values will not affect the simulation)
sim.droguechute(
    Cd          = 0.75,        # Drogue drag coefficient [-]
    diameter    = 1.40,        # Drogue diameter [m]
    open_time   = 1.00,        # time to open drogue chute after apogee [s]
)

# ------------------------------------------------------------------------------
# ------------------------------MAIN CHUTE--------------------------------------
# ------------------------------------------------------------------------------
# Adding main chute (if false at setup the values will not affect the simulation)
sim.mainchute(
    Cd          = 0.75,        # Main drag coefficient [-]
    diameter    = 3.00,        # Main diameter [m]
    altitude    = 500,         # Altitude to open the main chute after apogee [m]
)

# ------------------------------------------------------------------------------
# -------------------------FUSELAGE DEFINITION----------------------------------
# ------------------------------------------------------------------------------
launch_vehicle.fuselage(
    Fus_length  = 2.1,          # Fuselage length [m]
    ex_diameter = 0.126,        # External diameter [m]
    in_diameter = 0.120,        # Internal diameter [m]
    bs_diameter = 0.126,        # Base diameter [m]
    Fus_dens    = 1850,         # Fuselage material density [kg/m³]
)

# ------------------------------------------------------------------------------
# -------------------------NOSECONE DEFINITION----------------------------------
# ------------------------------------------------------------------------------
launch_vehicle.nosecone(
    Nose_length = 0.30,         # Nosecone length [m]
    Nose_mass   = 0.866,        # Nosecone mass [kg]
    Nose_dens   = 1850,         # Nosecone material density [kg/m³]
)

# ------------------------------------------------------------------------------
# ---------------------------FINS DEFINITION------------------------------------
# ------------------------------------------------------------------------------
launch_vehicle.fins(
    Number      = 4,           # Number of total fins [-]
    Tip_chord   = 0.06,        # Chord at the tip [m]
    Root_chord  = 0.21,        # Chord at the root [m]
    semi_span   = 0.12,        # Fin semi-span [m]
    max_tick    = 0.0231,      # Max tickness [m]
    sweep       = 51.3,        # Sweep angle [deg]
)

# ------------------------------------------------------------------------------
# ------------------------LOADING PROPULSION DATA-------------------------------
# ------------------------------------------------------------------------------
# File witch contains the propulsion data
# In csv form containting [time, Mass flux, Thrust] for 'solid' rocket motor
# and [time, Fuel mass flux, Oxidizer mass flux, Thrust] for 'hybrid' rocket motor
launch_vehicle.propulsion(
    file_name   = 'prop_data.csv',      # Filename in the same folder than this
    delimiter   = ';',                  # Delimiter in the csv file
    type        = 'hybrid'              # Engine type ('solid' or 'hybrid')
)

# ------------------------------------------------------------------------------
# ------------------------LOADING STRUCTURAL DATA-------------------------------
# ------------------------------------------------------------------------------
# File witch contains the propulsion data
# In csv form containting [Xcg, Ixx, Iyy, Izz], be carefull with the origin
# Of the calculations, the solver implements the rocket aft as origin
launch_vehicle.structural(
    file_name   = 'struct_data.csv',    # Filename in the same folder than this
    delimiter   = ';',                  # Delimiter in the csv file
)

# ------------------------------------------------------------------------------
# ------------------------------SOLVING EOM-------------------------------------
# ------------------------------------------------------------------------------
result = six_dof(sim, launch_vehicle)              # Calling the solution solver
