import sys
# sys.path.insert(0, '/home/mfayolle/Documents/PHD/04 - year 4/tudat-bundle/cmake-build-release/tudatpy')
sys.path.insert(0, '/home/mfayolle/Tudat/tudat-bundle/build/tudatpy')


# Load required standard modules
import os
import numpy as np
from matplotlib import pyplot as plt

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation, propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array

flyby_moon = "Callisto"

# Manually-defined colours for plotting
colors = {"blue": (2/255, 81/255, 158/255),
          "light_blue": (42/255, 193/255, 176/255),
          "red": (153/255, 61/255, 90/255),
          "orange": (205/255, 119/255, 66/255)}
colors_list = list(colors.values())


def get_juice_position_wrt_moon(time):
    return spice.get_body_cartesian_position_at_epoch(
        "-28", flyby_moon, "J2000", aberration_corrections="none", ephemeris_time=time)


def find_closest_approaches(lower_bound, upper_bound, threshold):

    flyby_times = []

    tolerance = 1.0
    step = 20.0

    lower_time = lower_bound
    mid_time = lower_bound + step
    upper_time = lower_bound + 2.0 * step

    while upper_time <= upper_bound:
        upper_value = np.linalg.norm(get_juice_position_wrt_moon(upper_time))
        mid_value = np.linalg.norm(get_juice_position_wrt_moon(mid_time))
        lower_value = np.linalg.norm(get_juice_position_wrt_moon(lower_time))

        if (upper_value - mid_value) > 0 and (mid_value - lower_value) < 0:

            current_lower_time = lower_time
            current_upper_time = upper_time
            current_mid_time = (current_lower_time+current_upper_time)/2.0
            current_test_time = current_mid_time
            counter = 0

            while np.abs(current_mid_time - current_test_time) > tolerance:

                current_test_time = current_mid_time

                current_lower_distance = np.linalg.norm(get_juice_position_wrt_moon(current_lower_time))
                current_upper_distance = np.linalg.norm(get_juice_position_wrt_moon(current_upper_time))
                current_test_distance = np.linalg.norm(get_juice_position_wrt_moon(current_test_time))

                sign_upper_derivative = np.sign((current_upper_distance - np.linalg.norm(get_juice_position_wrt_moon(current_upper_time - tolerance))) / tolerance)
                sign_lower_derivative = np.sign((current_lower_distance - np.linalg.norm(get_juice_position_wrt_moon(current_lower_time - tolerance))) / tolerance)
                sign_test_derivative = np.sign((current_test_distance - np.linalg.norm(get_juice_position_wrt_moon(current_test_time - tolerance))) / tolerance)

                if sign_upper_derivative > 0 and sign_test_derivative < 0:
                    current_mid_time = (current_upper_time+current_test_time)/2.0
                    current_lower_time = current_test_time
                elif sign_lower_derivative < 0 and sign_test_derivative > 0:
                    current_mid_time = (current_test_time + current_lower_time)/2.0
                    current_upper_bound = current_test_time

                counter += 1
                if counter > 1000:
                    raise Exception("no minimum identified")

            possible_time = current_mid_time

            if np.linalg.norm(get_juice_position_wrt_moon(possible_time)) <= threshold:
                flyby_times.append(possible_time)

        lower_time = lower_time + step
        mid_time = mid_time + step
        upper_time = upper_time + step

    return flyby_times



# Load spice kernels
path = os.path.dirname(__file__)
kernels = [path+'/../kernels/kernel_juice.bsp', path+'/../kernels/kernel_noe.bsp']
spice.load_standard_kernels(kernels)

colors = {"blue": (2/255, 81/255, 158/255),
          "light_blue": (42/255, 193/255, 176/255),
          "red": (153/255, 61/255, 90/255),
          "orange": (205/255, 119/255, 66/255)}
colors_list = list(colors.values())


# Global simulation settings
include_orbital_phase = True
include_clipper = True

# Set simulation start and end epochs
start_epoch = 32.5 * constants.JULIAN_YEAR
end_epoch = 33.0 * constants.JULIAN_YEAR


# Define default body settings
bodies_to_create = [flyby_moon, "Jupiter", "Sun"]
global_frame_origin = "Jupiter"
global_frame_orientation = "J2000"

body_settings = environment_setup.get_default_body_settings(bodies_to_create, global_frame_origin, global_frame_orientation)

body_settings.get(flyby_moon).rotation_model_settings = environment_setup.rotation_model.synchronous(
        "Jupiter", global_frame_orientation, "IAU_" + flyby_moon)

# Create empty settings for JUICE
body_settings.add_empty_settings("JUICE")

empty_ephemeris_dict = dict()
juice_ephemeris = environment_setup.ephemeris.tabulated(
        empty_ephemeris_dict,
        global_frame_origin,
        global_frame_orientation)
body_settings.get("JUICE").ephemeris_settings = juice_ephemeris

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add JUICE spacecraft to system of bodies
bodies.get("JUICE").mass = 5.0e3

# Create radiation pressure settings
ref_area = 100.0
srp_coef = 1.2
occulting_bodies = {"Sun": [flyby_moon]}
juice_srp_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
    ref_area, srp_coef, occulting_bodies)
environment_setup.add_radiation_pressure_target_model(bodies, "JUICE", juice_srp_settings)

# Find JUICE flybys
closest_approaches_juice = find_closest_approaches(start_epoch, end_epoch, 2.0e7)

# Extract first flyby
flyby_time = closest_approaches_juice[3]
print('flyby time', flyby_time / constants.JULIAN_YEAR)



# Define accelerations acting on JUICE
accelerations_settings_juice = dict(
    Callisto=[
        propagation_setup.acceleration.spherical_harmonic_gravity(6, 6),
        propagation_setup.acceleration.empirical()
    ],
    Jupiter=[
      propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)
    ],
    Sun=[
        propagation_setup.acceleration.radiation_pressure(),
        propagation_setup.acceleration.point_mass_gravity()
    ])

acceleration_settings = {"JUICE": accelerations_settings_juice}

body_to_propagate = ["JUICE"]
central_body = [flyby_moon]

acceleration_models = propagation_setup.create_acceleration_models(
    bodies, acceleration_settings, body_to_propagate, central_body)

# Define integrator settings
time_step = 50.0
integrator = propagation_setup.integrator.runge_kutta_fixed_step_size(
    initial_time_step=time_step, coefficient_set=propagation_setup.integrator.rkf_78)


time_around_ca = 0.5 * 3600.0

# Get initial state of JUICE wrt the flyby moon from SPICE (JUICE's SPICE ID: -28)
initial_state_1 = spice.get_body_cartesian_state_at_epoch("-28", flyby_moon, "J2000", "None", flyby_time)
initial_state_2 = spice.get_body_cartesian_state_at_epoch("-28", flyby_moon, "J2000", "None", flyby_time - time_around_ca)
print('initial state', initial_state_1)
print('initial_altitude wrt flyby moon', (np.linalg.norm(initial_state_1[:3]) - bodies.get(flyby_moon).shape_model.average_radius)/1e3, 'km')

# Define dependent variables
dependent_variables_names = [
    propagation_setup.dependent_variable.latitude("JUICE", flyby_moon),
    propagation_setup.dependent_variable.longitude("JUICE", flyby_moon)
]

# Define propagator settings
# propagator_settings = propagation_setup.propagator.translational(
#     central_body, acceleration_models, body_to_propagate, initial_state, start_gco, integrator_moons, propagation_setup.propagator.time_termination(end_gco))

# propagator_settings_list = []
# for i in range(nb_arcs):
#     propagator_settings_list.append(propagation_setup.propagator.translational(
#         central_body, acceleration_models, body_to_propagate, initial_states[i], arc_start_times[i], integrator_moons, propagation_setup.propagator.time_termination(arc_end_times[i]),
#         propagation_setup.propagator.cowell, dependent_variables_names))
# propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)

# Create termination settings
termination_condition = propagation_setup.propagator.non_sequential_termination(
    propagation_setup.propagator.time_termination(flyby_time + time_around_ca),
    propagation_setup.propagator.time_termination(flyby_time - time_around_ca))

propagator_settings_1 = propagation_setup.propagator.translational(
        central_body, acceleration_models, body_to_propagate, initial_state_1, flyby_time, integrator, termination_condition,
        propagation_setup.propagator.cowell, dependent_variables_names)

propagator_settings_2 = propagation_setup.propagator.translational(
        central_body, acceleration_models, body_to_propagate, initial_state_2, flyby_time - time_around_ca, integrator, propagation_setup.propagator.time_termination(flyby_time + time_around_ca),
        propagation_setup.propagator.cowell, dependent_variables_names)

# Propagate dynamics
simulator_1 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_1)
propagated_state_1 = result2array(simulator_1.state_history)
dependent_variables_1 = result2array(simulator_1.dependent_variable_history)

# Propagate dynamics
simulator_2 = numerical_simulation.create_dynamics_simulator(bodies, propagator_settings_2)
propagated_state_2 = result2array(simulator_2.state_history)
dependent_variables_2 = result2array(simulator_2.dependent_variable_history)

# Get state history from SPICE
state_from_spice_dict = dict()
for time in dependent_variables_1[:, 0]:
    state_from_spice_dict[time] = spice.get_body_cartesian_state_at_epoch("-28", flyby_moon, "J2000", "None", time)
state_from_spice = result2array(state_from_spice_dict)

# Plot trajectory
fig, axs = plt.subplots(1, 2)
axs[0] = plt.axes(projection='3d')
axs[0].plot(propagated_state_1[:, 1]/1e3, propagated_state_1[:, 2]/1e3, propagated_state_1[:, 3]/1e3, color=colors["blue"])
axs[0].plot(propagated_state_2[:, 1]/1e3, propagated_state_2[:, 2]/1e3, propagated_state_2[:, 3]/1e3, color=colors["red"])
axs[0].plot(state_from_spice[:, 1]/1e3, state_from_spice[:, 2]/1e3, state_from_spice[:, 3]/1e3, color='green')
axs[0].set_xlabel('x [km]')
axs[0].set_ylabel('y [km]')
axs[0].set_zlabel('z [km]')
axs[0].set_title('JUICE flyby')
axs[0].grid()


diff_1_wrt_spice = propagated_state_1 - state_from_spice
diff_2_wrt_spice = propagated_state_2 - state_from_spice
diff_1_wrt_spice[:, 0] = state_from_spice[:, 0]
diff_2_wrt_spice[:, 0] = state_from_spice[:, 0]

# for i in range(len(diff_1_wrt_spice)):


plt.figure()
plt.plot(diff_1_wrt_spice[:, 1], color='blue')
plt.plot(diff_2_wrt_spice[:, 1], color='red')
plt.grid()
plt.show()

# axs[1] = plt.axes(projection='3d')
# axs[1].plot(propagated_state_2[:, 1]/1e3, propagated_state_2[:, 2]/1e3, propagated_state_2[:, 3]/1e3, color=colors["blue"])
# axs[1].set_xlabel('x [km]')
# axs[1].set_ylabel('y [km]')
# axs[1].set_zlabel('z [km]')
# axs[1].set_title('JUICE flyby')
# axs[1].grid()

plt.show()


for i in range(len(dependent_variables_1)):
    if dependent_variables_1[i, 2] < 0:
        dependent_variables_1[i, 2] = dependent_variables_1[i, 2] + 2.0 * np.pi

for i in range(len(dependent_variables_2)):
    if dependent_variables_2[i, 2] < 0:
        dependent_variables_2[i, 2] = dependent_variables_2[i, 2] + 2.0 * np.pi

moon_map = '/home/mfayolle/Documents/PHD/04 - year 4/tudat-bundle/tudatpy/examples/estimation/callisto_map.jpg'
img = plt.imread(moon_map)

fig, ax = plt.subplots()
ax.imshow(img, extent = [0, 360, -90, 90])
ax.plot(dependent_variables_1[:, 2]*180/np.pi, dependent_variables_1[:, 1]*180.0/np.pi, color='blue')
ax.plot(dependent_variables_2[:, 2]*180/np.pi, dependent_variables_2[:, 1]*180.0/np.pi, color='red')
plt.xlabel('Longitude [deg]')
plt.ylabel('Latitude [deg]')
plt.xticks(np.arange(0, 361, 40))
plt.yticks(np.arange(-90, 91, 30))
plt.show()


