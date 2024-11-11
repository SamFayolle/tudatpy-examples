# MARS EXPRESS - Using Different Dynamical Models for the Simulation of Observations and the Estimation
"""
Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""


## Context
"""
"""

# import sys
# sys.path.insert(0, "/home/mfayolle/Tudat/tudat-bundle/cmake-build-release-2/tudatpy")

# Load required standard modules
import multiprocessing as mp
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# Load required tudatpy modules
from tudatpy import constants
from tudatpy.io import save2txt
from tudatpy.io import grail_mass_level_0_file_reader
from tudatpy.io import grail_antenna_file_reader
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.astro import time_conversion
from tudatpy.astro import frame_conversion
from tudatpy.astro import element_conversion
from tudatpy.math import interpolators
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation
from tudatpy.numerical_simulation.environment_setup import radiation_pressure
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.numerical_simulation import estimation, estimation_setup
from tudatpy.numerical_simulation import create_dynamics_simulator
from tudatpy.numerical_simulation.estimation_setup import observation
from tudatpy import util

from load_pds_files import download_url_files_time, download_url_files_time_interval
from datetime import datetime, timedelta
from urllib.request import urlretrieve

current_directory = os.getcwd()

def get_grail_files(local_path, start_date, end_date):

    all_dates = [start_date+timedelta(days=x) for x in range((end_date-start_date).days+1)]

    # Clock files
    print('---------------------------------------------')
    print('Download GRAIL clock files')
    clock_files=["gra_sclkscet_00013.tsc", "gra_sclkscet_00014.tsc"]
    url_clock_files="https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/sclk/"
    for file in clock_files:
        if ( os.path.exists(local_path+file) == False ):
            print('download', local_path+file)
            urlretrieve(url_clock_files+file, local_path+file)

    print('relevant clock files')
    for f in clock_files:
        print(f)


    # Orientation files
    print('---------------------------------------------')
    print('Download GRAIL orientation kernels')
    url_orientation_files = "https://naif.jpl.nasa.gov/pub/naif/pds/data/grail-l-spice-6-v1.0/grlsp_1000/data/ck/"
    orientation_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='gra_rec_*.bc', start_date=start_date,end_date=end_date,
        url=url_orientation_files, time_interval_format='%y%m%d_%y%m%d' )

    print('relevant orientation files')
    for f in orientation_files_to_load:
        print(f)


    # Tropospheric corrections
    print('---------------------------------------------')
    print('Download GRAIL tropospheric corrections files')
    url_tro_files = "https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/ancillary/tro/"
    tro_files_to_load = download_url_files_time_interval(
        local_path=local_path, filename_format='grxlugf*.tro', start_date=start_date,
        end_date=end_date, url=url_tro_files, time_interval_format='%Y_%j_%Y_%j' )

    print('relevant tropospheric corrections files')
    for f in tro_files_to_load:
        print(f)


    # Ionospheric corrections
    print('---------------------------------------------')
    print('Download GRAIL ionospheric corrections files')
    url_ion_files = "https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/ancillary/ion/"
    ion_files_to_load = download_url_files_time_interval(local_path=local_path, filename_format='gralugf*.ion', start_date=start_date,
                                     end_date=end_date, url=url_ion_files, time_interval_format='%Y_%j_%Y_%j' )

    print('relevant ionospheric corrections files')
    for f in ion_files_to_load:
        print(f)


    # Manoeuvres file (identical for all dates)
    print('---------------------------------------------')
    print('Download GRAIL manoeuvres file')
    manoeuvres_file="mas00_2012_04_06_a_04.asc"
    url_manoeuvres_files="https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-2-edr-v1/grail_0001/level_0/2012_04_06/"
    if ( os.path.exists(local_path+manoeuvres_file) == False ):
        print('download', local_path+manoeuvres_file)
        urlretrieve(url_manoeuvres_files+manoeuvres_file, local_path+manoeuvres_file)

    print('relevant manoeuvres files')
    print(manoeuvres_file)


    # Antenna switch files
    print('---------------------------------------------')
    print('Download antenna switch files')
    url_antenna_files = ("https://pds-geosciences.wustl.edu/grail/grail-l-lgrs-3-cdr-v1/grail_0101/level_1b/")
    # antenna_files_to_load = download_url_files_time(local_path=ancillary_path, filename_format='*/vgx1b_*_a_04.asc', start_date=start_date,
    #                         end_date=end_date, url=url_antenna_files, time_format='%Y_%m_%d', filename_size=36, indices_date_filename=[0,8])
    antenna_files_to_load = download_url_files_time(
        local_path=local_path, filename_format='*/vgs1b_*_\w\w\w\w.asc', start_date=start_date,
        end_date=end_date, url=url_antenna_files, time_format='%Y_%m_%d', filename_size=36, indices_date_filename=[0,8])

    print('relevant antenna files')
    for f in antenna_files_to_load:
        print(f)


    # ODF files
    print('---------------------------------------------')
    print('Download GRAIL ODF files')
    url_odf = ("https://pds-geosciences.wustl.edu/grail/grail-l-rss-2-edr-v1/grail_0201/odf/")
    odf_files_to_load = download_url_files_time(
        local_path=local_path, filename_format='gralugf*_\w\w\w\wsmmmv1.odf', start_date=start_date,
        end_date=end_date, url=url_odf, time_format='%Y_%j', filename_size=30, indices_date_filename=[7])

    print('relevant odf files')
    for f in odf_files_to_load:
        print(f)

    return clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file, antenna_files_to_load, odf_files_to_load


def load_clock_kernels( test_index ):
    spice.load_kernel(current_directory + "/grail_kernels/gra_sclkscet_00013.tsc")
    spice.load_kernel(current_directory + "/grail_kernels/gra_sclkscet_00014.tsc")


def load_orientation_kernels( test_index ):
    if test_index == 0:
        spice.load_kernel(current_directory + "/grail_kernels/gra_rec_120402_120408.bc")
    if( test_index > 0 and test_index <= 6 ):
        spice.load_kernel(current_directory + "/grail_kernels/gra_rec_120409_120415.bc")
    if( test_index >= 4 ):
        spice.load_kernel(current_directory + "/grail_kernels/gra_rec_120416_120422.bc")

def get_grail_odf_file_name( test_index ):
    if test_index == 0:
        return current_directory + '/grail_data/gralugf2012_097_0235smmmv1.odf'
    elif test_index == 1:
        return current_directory + '/grail_data/gralugf2012_100_0540smmmv1.odf'
    elif test_index == 2:
        return current_directory + '/grail_data/gralugf2012_101_0235smmmv1.odf'
    elif test_index == 3:
        return current_directory + '/grail_data/gralugf2012_102_0358smmmv1.odf'
    elif test_index == 4:
        return current_directory + '/grail_data/gralugf2012_103_0145smmmv1.odf'
    elif test_index == 5:
        return current_directory + '/grail_data/gralugf2012_105_0352smmmv1.odf'
    elif test_index == 6:
        return current_directory + '/grail_data/gralugf2012_107_0405smmmv1.odf'
    elif test_index == 7:
        return current_directory + '/grail_data/gralugf2012_108_0450smmmv1.odf'
        
def get_grail_antenna_file_name( test_index ):
    if test_index == 0:
        return current_directory + '/kernels_test/vgs1b_2012_04_07_a_04.asc'
    elif test_index == 1:
        return current_directory + '/kernels_test/vgs1b_2012_04_10_a_04.asc'
    elif test_index == 2:
        return current_directory + '/kernels_test/vgs1b_2012_04_11_a_04.asc'
    elif test_index == 3:
        return current_directory + '/kernels_test/vgs1b_2012_04_12_a_04.asc'
    elif test_index == 4:
        return current_directory + '/kernels_test/vgs1b_2012_04_13_a_04.asc'
    elif test_index == 5:
        return current_directory + '/kernels_test/vgs1b_2012_04_15_a_04.asc'
    elif test_index == 6:
        return current_directory + '/kernels_test/vgs1b_2012_04_17_a_04.asc'
    elif test_index == 7:
        return current_directory + '/kernels_test/vgs1b_2012_04_18_a_04.asc'


def get_grail_panel_geometry( ):
    # first read the panel data from input file
    this_file_path = os.path.dirname(os.path.abspath(__file__))
    panel_data = pd.read_csv(this_file_path + "/input/grail_macromodel.txt", delimiter=", ", engine="python")
    material_data = pd.read_csv(this_file_path + "/input/grail_materials.txt", delimiter=", ", engine="python")

    # initialize list to store all panel settings
    all_panel_settings = []

    for i, row in panel_data.iterrows():
        # create panel geometry settings
        # Options are: frame_fixed_panel_geometry, time_varying_panel_geometry, body_tracking_panel_geometry
        panel_geometry_settings = environment_setup.vehicle_systems.frame_fixed_panel_geometry(
            np.array([row["x"], row["y"], row["z"]]),  # panel position in body reference frame
            row["area"]  # panel area
        )

        panel_material_data = material_data[material_data["material"] == row["material"]]

        # create panel radiation settings (for specular and diffuse reflection)
        specular_diffuse_body_panel_reflection_settings = environment_setup.radiation_pressure.specular_diffuse_body_panel_reflection(
            specular_reflectivity=float(panel_material_data["Cs"].iloc[0]),
            diffuse_reflectivity=float(panel_material_data["Cd"].iloc[0]), with_instantaneous_reradiation=True
        )

        # create settings for complete pannel (combining geometry and material properties relevant for radiation pressure calculations)
        complete_panel_settings = environment_setup.vehicle_systems.body_panel_settings(
            panel_geometry_settings,
            specular_diffuse_body_panel_reflection_settings
        )

        # add panel settings to list of all panel settings
        all_panel_settings.append(
            complete_panel_settings
        )

    # Create settings object for complete vehicle shape
    full_panelled_body_settings = environment_setup.vehicle_systems.full_panelled_body_settings(
        all_panel_settings
    )
    return full_panelled_body_settings

def get_rsw_state_difference(
        estimated_state_history,
        spacecraft_name,
        spacecraft_central_body,
        global_frame_orientation ):
    rsw_state_difference = dict()
    counter = 0
    for time in estimated_state_history:
        current_estimated_state = estimated_state_history[time]
        current_spice_state = spice.get_body_cartesian_state_at_epoch(spacecraft_name, spacecraft_central_body,
                                                                      global_frame_orientation, "None", time)
        current_state_difference = current_estimated_state - current_spice_state
        current_position_difference = current_state_difference[0:3]
        current_velocity_difference = current_state_difference[3:6]
        rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(current_estimated_state)
        current_rsw_state_difference = np.ndarray([6])
        current_rsw_state_difference[0:3] = rotation_to_rsw @ current_position_difference
        current_rsw_state_difference[3:6] = rotation_to_rsw @ current_velocity_difference
        rsw_state_difference[time] = current_rsw_state_difference
        counter = counter+1
    return rsw_state_difference


def run_estimation( input_index ):

    with util.redirect_std( 'estimation_output_' + str( input_index ) + ".dat", True, True ):

        print("input_index", input_index)

        global_filename_suffix = str(input_index) + '_testLT'

    # while True:
        number_of_files = 8
        test_index = input_index % number_of_files

        perform_estimation = True
        fit_to_kernel = False
        # perform_estimation = False
        # fit_to_kernel = False
        # if( input_index == test_index ):
        #     perform_estimation = True
        # else:
        #     fit_to_kernel = True
        # Load standard spice kernels as well as the one describing the orbit of Mars Express
        spice.load_standard_kernels()
        spice.load_kernel(current_directory + "/grail_kernels/moon_de440_200625.tf")
        spice.load_kernel(current_directory + "/grail_kernels/grail_v07.tf")
        load_clock_kernels( test_index )
        spice.load_kernel(current_directory + "/grail_kernels/grail_120301_120529_sci_v02.bsp")
        load_orientation_kernels( test_index )
        spice.load_kernel(current_directory + "/grail_kernels/moon_pa_de440_200625.bpc")

        # Define start and end times for environment
        initial_time_environment = time_conversion.DateTime( 2012, 3, 2, 0, 0, 0.0 ).epoch( )
        final_time_environment = time_conversion.DateTime( 2012, 5, 29, 0, 0, 0.0 ).epoch( )

        start_date = datetime( 2012, 3, 30 )
        end_date = datetime( 2012, 4, 12 )

        # clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file, \
        # antenna_files_to_load, odf_files_to_load = get_grail_files("kernels_test/", start_date, end_date)

        # Load ODF file
        single_odf_file_contents = estimation_setup.observation.process_odf_data_single_file(
            get_grail_odf_file_name(test_index), 'GRAIL-A', True)

        # Create observation collection from ODF file
        original_odf_observations = estimation_setup.observation.create_odf_observed_observation_collection(
            single_odf_file_contents, list(), [numerical_simulation.Time(0, np.nan), numerical_simulation.Time(0, np.nan)])
        observation_time_limits = original_odf_observations.time_bounds
        initial_time = observation_time_limits[0] - 3600.0
        final_time = observation_time_limits[1] + 3600.0

        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        print('Initial time', initial_time.to_float())
        print('Final time', final_time.to_float())
        print('Time in hours: ', (final_time.to_float() - initial_time.to_float()) / 3600)

        # time_bounds_test = original_odf_observations.get_single_observation_sets()[0].time_bounds


        # original_odf_observations.filter_observations(
        #     estimation.observation_filter(estimation.time_bounds_filtering, (time_bounds_test[0].to_float(),time_bounds_test[1].to_float()), False))

        print('original_odf_observations')
        original_odf_observations.print_observation_sets_start_and_size()

        # test = original_odf_observations.get_reference_points_in_link_ends()
        # test2 = original_odf_observations.get_bodies_in_link_ends()
        # test_sets = original_odf_observations.get_single_observation_sets()
        # print('nb sets', len(test_sets))
        # for set in test_sets:
        #     link_ends_transmitter = set.link_definition.link_end_id(observation.transmitter)
        #     link_ends_receiver = set.link_definition.link_end_id(observation.receiver)
        #     link_ends_reflector1 = set.link_definition.link_end_id(observation.reflector1)
        #     new_link_ends = dict()
        #     new_link_ends[observation.transmitter] = observation.body_reference_point_link_end_id(link_ends_transmitter.body_name, link_ends_transmitter.reference_point )
        #     new_link_ends[observation.receiver] = observation.body_reference_point_link_end_id(link_ends_receiver.body_name, link_ends_receiver.reference_point)
        #     new_link_ends[observation.reflector1] = observation.body_reference_point_link_end_id(link_ends_reflector1.body_name, "Antenna")
        #     set.link_definition = observation.LinkDefinition(new_link_ends)
        #
        # new_obs_collection = estimation.ObservationCollection(test_sets)
        # test = new_obs_collection.get_reference_points_in_link_ends()
        # test2 = new_obs_collection.get_bodies_in_link_ends()

        arc_start_times = [initial_time]
        arc_end_times = [final_time]

        # Create default body settings for celestial bodies
        bodies_to_create = [ "Earth", "Sun", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", "Moon" ]
        global_frame_origin = "SSB"
        global_frame_orientation = "J2000"
        body_settings = environment_setup.get_default_body_settings_time_limited(
            bodies_to_create, initial_time_environment.to_float( ), final_time_environment.to_float( ), global_frame_origin, global_frame_orientation)

        # Modify Earth default settings
        body_settings.get( 'Earth' ).shape_settings = environment_setup.shape.oblate_spherical_spice( )
        body_settings.get( 'Earth' ).rotation_model_settings = environment_setup.rotation_model.gcrs_to_itrs(
            environment_setup.rotation_model.iau_2006, global_frame_orientation,
            interpolators.interpolator_generation_settings_float( interpolators.cubic_spline_interpolation( ), initial_time_environment.to_float( ), final_time_environment.to_float( ), 3600.0 ),
            interpolators.interpolator_generation_settings_float( interpolators.cubic_spline_interpolation( ), initial_time_environment.to_float( ), final_time_environment.to_float( ), 3600.0 ),
            interpolators.interpolator_generation_settings_float( interpolators.cubic_spline_interpolation( ), initial_time_environment.to_float( ), final_time_environment.to_float( ), 60.0 ) )
        body_settings.get( 'Earth' ).gravity_field_settings.associated_reference_frame = "ITRS"
        body_settings.get( "Earth" ).ground_station_settings = environment_setup.ground_station.dsn_stations( )

        # Modify Moon default settings
        body_settings.get( 'Moon' ).rotation_model_settings = environment_setup.rotation_model.spice( global_frame_orientation, "MOON_PA_DE440", "MOON_PA_DE440" )
        body_settings.get( 'Moon' ).gravity_field_settings = environment_setup.gravity_field.predefined_spherical_harmonic(
            environment_setup.gravity_field.gggrx1200 , 500)
        body_settings.get( 'Moon' ).gravity_field_settings.associated_reference_frame = "MOON_PA_DE440"
        moon_gravity_field_variations = list()
        moon_gravity_field_variations.append( environment_setup.gravity_field_variation.solid_body_tide( 'Earth', 0.02405, 2) )
        moon_gravity_field_variations.append( environment_setup.gravity_field_variation.solid_body_tide( 'Sun', 0.02405, 2) )
        body_settings.get( 'Moon' ).gravity_field_variation_settings = moon_gravity_field_variations
        body_settings.get( 'Moon' ).ephemeris_settings.frame_origin = "Earth"

        # Add Moon radiation properties
        moon_surface_radiosity_models = [
            radiation_pressure.thermal_emission_angle_based_radiosity(
                95.0, 385.0, 0.95, "Sun" ),
            radiation_pressure.variable_albedo_surface_radiosity(
                radiation_pressure.predefined_spherical_harmonic_surface_property_distribution( radiation_pressure.albedo_dlam1 ), "Sun" ) ]
        body_settings.get( "Moon" ).radiation_source_settings = radiation_pressure.panelled_extended_radiation_source(
            moon_surface_radiosity_models, [ 6, 12 ] )

        # Create vehicle properties
        spacecraft_name = "GRAIL-A"
        spacecraft_central_body = "Moon"
        body_settings.add_empty_settings( spacecraft_name )

        body_settings.get( spacecraft_name ).ephemeris_settings = environment_setup.ephemeris.interpolated_spice(
            initial_time_environment.to_float(), final_time_environment.to_float(), 10.0, spacecraft_central_body, global_frame_orientation )

        body_settings.get( spacecraft_name ).rotation_model_settings = environment_setup.rotation_model.spice( global_frame_orientation, spacecraft_name + "_SPACECRAFT", "" )
        occulting_bodies = dict()
        occulting_bodies[ "Sun" ] = [ "Moon"]

        body_settings.get( spacecraft_name ).constant_mass = 150
        body_settings.get(spacecraft_name).vehicle_shape_settings = get_grail_panel_geometry()

        # position_antenna = spice.get_body_cartesian_position_at_epoch("-177010", "-177000", "GRAIL-A_SPACECRAFT", "None",
        #                                                               initial_time.to_float())
        # stop = 1
        # bodies.get(spacecraft_name).system_models.set_reference_point("Antenna", np.array(position_antenna))

        single_obs_sets = original_odf_observations.get_single_observation_sets()
        ref_frequencies = []
        print('ref frequency')
        for set in single_obs_sets:
            ref_frequencies.append(set.ancilliary_settings.get_float_settings( observation.doppler_reference_frequency ))
            print(set.ancilliary_settings.get_float_settings( observation.doppler_reference_frequency ))
        stop = 1

        # # Define antenna positions
        # grail_antennas = dict()
        # grail_antennas["AntennaSBand1"] = [-0.082, 0.152, -0.810]
        # grail_antennas["AntennaSBand2"] = [1.1920, -0.1780, 0.0]
        # grail_antennas["AntennaXBand1"] = [-0.0820, 0.3180, 0.7620]
        # grail_antennas["AntennaXBand2"] = [1.1920, 0.2070, 0.2340]
        # grail_antennas["Antenna1"] = [-0.082, 0.15200000000000002, -0.808]
        # grail_antennas["Antenna2"] =[1.1919999999999997, -0.17799999999999994, 0.0]


        # Create environment
        bodies = environment_setup.create_system_of_bodies(body_settings)
        # bodies.get(spacecraft_name).system_models.set_reference_point("Antenna", np.array([1.1919999999999997, -0.17799999999999994, 0.0]))
        # bodies.get( spacecraft_name ).system_models.set_reference_point( "AntennaSBand1", np.array([-0.082, 0.152, -0.810]))
        # bodies.get( spacecraft_name ).system_models.set_reference_point( "AntennaSBand2", np.array([1.1920, -0.1780, 0.0]))
        # bodies.get( spacecraft_name ).system_models.set_reference_point( "AntennaXBand1", np.array([-0.0820, 0.3180, 0.7620]))
        # bodies.get( spacecraft_name ).system_models.set_reference_point( "AntennaXBand2", np.array([1.1920, 0.2070, 0.2340]))
        # bodies.get(spacecraft_name).system_models.set_reference_point("Antenna1",np.array([-0.082, 0.15200000000000002, -0.808]))
        # bodies.get(spacecraft_name).system_models.set_reference_point("Antenna2",np.array([1.1919999999999997, -0.17799999999999994, 0.0]))
        environment_setup.add_radiation_pressure_target_model(
            bodies, spacecraft_name, radiation_pressure.cannonball_radiation_target(5, 1.5, occulting_bodies))
        environment_setup.add_radiation_pressure_target_model(
            bodies, spacecraft_name, radiation_pressure.panelled_radiation_target(occulting_bodies))

        # Update bodies based on ODF file
        estimation_setup.observation.set_odf_information_in_bodies(single_odf_file_contents, bodies)

        # Load antenna switch files
        manoeuvres_times = grail_mass_level_0_file_reader( current_directory + '/grail_data/mas00_2012_04_06_a_04.asc' )
        antenna_switch_times = grail_antenna_file_reader( get_grail_antenna_file_name( test_index ) )[ 0 ] #current_directory + '/kernels_test/vgs1b_2012_04_13_a_04.asc' )[0] #get_grail_antenna_file_name( test_index ) )[ 0 ]
        antenna_switch_positions = grail_antenna_file_reader( get_grail_antenna_file_name( test_index ) )[1]

        antenna_switch_history = dict()
        for i in range(len(antenna_switch_times)):
            antenna_switch_history[ antenna_switch_times[i] ] = np.array(antenna_switch_positions[i*3:(i+1)*3])
        antenna_switch_history[ initial_time.to_float() ] = np.array(antenna_switch_positions[0:3])
        antenna_switch_history[ final_time.to_float( ) ] = np.array(antenna_switch_positions[-3:])

        # Define arc split times based on antenna switch epochs
        split_times = []
        if (len(antenna_switch_times)>2):
            for switch_time in antenna_switch_times:
                if ( switch_time >= initial_time.to_float() and switch_time <= final_time.to_float() ):
                    print("antenna switch detected!")
                    split_times.append(switch_time)

        # Retrieve manoeuvres epochs
        relevant_manoeuvres = []
        for manoeuvre_time in manoeuvres_times:
            if (manoeuvre_time >= initial_time.to_float() and manoeuvre_time <= final_time.to_float()):
                print("manoeuvre detected!")
                # split_times.append(manoeuvre_time)
                relevant_manoeuvres.append(manoeuvre_time)

        # # Split observation sets based on antenna switch times and manoeuvres
        # original_odf_observations.split_observation_sets(
        #     estimation.observation_set_splitter(estimation.time_tags_splitter, split_times, 10))
        #
        # print('obs sets start and size post-splitting')
        # original_odf_observations.print_observation_sets_start_and_size()

        # test_sets = original_odf_observations.get_single_observation_sets()
        # for set in test_sets:
        #     set_start_time = set.time_bounds[0].to_float()
        #     set_end_time = set.time_bounds[1].to_float()
        #     print('set_start_time', set_start_time)
        #     print('set_end_time', set_end_time)
        #     # TO BE CHANGED
        #     if set_start_time < antenna_switch_times[0]:
        #         current_antenna = antenna_switch_positions[0:3]
        #         print('current antenna', current_antenna)
        #     if set_end_time > antenna_switch_times[-1]:
        #         current_antenna = antenna_switch_positions[-3:]
        #         print('current antenna', current_antenna)
        #     else:
        #         for i in range(len(antenna_switch_times)-1):
        #             if set_start_time >= antenna_switch_times[i] and set_end_time <= antenna_switch_times[i+1]:
        #                 current_antenna = antenna_switch_positions[i*3:(i+1)*3]
        #                 print('current antenna', current_antenna)
        #
        #
        #     if ( current_antenna in grail_antennas.values() == False ):
        #         new_antenna_name = "Antenna" + str(len(grail_antennas))
        #         print('new_antenna_name', new_antenna_name)
        #         grail_antennas[ new_antenna_name ] = current_antenna
        #         bodies.get(spacecraft_name).system_models.set_reference_point(new_antenna_name, np.array(current_antenna))
        #     else:
        #         name = [i for i in grail_antennas if grail_antennas[i] == current_antenna]
        #         for nametest in name:
        #             print('antenna name', nametest)
        #
        #     link_ends_transmitter = set.link_definition.link_end_id(observation.transmitter)
        #     link_ends_receiver = set.link_definition.link_end_id(observation.receiver)
        #     link_ends_reflector1 = set.link_definition.link_end_id(observation.reflector1)
        #     new_link_ends = dict()
        #     new_link_ends[observation.transmitter] = observation.body_reference_point_link_end_id(
        #         link_ends_transmitter.body_name, link_ends_transmitter.reference_point)
        #     new_link_ends[observation.receiver] = observation.body_reference_point_link_end_id(
        #         link_ends_receiver.body_name, link_ends_receiver.reference_point)
        #     new_link_ends[observation.reflector1] = observation.body_reference_point_link_end_id(
        #         link_ends_reflector1.body_name, name[0])
        #     set.link_definition = observation.LinkDefinition(new_link_ends)
        # stop = 1

        original_odf_observations.set_reference_points(bodies, antenna_switch_history, spacecraft_name, observation.reflector1)

        # Pre-splitting arc definition (TO BE REMOVED)
        for arc in range(len(arc_start_times)):
            print('arc start time', arc_start_times[arc].to_float())
            print('arc end time', arc_end_times[arc].to_float())

        # Define new arcs after splitting based on antenna switches
        for time in split_times:
            print('split set at ', time)
            new_arc_start_dt = datetime.utcfromtimestamp(time) + (datetime(2000, 1, 1, 12) - datetime(1970, 1, 1))
            new_arc_start = time_conversion.datetime_to_tudat(new_arc_start_dt).epoch()

            for arc in range(len(arc_start_times)):
                if ( new_arc_start.to_float() > arc_start_times[arc].to_float() and new_arc_start.to_float( ) < arc_end_times[arc] ):
                    arc_start_times.insert(arc+1,new_arc_start)
                    arc_end_times.insert(arc,new_arc_start)

        # Post-splitting arc definition (TO BE REMOVED)
        for arc in range(len(arc_start_times)):
            print('arc start time', arc_start_times[arc].to_float())
            print('arc end time', arc_end_times[arc].to_float())

        nb_arcs = len(arc_start_times)


        times_manoeuvres = []
        for time in relevant_manoeuvres:
            time_dt = datetime.utcfromtimestamp(time) + (datetime(2000, 1, 1, 12) - datetime(1970, 1, 1))
            times_manoeuvres.append(time_conversion.datetime_to_tudat(time_dt).epoch())

        # Create accelerations
        accelerations_settings_spacecraft = dict(
            Sun=[
                propagation_setup.acceleration.radiation_pressure( environment_setup.radiation_pressure.paneled_target ),
                propagation_setup.acceleration.point_mass_gravity() ],
            Earth=[
                propagation_setup.acceleration.point_mass_gravity() ],
            Moon=[
                propagation_setup.acceleration.spherical_harmonic_gravity(256, 256),
                propagation_setup.acceleration.radiation_pressure( environment_setup.radiation_pressure.cannonball_target ),
                propagation_setup.acceleration.empirical()
                ],
            Mars=[
                propagation_setup.acceleration.point_mass_gravity() ],
            Venus=[
                propagation_setup.acceleration.point_mass_gravity() ],
            Jupiter=[
                propagation_setup.acceleration.point_mass_gravity() ],
            Saturn=[
                propagation_setup.acceleration.point_mass_gravity() ]
        )

        # Add manoeuvres if detected
        if len(relevant_manoeuvres)>0:
            accelerations_settings_spacecraft[spacecraft_name] = [
                propagation_setup.acceleration.quasi_impulsive_shots_acceleration(relevant_manoeuvres, [np.zeros((3,1))], 3600.0, 60.0)]

        # Create global accelerations settings dictionary.
        acceleration_settings = { spacecraft_name: accelerations_settings_spacecraft}

        # Create acceleration models.
        bodies_to_propagate = [ spacecraft_name ]
        central_bodies = [ spacecraft_central_body ]
        acceleration_models = propagation_setup.create_acceleration_models(
            bodies,
            acceleration_settings,
            bodies_to_propagate,
            central_bodies)

        # Define integrator settings
        integration_step = 30.0
        integrator_settings = propagation_setup.integrator.runge_kutta_fixed_step_size(
            numerical_simulation.Time( 0, integration_step ), propagation_setup.integrator.rkf_78 )

        # Retrieve initial states
        initial_states = []
        for time in arc_start_times:
            initial_states.append(
                propagation.get_state_of_bodies(bodies_to_propagate,central_bodies,bodies,time))
        print('initial states', initial_states)

        # Define propagator settings
        propagator_settings_list = []
        for i in range(nb_arcs):
            arc_wise_propagator_settings = propagation_setup.propagator.translational(
                central_bodies, acceleration_models, bodies_to_propagate, initial_states[i], arc_start_times[i],
                integrator_settings, propagation_setup.propagator.time_termination(arc_end_times[i].to_float()))
            arc_wise_propagator_settings.print_settings.results_print_frequency_in_steps = 3600.0 / integration_step
            propagator_settings_list.append(arc_wise_propagator_settings)
        # propagator_settings = propagation_setup.propagator.multi_arc(propagator_settings_list)


        if perform_estimation:

            # Compress observations
            compressed_observations = estimation_setup.observation.create_compressed_doppler_collection( original_odf_observations, 60, 10)
                # original_odf_observations, 60, 10)
            print('Compressed observations: ')
            print(compressed_observations.concatenated_observations.size)


            #  Create light-time corrections list
            light_time_correction_list = list()
            light_time_correction_list.append(
                estimation_setup.observation.first_order_relativistic_light_time_correction(["Sun"]))

            tropospheric_correction_files = [
                current_directory + '/grail_data/grxlugf2012_092_2012_122.tro',
                current_directory + '/grail_data/grxlugf2012_122_2012_153.tro']
            light_time_correction_list.append(
                estimation_setup.observation.dsn_tabulated_tropospheric_light_time_correction(
                    tropospheric_correction_files))

            ionospheric_correction_files = [
                current_directory + '/grail_data/gralugf2012_092_2012_122.ion',
                current_directory + '/grail_data/gralugf2012_122_2012_153.ion']
            spacecraft_name_per_id = dict()
            spacecraft_name_per_id[177] = "GRAIL-A"
            light_time_correction_list.append(
                estimation_setup.observation.dsn_tabulated_ionospheric_light_time_correction(
                    ionospheric_correction_files, spacecraft_name_per_id))

            # Create observation model settings
            doppler_link_ends = compressed_observations.link_definitions_per_observable[
                estimation_setup.observation.dsn_n_way_averaged_doppler]
            observation_model_settings = list()
            for current_link_definition in doppler_link_ends:
                observation_model_settings.append(estimation_setup.observation.dsn_n_way_doppler_averaged(
                    current_link_definition, light_time_correction_list))

            # Create observation simulators
            observation_simulators = estimation_setup.create_observation_simulators(observation_model_settings, bodies)

            per_set_time_bounds = compressed_observations.sorted_per_set_time_bounds
            print('Arc times ================= ')
            for observable_type in per_set_time_bounds:
                for link_end_index in per_set_time_bounds[observable_type]:
                    current_times_list = per_set_time_bounds[observable_type][link_end_index]
                    for time_bounds in current_times_list:
                        print('Arc times', observable_type, ' ', link_end_index, ' ', time_bounds)

            # Compute residuals
            estimation.compute_and_set_residuals(compressed_observations, observation_simulators, bodies)

            # # # Save unfiltered residuals
            # # np.savetxt('TEST_unfiltered_residual_' + str(input_index) + '.dat',
            # #            compressed_observations.get_concatenated_residuals(),
            # #            delimiter=',')
            # # np.savetxt('TEST_unfiltered_time_' + str(input_index) + '.dat',
            # #            compressed_observations.concatenated_float_times,
            # #            delimiter=',')
            # # np.savetxt('TEST_unfiltered_link_end_ids_' + str(input_index) + '.dat',
            # #            compressed_observations.concatenated_link_definition_ids, delimiter=',')
            #
            # print('obs sets start and size pre-filtering')
            # compressed_observations.print_observation_sets_start_and_size()
            #
            # # Filter residual outliers
            # compressed_observations.filter_observations(estimation.observation_filter(estimation.residual_filtering, 0.01))
            #
            # ### ---------------------------------------------------------- TO BE REMOVED
            # print('obs sets start and size pre-splitting')
            # compressed_observations.print_observation_sets_start_and_size()
            #
            # obs_sets = compressed_observations.get_single_observation_sets()
            # print('nb obs sets', len(obs_sets))
            #
            # for set in obs_sets:
            #     print('time bounds', set.time_bounds[0].to_float(), " - ", set.time_bounds[1].to_float())
            #
            # for time in split_times:
            #     print('split set at ', time)
            # # ----------------------------------------------------------------
            #
            # # Split observation sets based on antenna switch times and manoeuvres
            # compressed_observations.split_observation_sets(
            #     estimation.observation_set_splitter(estimation.time_tags_splitter, split_times, 10))
            #
            # print('obs sets start and size post-splitting')
            # compressed_observations.print_observation_sets_start_and_size()
            #
            # print('Filtered observations: ')
            # print(compressed_observations.concatenated_observations.size)

            np.savetxt('relevant_manoeuvres_'+ str(input_index) + '.dat', relevant_manoeuvres, delimiter=',')
            np.savetxt('split_times_' + str(input_index) + '.dat', split_times, delimiter=',')


        for arc in range(0,nb_arcs):

            filename_suffix = global_filename_suffix
            if (nb_arcs > 1):
                filename_suffix += '_arc_' + str(arc)

            # Create new observation collection based on new arcs definition
            arc_wise_obs_collection = estimation.create_new_observation_collection( compressed_observations,
                estimation.observation_parser((arc_start_times[arc].to_float(), arc_end_times[arc].to_float())))
            print('Arc-wise observations: ')
            print(arc_wise_obs_collection.concatenated_observations.size)

            if (arc_wise_obs_collection.concatenated_observations.size>0):

                # Save unfiltered residuals
                np.savetxt('unfiltered_residual_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.get_concatenated_residuals(), delimiter=',')
                np.savetxt('unfiltered_obs_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.get_concatenated_observations(), delimiter=',')
                np.savetxt('unfiltered_time_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_float_times, delimiter=',')
                np.savetxt('unfiltered_link_end_ids_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_link_definition_ids, delimiter=',')

                print('obs sets start and size pre-filtering')
                arc_wise_obs_collection.print_observation_sets_start_and_size()

                # Filter residual outliers
                arc_wise_obs_collection.filter_observations(
                    estimation.observation_filter(estimation.residual_filtering, 0.01))

                ### ---------------------------------------------------------- TO BE REMOVED
                print('obs sets start and size pre-splitting')
                arc_wise_obs_collection.print_observation_sets_start_and_size()

                obs_sets = arc_wise_obs_collection.get_single_observation_sets()
                print('nb obs sets', len(obs_sets))

                for set in obs_sets:
                    print('time bounds', set.time_bounds[0].to_float(), " - ", set.time_bounds[1].to_float())

                for time in split_times:
                    print('split set at ', time)
                # ----------------------------------------------------------------

                # # Split observation sets based on antenna switch times and manoeuvres
                # arc_wise_obs_collection.split_observation_sets(
                #     estimation.observation_set_splitter(estimation.time_tags_splitter, split_times, 10))
                #
                # print('obs sets start and size post-splitting')
                # arc_wise_obs_collection.print_observation_sets_start_and_size()

                print('Filtered observations: ')
                print(arc_wise_obs_collection.concatenated_observations.size)

                # Save filtered residuals
                np.savetxt('filtered_residual_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.get_concatenated_residuals( ), delimiter=',')
                np.savetxt('filtered_time_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_float_times, delimiter=',')
                np.savetxt('filtered_link_end_ids_' + filename_suffix + '.dat',
                           arc_wise_obs_collection.concatenated_link_definition_ids, delimiter=',')


                # Define parameters to estimate
                empirical_components = dict()
                empirical_components[estimation_setup.parameter.along_track_empirical_acceleration_component] = \
                    list([estimation_setup.parameter.constant_empirical,estimation_setup.parameter.sine_empirical,estimation_setup.parameter.cosine_empirical])

                extra_parameters = [
                    estimation_setup.parameter.radiation_pressure_target_direction_scaling(spacecraft_name,"Sun"),
                    estimation_setup.parameter.radiation_pressure_target_perpendicular_direction_scaling(spacecraft_name,"Sun"),
                    estimation_setup.parameter.radiation_pressure_target_direction_scaling(spacecraft_name, "Moon") ,
                    estimation_setup.parameter.radiation_pressure_target_perpendicular_direction_scaling(spacecraft_name, "Moon"),
                    estimation_setup.parameter.empirical_accelerations(spacecraft_name, "Moon", empirical_components),
                    # estimation_setup.parameter.reference_point_position(spacecraft_name, "Antenna")
                ]

                if len(relevant_manoeuvres)>0:
                    extra_parameters.append(estimation_setup.parameter.quasi_impulsive_shots(spacecraft_name))

                # # Retrieve GRAIL state at manoeuvres epochs
                # states_manoeuvres_times = []
                # rotation_to_rsw_manoeuvres = []
                # for manoeuvre_time in times_manoeuvres:
                #     # states_manoeuvres_times.append(
                #     #     propagation.get_state_of_bodies(bodies_to_propagate, central_bodies, bodies, manoeuvre_time))
                #
                #     rotation_to_rsw = frame_conversion.inertial_to_rsw_rotation_matrix(
                #         propagation.get_state_of_bodies(bodies_to_propagate, central_bodies, bodies, manoeuvre_time))
                #     rotation_to_rsw_manoeuvres.append(rotation_to_rsw)
                #     # current_rsw_state_difference = np.ndarray([6])
                #     # current_rsw_state_difference[0:3] = rotation_to_rsw @ current_position_difference
                #     # current_rsw_state_difference[3:6] = rotation_to_rsw @ current_velocity_difference

                # if perform_estimation:
                #
                #     # Define parameters to estimate
                #     parameter_settings = estimation_setup.parameter.initial_states(propagator_settings_list[arc], bodies)
                #     parameter_settings += extra_parameters
                #
                #     parameters_to_estimate = estimation_setup.create_parameter_set(parameter_settings, bodies, propagator_settings_list[arc])
                #     nb_parameters = len(parameters_to_estimate.parameter_vector)
                #     estimation_setup.print_parameter_names(parameters_to_estimate)
                #
                #     original_parameters = parameters_to_estimate.parameter_vector
                #
                #     print('parameters', parameters_to_estimate.parameter_vector)
                #
                #     # Define inverse a priori covariance
                #     inv_a_priori_cov = np.zeros((nb_parameters, nb_parameters))
                #     # inv_a_priori_cov[-2, -2] = 1.0 / (1.0e-3 * 1.0e-3)
                #     # inv_a_priori_cov[-3, -3] = 1.0 / (1.0e-3 * 1.0e-3)
                #     # inv_a_priori_cov[-1,-1] = 1.0 / (1.0e-8 * 1.0e-8)
                #     # inv_a_priori_cov[-3, -3] = 1.0 / (1.0e-8 * 1.0e-8)
                #     # inv_a_priori_cov[-3:, -3:] = (np.transpose(rotation_to_rsw_manoeuvres[0]) @ inv_a_priori_cov[-3:, -3:]) @ \
                #     #                              rotation_to_rsw_manoeuvres[0]
                #
                #     # print('rsw', rotation_to_rsw_manoeuvres[0])
                #     # print('inv_a_priori_cov[-3:, -3:]', inv_a_priori_cov[-3:, -3:])
                #     # test_inv = np.linalg.inv(inv_a_priori_cov[-3:, -3:])
                #     # print('a_priori_cov[-3:,-3:]', test_inv)
                #     # print('test', np.linalg.inv(test_inv))
                #     # print('test', rotation_to_rsw_manoeuvres[0] @ inv_a_priori_cov[-3:, -3:])
                #     # print('inv_a_priori_cov', inv_a_priori_cov)
                #     #
                #     # inv_a_priori_cov_inertial = inv_a_priori_cov[-3:,-3:]
                #     # print('inv_a_priori_cov_inertial', inv_a_priori_cov_inertial)
                #     # print('a_priori_cov_inertial', np.linalg.inv(inv_a_priori_cov_inertial))
                #
                #     # Create estimator
                #     estimator = numerical_simulation.Estimator( bodies, parameters_to_estimate, observation_model_settings, propagator_settings_list[arc] )
                #
                #     # print('test state GRAIL', propagation.get_state_of_bodies(bodies_to_propagate, central_bodies, bodies, arc_start_times[1]))
                #
                #     estimation_input = estimation.EstimationInput( arc_wise_obs_collection, inverse_apriori_covariance = inv_a_priori_cov,
                #                                                    convergence_checker = estimation.estimation_convergence_checker( 2 ) )
                #     estimation_input.define_estimation_settings(
                #         reintegrate_equations_on_first_iteration = False,
                #         reintegrate_variational_equations = False,
                #         print_output_to_terminal = True,
                #         save_state_history_per_iteration=True)
                #     estimation_output = estimator.perform_estimation(estimation_input)
                #     np.savetxt('filtered_postfit_residual_' + filename_suffix + '.dat',
                #                estimation_output.final_residuals, delimiter=',')
                #     estimated_state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
                #
                #     # Estimated parameters
                #     print("estimated parameters", parameters_to_estimate.parameter_vector)
                #
                #     # reset parameters for next arc
                #     parameters_to_estimate.parameter_vector = original_parameters
                #
                #     print('Getting RSW difference',len(estimated_state_history),len(estimation_output.simulation_results_per_iteration))
                #     rsw_state_difference = get_rsw_state_difference(
                #         estimated_state_history, spacecraft_name, spacecraft_central_body, global_frame_orientation )
                #     print('Gotten RSW difference')
                #
                #     save2txt(rsw_state_difference, 'postfit_rsw_state_difference_' + filename_suffix + '.dat', current_directory)
                #
                #     # Correlations
                #     correlations = estimation_output.correlations
                #     np.savetxt('correlations_' + filename_suffix + '.dat', correlations, delimiter=',')
                #     np.savetxt('covariance_' + filename_suffix + '.dat', estimation_output.covariance, delimiter=',')
                #     np.savetxt('inv_a_priori_cov_' + filename_suffix + '.dat', inv_a_priori_cov, delimiter=',')


            # fit_to_kernel = False
            # if fit_to_kernel:
            #     estimation_output = estimation.create_best_fit_to_ephemeris( bodies, acceleration_models, bodies_to_propagate, central_bodies, integrator_settings,
            #                                   initial_time, final_time, numerical_simulation.Time( 0, 60.0 ), extra_parameters )
            #     estimated_state_history = estimation_output.simulation_results_per_iteration[-1].dynamics_results.state_history_float
            #     print('Getting RSW difference',len(estimated_state_history),len(estimation_output.simulation_results_per_iteration))
            #
            #     rsw_state_difference = get_rsw_state_difference(
            #         estimated_state_history, spacecraft_name, spacecraft_central_body, global_frame_orientation)
            #     save2txt(rsw_state_difference, 'fit_spice_rsw_state_difference_' + filename_suffix + '.dat', current_directory)
            #
            #     # Estimated parameters
            #     print("estimated parameters", estimation_output.parameter_history[-1])


if __name__ == "__main__":
    print('Start')
    inputs = []

    nb_cores = 8

    start_date = datetime(2012, 3, 30)
    end_date = datetime(2012, 4, 2)
    # clock_files, orientation_files_to_load, tro_files_to_load, ion_files_to_load, manoeuvres_file, \
    # antenna_files_to_load, odf_files_to_load = get_grail_files("kernels_test/", start_date, end_date)

    # nb_arcs = len(odf_files_to_load)

    # nb_arcs_per_core = int(nb_arcs//nb_cores)


    for i in range(nb_cores):
        inputs.append(i)

    print('inputs', inputs)

    # Run parallel MC analysis
    with mp.get_context("fork").Pool(nb_cores) as pool:
        pool.map(run_estimation,inputs)




