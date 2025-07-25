import traceback
import sys
import subprocess
import glob
import shutil
import os
import concurrent.futures
from datetime import datetime

try:
    sys.path.append(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    print(glob.glob('./carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64')))
except IndexError:
    pass

import carla
from carla import Transform, Location, Rotation
import argparse
import logging
from npc_spawning import spawnWalkers, spawnVehicles
from configuration import attachSensorsToVehicle, SimulationParams, setupTrafficManager, setupWorld, setupWorldWeather, createOutputDirectories, CarlaSyncMode
import save_sensors
import random
import json
import time
import queue
import os
from os import path
from ego_vehicle import EgoVehicle
from fixed_perception import FixedPerception
from utils.arg_parser import CommandLineArgsParser
from utils.weather import weather_presets


def main():

    args_parser = CommandLineArgsParser()
    args = args_parser.parse_args()
    print(args)

    # Create CARLA client
    client = carla.Client(args.host, args.port)
    client.set_timeout(args.timeout)

    # Setup simulation parameters
    SimulationParams.town_map = args.map
    SimulationParams.num_of_walkers = args.number_of_walkers
    SimulationParams.num_of_vehicles = args.number_of_vehicles
    SimulationParams.delta_seconds = args.delta_seconds
    SimulationParams.ignore_first_n_ticks = args.ignore_first_n_ticks
    # TODO: Is > 1 ego vehicle really required?
    SimulationParams.number_of_ego_vehicles = args.number_of_ego_vehicles
    SimulationParams.PHASE = SimulationParams.town_map + \
        "_" + SimulationParams.dt_string
    SimulationParams.data_output_subfolder = os.path.join(
        "out/", SimulationParams.PHASE)
    SimulationParams.manual_control = args.manual_control
    SimulationParams.fixed_perception = args.fixed_perception
    SimulationParams.res = args.res
    SimulationParams.autopilot = args.autopilot
    SimulationParams.debug = args.debug
    SimulationParams.start_weather = args.start_weather
    SimulationParams.end_weather = args.end_weather
    SimulationParams.duration = args.duration

    world = client.load_world(SimulationParams.town_map)
    world = client.get_world()

    # Remove all parked vehicles etc.
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Car)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Bicycle)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Bus)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Motorcycle)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Pedestrians)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Train)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)
    env_objs = world.get_environment_objects(carla.CityObjectLabel.Truck)
    for i in range(0, len(env_objs)):
        world.enable_environment_objects({env_objs[i].id}, False)

    # Setup
    setupWorld(world)
    setupTrafficManager(client)

    # Get all required blueprints
    blueprint_library = world.get_blueprint_library()
    blueprintsVehicles = blueprint_library.filter('vehicle.*')
    vehicles_spawn_points = world.get_map().get_spawn_points()
    blueprintsWalkers = blueprint_library.filter('walker.pedestrian.*')
    walker_controller_bp = blueprint_library.find('controller.ai.walker')
    walkers_spawn_points = world.get_random_location_from_navigation()
    lidar_segment_bp = blueprint_library.find('sensor.lidar.ray_cast')

    participant_density = {
        'bicycle': 0,
        'truck': 0,
        'van': 0,
        'car': 0,
        'motorcycle': 0,
        'pedestrian': 0
    }

    w_all_actors, w_all_id = spawnWalkers(
        client, world, blueprintsWalkers, SimulationParams.num_of_walkers)
    for actor in w_all_actors:
        actor_type = actor.attributes['role_name']
        if actor_type == "pedestrian":
            participant_density["pedestrian"] += 1
    world.tick()

    egos = []
    fixed = []
    map_name = world.get_map().name

    if SimulationParams.fixed_perception == True:
        with open(SimulationParams.fixed_perception_sensor_locations_json_filepath, 'r') as json_file:
            sensor_locations = json.load(json_file)
        SimulationParams.town_map = map_name.split("/")[-1]
        for config_entry in sensor_locations:
            if config_entry["town"] == map_name:
                for coordinate in config_entry["cordinates"]:
                    fixed.append(FixedPerception(
                        SimulationParams.fixed_perception_sensor_json_filepath, None, world, args, coordinate))

    for i in range(SimulationParams.number_of_ego_vehicles):
        egos.append(EgoVehicle(
            SimulationParams.sensor_json_filepath, None, world, args))

    v_all_actors, v_all_id = spawnVehicles(
        client, world, vehicles_spawn_points, blueprintsVehicles, SimulationParams.num_of_vehicles)

    for actor in v_all_actors:
        actor_type = actor.attributes.get('base_type')

        if actor_type in participant_density:
            participant_density[actor_type] += 1
    world.tick()

    print("Starting simulation...")

    def process_egos(i, frame_id):
        data = egos[i].getSensorData(frame_id)
        output_folder = os.path.join(
            SimulationParams.data_output_subfolder, "ego" + str(i))
        try:
            save_sensors.saveAllSensors(
                output_folder, data, egos[i].sensor_names, world)
            control = egos[i].ego.get_control()
            angle = control.steer
            save_sensors.saveSteeringAngle(angle, output_folder)
        except Exception as error:
            print("An exception occurred in egos - perception and control saving:", error)
            traceback.print_exc()

    def process_fixed(i, frame_id):
        data = fixed[i].getSensorData(frame_id)
        output_folder = os.path.join(
            SimulationParams.data_output_subfolder, "fixed-" + str(i+1))
        try:
            save_sensors.saveAllSensors(
                output_folder, data, fixed[i].sensor_names, world)
        except Exception as error:
            print("An exception occurred in fixed - perception saving:", error)
            traceback.print_exc()

    def interpolate_weather(start_weather, end_weather, progress):
        weather = carla.WeatherParameters(
            cloudiness=start_weather.cloudiness +
            (end_weather.cloudiness - start_weather.cloudiness) * progress,
            dust_storm=start_weather.dust_storm +
            (end_weather.dust_storm - start_weather.dust_storm) * progress,
            fog_density=start_weather.fog_density +
            (end_weather.fog_density - start_weather.fog_density) * progress,
            fog_distance=start_weather.fog_distance +
            (end_weather.fog_distance - start_weather.fog_distance) * progress,
            fog_falloff=start_weather.fog_falloff +
            (end_weather.fog_falloff - start_weather.fog_falloff) * progress,
            mie_scattering_scale=start_weather.mie_scattering_scale,
            precipitation=start_weather.precipitation +
            (end_weather.precipitation - start_weather.precipitation) * progress,
            precipitation_deposits=start_weather.precipitation_deposits +
            (end_weather.precipitation_deposits -
             start_weather.precipitation_deposits) * progress,
            rayleigh_scattering_scale=start_weather.rayleigh_scattering_scale,
            scattering_intensity=start_weather.scattering_intensity +
            (end_weather.scattering_intensity -
             start_weather.scattering_intensity) * progress,
            sun_azimuth_angle=start_weather.sun_azimuth_angle +
            (end_weather.sun_azimuth_angle -
             start_weather.sun_azimuth_angle) * progress,
            sun_altitude_angle=start_weather.sun_altitude_angle +
            (end_weather.sun_altitude_angle -
             start_weather.sun_altitude_angle) * progress,
            wind_intensity=start_weather.wind_intensity +
            (end_weather.wind_intensity - start_weather.wind_intensity) * progress,
            wetness=start_weather.wetness +
            (end_weather.wetness - start_weather.wetness) * progress
        )
        return weather

    start_weather = SimulationParams.start_weather
    end_weather = SimulationParams.end_weather
    duration = SimulationParams.duration
    metadata = {
        "start_weather": start_weather,
        "end_weather": end_weather,
        "duration": duration,
        "map_name": map_name,
        "participant_density": participant_density,
        "delta_seconds": SimulationParams.delta_seconds,
        "egos": len(egos),
        "fixed-views": len(fixed)
    }

    for name, value in weather_presets:
        if name == start_weather:
            start_weather = value
            break

    for name, value in weather_presets:
        if name == end_weather:
            end_weather = value
            break

    world.set_weather(start_weather)

    step = 0
    k = 0

    json_string = json.dumps(metadata, indent=4)
    file_path = f'./out/metadata-{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    with open(file_path, "w") as file:
        file.write(json_string)
    try:
        with CarlaSyncMode(world, []) as sync_mode:
            while True:
                frame_id = sync_mode.tick(timeout=5.0)
                if (k < SimulationParams.ignore_first_n_ticks):
                    k = k + 1
                    print("Ignore Count: ", k)
                    continue

                if step > duration:
                    break

                print("Frame: ", step)
                step = step + 1

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(
                        process_egos, i, frame_id) for i in range(len(egos))]
                    concurrent.futures.wait(futures)

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(
                        process_fixed, i, frame_id) for i in range(len(fixed))]
                    concurrent.futures.wait(futures)

                progress = step / duration
                current_weather = interpolate_weather(
                    start_weather, end_weather, progress)
                world.set_weather(current_weather)
    finally:
        # stop pedestrians (list is [controller, actor, controller, actor ...])
        for i in range(0, len(w_all_actors)):
            try:
                w_all_actors[i].stop()
            except:
                pass
        # destroy pedestrian (actor and controller)
        client.apply_batch([carla.command.DestroyActor(x) for x in w_all_id])
        client.apply_batch([carla.command.DestroyActor(x) for x in v_all_id])

        for ego in egos:
            ego.destroy()

        # This is to prevent Unreal from crashing from waiting the client.
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)


if __name__ == '__main__':
    try:
        # assert len(sys.argv) > 1, "no path for destination folder given.."
        main()
    except KeyboardInterrupt:
        pass
