# ------------------------------------------------------------
# CONFIGURE TRAFFIC MANAGER RULES
# ------------------------------------------------------------

import random
def configure_traffic_manager(tm, vehicles):
    for vehicle in vehicles:
        tm.ignore_lights_percentage(vehicle, 0)         # Obey traffic lights
        tm.ignore_signs_percentage(vehicle, 0)          # Obey stop/yield signs
        tm.ignore_vehicles_percentage(vehicle, 10)      # Mostly respect other cars (0-100)
        tm.auto_lane_change(vehicle, True)              # Allow lane changes when safe
        tm.distance_to_leading_vehicle(vehicle, 5.0)    # Following distance (meters)
        tm.vehicle_percentage_speed_difference(vehicle, 10)  # Obey speed limits
        # tm.set_collision_detection(vehicle, True)     # not needed - default is collision aware

    print("Traffic Manager rules applied to all spawned vehicles.")




def destroy_vehicles(vehicles):
    for v in vehicles:
        try:
            if v is not None and v.is_alive:
                v.destroy()
        except RuntimeError as e:
            print(f"Error destroying vehicle: {e}")
    vehicles.clear()



def spawn_directional_vehicles(world, client, tm_port, direction_config):
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(False)

    vehicles = []

    for direction, (spawn_id, vehicle_list, speed_mph) in direction_config.items():
        if spawn_id >= len(spawn_points):
            print(f"Invalid spawn index for {direction}: {spawn_id}")
            continue

        allowed_blueprints = [bp for bp in blueprint_library.filter('vehicle.*') if bp.id in vehicle_list]
        if not allowed_blueprints:
            print(f"No allowed blueprints for {direction}")
            continue

        transform = spawn_points[spawn_id]
        if world.get_actors().filter('vehicle.*') and any(
            v.get_transform().location.distance(transform.location) < 4.0
            for v in world.get_actors().filter('vehicle.*')
        ):
            print(f"Spawn point {spawn_id} ({direction}) too close to existing vehicle → skipped")
            continue

        blueprint = random.choice(allowed_blueprints)
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is None:
            print(f"Spawn failed at {direction} (index {spawn_id})")
            continue

        vehicle.set_autopilot(True, tm_port)
        speed_kph = speed_mph * 1.60934
        tm.vehicle_percentage_speed_difference(vehicle, 100 - (speed_kph / 90 * 100))
        vehicles.append(vehicle)
        print(f"Spawned {direction} vehicle {vehicle.id} ({blueprint.id}) at {spawn_id} with {speed_mph} mph")

    return vehicles, tm