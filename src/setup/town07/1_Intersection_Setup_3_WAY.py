import carla
import random
import time
import traceback
from Utility_Carla import BIKES, TRUCKS, VANS, CARS
from Utility_Carla import get_junction_centroid, set_spectator_top_view, draw_radius_circle, draw_spawn_point_markers, draw_intersection_labels
from Utility_Vehicles import destroy_vehicles

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town07"
TM_PORT = 8000       

def configure_traffic_manager(tm, vehicles):
    for vehicle in vehicles:
        tm.ignore_lights_percentage(vehicle, 100)         # Obey traffic lights
        tm.ignore_signs_percentage(vehicle, 100)          # Obey stop/yield signs
        tm.ignore_vehicles_percentage(vehicle, 100)      # Mostly respect other cars (0-100)
        tm.auto_lane_change(vehicle, True)              # Allow lane changes when safe
        tm.distance_to_leading_vehicle(vehicle, 1.0)    # Following distance (meters)
        # tm.vehicle_percentage_speed_difference(vehicle, 10)  # Obey speed limits
        # tm.set_collision_detection(vehicle, True)     # not needed - default is collision aware

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
        # Override only spawn id 83 with your custom waypoint coordinate
        if spawn_id == 83:
            loc = carla.Location(x=21.4, y=118.4, z=0.0)
            wp = world.get_map().get_waypoint(
                loc,
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
            if wp is None:
                print("Custom waypoint not found on a driving lane.")
                continue

            transform = wp.transform
            transform.location.z += 0.5  # lift slightly to avoid ground collision

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

def draw_direction_lines(world, center, directions, length=30.0, z=2.0):
    vector_map = {
        "E": (0, length), "W": (0, -length),
        "S": (-length, 0), "N": (length, 0),
        "SE": (-length, -length)
    }

    color_map = {
        "N": carla.Color(255, 0, 0),
        "S": carla.Color(0, 255, 0),
        "E": carla.Color(0, 0, 255),
        "W": carla.Color(255, 255, 0),
        "SE": carla.Color(255, 0, 255)
    }

    for dir in directions:
        if dir not in vector_map:
            continue
        dx, dy = vector_map[dir]
        start = carla.Location(center.x, center.y, center.z + z)
        end = carla.Location(center.x + dx, center.y + dy, center.z + z)

        world.debug.draw_line(start, end, thickness=0.4, color=color_map[dir],
                              life_time=180.0, persistent_lines=True)
        world.debug.draw_string(end, dir, color=color_map[dir],
                                life_time=180.0, persistent_lines=True)


# In your main loop, replace the call to spawn_fixed_autopilot_vehicles with:
# vehicles, tm = spawn_directional_vehicles(world, client, TM_PORT)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(10.0)

    print(f"Loading world: {TOWN}")
    world = client.load_world(TOWN)
    carla_map = world.get_map()
    print(f"Current map: {carla_map.name}")
    settings = world.get_settings()
    settings.no_rendering_mode = False
    world.apply_settings(settings)

    #draw_spawn_point_markers(world, carla_map)
    #draw_intersection_labels(world, carla_map)

    junction_id = 335 # 335 - 3 way Y shaped, 68 -4way irregular

    # Define direction configs for each junction
    # ...existing code...

    direction_configs = {
        335: {
            "E":  (83, CARS, 80), # Custom spawn point for E direction
            "W":  (20, VANS, 62),
            "S":  (64, CARS, 55)
        }
    }
    # ...existing code...
    
    print(f"Analyzing Junction ID: {junction_id}")
    intersection_loc = get_junction_centroid(carla_map, junction_id)

    

    if intersection_loc:
        set_spectator_top_view(world, intersection_loc, height=120.0)

    print("\nSimulation running. Press Ctrl+C to stop.")
    episode_count = 0
    while episode_count < 5:
        print(f"\n========== STARTING EPISODE {episode_count + 1} ==========")

        vehicles, tm = spawn_directional_vehicles(
            world, client, TM_PORT, direction_configs[junction_id]
        )
        
        if vehicles:
            configure_traffic_manager(tm, vehicles)

        try:
            while True:
                time.sleep(1)  # keep script alive (async mode)
                draw_radius_circle(world, intersection_loc, 60, carla.Color(0, 0, 255))  # Blue outer
                draw_direction_lines(world, intersection_loc, ["S", "E", "W"])
                #draw_radius_circle(world, intersection_loc, 58, carla.Color(255, 0, 0))  # Red congestion
                #draw_radius_circle(world, intersection_loc, 49, carla.Color(0, 0, 255))  # Blue outer

        except KeyboardInterrupt:
            print("\nCleaning up...")
            destroy_vehicles(vehicles)
            break  # Exit the episode loop on Ctrl+C
            
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
