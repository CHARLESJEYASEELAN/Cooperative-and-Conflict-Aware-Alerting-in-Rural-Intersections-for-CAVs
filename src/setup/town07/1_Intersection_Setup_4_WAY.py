import carla
import random
import time
import traceback
from Utility_Carla import BIKES, TRUCKS, VANS, CARS, get_junction_centroid
from Utility_Carla import set_spectator_top_view, draw_radius_circle
from Utility_Vehicles import spawn_directional_vehicles, destroy_vehicles

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

    junction_id = 68 # 238 4-way, 861 5-way, 1352, 3-way...

    # Define direction configs for each junction
    # ...existing code...

    direction_configs = {
        68: {
            "E":  (82, TRUCKS, 60),   # East: Truck, spawn id 194, speed 60
            "S":  (49, VANS, 68),      # South: Van, spawn id 44, speed 68
            "W":  (76, BIKES, 55),     # West: Bike, spawn id 30, speed 55
            "N":  (13, CARS, 48)       # North: Car, spawn id 52, speed 48
        }
    }
    # ...existing code...
    

    print(f"Analyzing Junction ID: {junction_id}")
    intersection_loc = get_junction_centroid(carla_map, junction_id)

    #draw_direction_lines(world, intersection_loc, ["W", "N", "E", "S"])
    if intersection_loc:
        set_spectator_top_view(world, intersection_loc, height=110.0)

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
                #draw_radius_circle(world, intersection_loc, 85, carla.Color(0, 0, 255))  # Blue outer
                #draw_radius_circle(world, intersection_loc, 80, carla.Color(255, 0, 0))  # Red congestion
                #draw_radius_circle(world, intersection_loc, 74, carla.Color(0, 0, 255))  # Blue outer
                #draw_radius_circle(world, intersection_loc, 52, carla.Color(255, 0, 0))  # Red congestion

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