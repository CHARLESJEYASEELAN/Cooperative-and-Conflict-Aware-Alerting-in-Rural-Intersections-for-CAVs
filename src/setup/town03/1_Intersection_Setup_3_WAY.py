import carla
import random
import time
import traceback
from Utility_Carla import BIKES, TRUCKS, VANS, CARS
from Utility_Carla import get_junction_centroid, set_spectator_top_view, draw_radius_circle, draw_spawn_point_markers, draw_intersection_labels
from Utility_Vehicles import spawn_directional_vehicles, configure_traffic_manager, destroy_vehicles

CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town03"
TM_PORT = 8000       



def draw_direction_lines(world, center, directions, length=30.0, z=2.0):
    vector_map = {
        "N": (0, length), "S": (0, -length),
        "E": (-length, 0), "W": (length, 0),
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

    junction_id = 1352 # 238 4-way, 861 5-way, 1352, 3-way...
    spawnPoints = None

    # Define direction configs for each junction
    # ...existing code...

    direction_configs = {
        238: {
            "E":  (194, TRUCKS, 60),   # East: Truck, spawn id 194, speed 60
            "S":  (44, VANS, 68),      # South: Van, spawn id 44, speed 68
            "W":  (30, BIKES, 55),     # West: Bike, spawn id 30, speed 55
            "N":  (165, CARS, 48)       # North: Car, spawn id 52, speed 48
        },
        861: {
            "N":  (60, CARS, 48),
            "E":  (61, TRUCKS, 60),
            "SE": (62, VANS, 68),
            "S":  (36, BIKES, 55),
            "W":  (228, CARS, 48)
        },
        1352: {
            "E":  (253, VANS, 45),
            "N":  (43, CARS, 62),
            "S":  (223, CARS, 55)
        }
    }
    # ...existing code...
    

    if junction_id == 238:
        spawnPoints = [
            44, 106, 34, 33, 32, # East
            110, 108, 107,28, # South        
            67, 36, 35, 168, 167, 165, # West 
            194, 195   # North 
        ]
    elif junction_id == 861:
        spawnPoints = [
            60,61,62,36,228,85,11,13,15,25,98,190,254,90
        ]
    elif junction_id == 1352:
        spawnPoints = [
            253,115,116,45,34,3,223,84,83
        ]

    print(f"Analyzing Junction ID: {junction_id}")
    intersection_loc = get_junction_centroid(carla_map, junction_id)
    # Custom directions per intersection
    if junction_id == 861:
        directions = ["N", "E", "SE", "S", "W"]
    elif junction_id == 1352:
        directions = ["E", "N", "S"]
    else:
        directions = ["W", "N", "E", "S"]  # Default

    draw_direction_lines(world, intersection_loc, directions)
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
                draw_radius_circle(world, intersection_loc, 62, carla.Color(0, 0, 255))  # Blue outer
                draw_radius_circle(world, intersection_loc, 58, carla.Color(255, 0, 0))  # Red congestion
                draw_radius_circle(world, intersection_loc, 49, carla.Color(0, 0, 255))  # Blue outer

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
