import carla
import random
import time
import traceback
from Utility_Carla import BIKES, TRUCKS, VANS, CARS
from Utility_Carla import get_junction_centroid, set_spectator_top_view, draw_radius_circle, draw_spawn_point_markers, draw_intersection_labels
from Utility_Vehicles import spawn_directional_vehicles, configure_traffic_manager, destroy_vehicles
import pandas as pd 


import numpy as np
import math
import os

DIRECTION_INDEX = {"N":0, "E":1, "S":2, "W":3}
TYPE_INDEX = {"car":0, "truck":1, "van":2, "bike":3}


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

def get_vehicle_type(vehicle):
    blueprint = vehicle.type_id.lower()
    if blueprint in TRUCKS:
        return 1
    elif blueprint in VANS:
        return 2
    elif blueprint in BIKES:
        return 3
    else:
        return 0  # car
    
def extract_graph_state(vehicles, intersection_loc, direction_keys):

    node_features = []

    for idx, veh in enumerate(vehicles):

        loc = veh.get_location()
        vel = veh.get_velocity()

        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        dist = loc.distance(intersection_loc)
        yaw = veh.get_transform().rotation.yaw

        # ----- Direction one-hot -----
        dir_onehot = np.zeros(len(direction_keys))
        dir_onehot[idx] = 1.0   # order matches spawn order

        # ----- Type one-hot -----
        type_onehot = np.zeros(4)
        type_idx = get_vehicle_type(veh)
        type_onehot[type_idx] = 1.0

        features = np.array([
            loc.x,
            loc.y,
            speed,
            dist,
            yaw
        ])

        full_feature = np.concatenate([features, dir_onehot, type_onehot])
        node_features.append(full_feature)

    return np.array(node_features, dtype=np.float32)


def construct_multi_relational_edges(vehicles, intersection_location):

    n = len(vehicles)
    edge_tensor = np.zeros((3, n, n))  # 3 edge types

    epsilon = 0.1

    for i in range(n):
        for j in range(n):

            if i == j:
                continue

            vi = vehicles[i]
            vj = vehicles[j]

            loc_i = vi.get_location()
            loc_j = vj.get_location()

            speed_i = math.sqrt(
                vi.get_velocity().x**2 +
                vi.get_velocity().y**2 +
                vi.get_velocity().z**2
            )

            speed_j = math.sqrt(
                vj.get_velocity().x**2 +
                vj.get_velocity().y**2 +
                vj.get_velocity().z**2
            )

            distance = loc_i.distance(loc_j)

            # Edge 0: Proximity
            if distance < 40:
                edge_tensor[0, i, j] = 1

            # Edge 1: Conflict geometry
            delta_theta = abs(
                vi.get_transform().rotation.yaw -
                vj.get_transform().rotation.yaw
            )
            delta_theta = math.radians(delta_theta)
            H_ij = abs(math.sin(delta_theta))

            if H_ij > 0.5:
                edge_tensor[1, i, j] = 1

            # Edge 2: TTC risk
            v_rel = abs(speed_i - speed_j)
            TTC = distance / (v_rel + epsilon)

            if TTC < 5.0:
                edge_tensor[2, i, j] = 1

    return edge_tensor


























def compute_total_reward(world,
                         vehicles,
                         intersection_location,
                         collision_flags):
    """
    world: carla.World
    vehicles: list of carla.Vehicle actors
    intersection_location: carla.Location (center of intersection)
    collision_flags: dict {vehicle_id: True/False}

    Returns:
        total_reward (float)
    """

    # ----------------------------
    # GLOBAL PARAMETERS (FROZEN)
    # ----------------------------

    w_s = 3.0
    w_e = 1.0
    w_p = 1.0

    alpha = 50.0
    beta = 2.0
    tau = 3.0
    epsilon = 0.1

    gamma = 1.0
    delta = 1.5
    T_max = 60.0

    lambda_1 = 0.4
    lambda_2 = 0.2
    lambda_3 = 0.2
    lambda_4 = 0.2
    mu = 1.0

    # ----------------------------
    # VEHICLE TYPE WEIGHTS
    # ----------------------------

    def get_type_weight(vehicle):
        blueprint = vehicle.type_id.lower()
        if blueprint in TRUCKS:
            return 1.5, 40.0
        elif blueprint in VANS:
            return 1.2, 50.0
        elif blueprint in BIKES:
            return 0.8, 65.0
        else:
            return 1.0, 60.0  # default car

    # ----------------------------
    # HELPER FUNCTIONS
    # ----------------------------

    def get_speed(vehicle):
        vel = vehicle.get_velocity()
        return math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    def get_yaw(vehicle):
        return vehicle.get_transform().rotation.yaw

    def get_distance(loc1, loc2):
        return math.sqrt(
            (loc1.x - loc2.x)**2 +
            (loc1.y - loc2.y)**2 +
            (loc1.z - loc2.z)**2
        )

    def compute_conflict_indicator(v1, v2):
        """
        You should replace this with your
        direction-based conflict logic.
        """
        # Simple geometric approximation:
        # If both within 30m of intersection and
        # coming from different axes → conflict
        loc1 = v1.get_location()
        loc2 = v2.get_location()

        d1 = get_distance(loc1, intersection_location)
        d2 = get_distance(loc2, intersection_location)

        if d1 < 30 and d2 < 30:
            return 1
        return 0

    # ----------------------------
    # MAIN REWARD LOOP
    # ----------------------------

    total_reward = 0.0
    n = len(vehicles)

    for i, veh_i in enumerate(vehicles):

        loc_i = veh_i.get_location()
        speed_i = get_speed(veh_i)
        yaw_i = get_yaw(veh_i)

        type_weight_i, v_desired_i = get_type_weight(veh_i)

        distance_to_intersection_i = get_distance(loc_i, intersection_location)

        waiting_i = 0.0
        if speed_i < 0.5:
            waiting_i = 1.0  # You can accumulate externally instead

        # ----------------------------
        # SAFETY COMPONENT
        # ----------------------------

        R_safety = 0.0

        # Collision penalty
        if collision_flags.get(veh_i.id, False):
            R_safety -= alpha

        conflict_count = 0

        for j, veh_j in enumerate(vehicles):
            if i == j:
                continue

            loc_j = veh_j.get_location()
            speed_j = get_speed(veh_j)
            yaw_j = get_yaw(veh_j)

            type_weight_j, _ = get_type_weight(veh_j)

            d_ij = get_distance(loc_i, loc_j)
            v_rel = abs(speed_i - speed_j)

            TTC = d_ij / (v_rel + epsilon)

            delta_theta = abs(yaw_i - yaw_j)
            delta_theta = math.radians(delta_theta)

            H_ij = abs(math.sin(delta_theta))

            C_ij = compute_conflict_indicator(veh_i, veh_j)

            if C_ij == 1:
                conflict_count += 1

            W_type = (type_weight_i + type_weight_j) / 2.0

            risk_term = (
                beta
                * C_ij
                * H_ij
                * W_type
                * math.exp(-TTC / tau)
            )

            R_safety -= risk_term

        # ----------------------------
        # EFFICIENCY COMPONENT
        # ----------------------------

        speed_term = speed_i / v_desired_i
        waiting_term = waiting_i / T_max

        R_efficiency = gamma * speed_term - delta * waiting_term

        # ----------------------------
        # PRIORITY COMPONENT
        # ----------------------------

        if n > 1:
            conflict_density = conflict_count / (n - 1)
        else:
            conflict_density = 0.0

        priority_value = (
            lambda_1 * (waiting_i / T_max)
            + lambda_2 * (1.0 / (distance_to_intersection_i + 1.0))
            + lambda_3 * (speed_i / v_desired_i)
            + lambda_4 * conflict_density
        )

        R_priority = mu * priority_value

        # ----------------------------
        # TOTAL VEHICLE REWARD
        # ----------------------------

        R_i = w_s * R_safety + w_e * R_efficiency + w_p * R_priority

        total_reward += R_i
    return total_reward


def attach_collision_sensors(world, vehicles, collision_flags):

    blueprint_library = world.get_blueprint_library()
    collision_bp = blueprint_library.find('sensor.other.collision')

    sensors = []

    for vehicle in vehicles:
        sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)

        def callback(event, vehicle_id=vehicle.id, collision_flags=collision_flags):
            
            try:
                other_actor = event.other_actor

                if other_actor is not None and "vehicle" in other_actor.type_id:
                    collision_flags[vehicle_id] = True
            except RuntimeError:
                pass  # Actor might have been destroyed

        sensor.listen(callback)
        sensors.append(sensor)

    return sensors














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
    settings.no_rendering_mode = True
    settings.fixed_delta_seconds = 0.05
    settings.synchronous_mode = True
    world.apply_settings(settings)
    
    collision_flags = {}

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
        set_spectator_top_view(world, intersection_loc, height=125.0)

    

    print("\nSimulation running. Press Ctrl+C to stop.")
    episode_count = 0
    while episode_count < 100:
        print(f"\n========== STARTING EPISODE {episode_count + 1} ==========")

        episode_states = []
        episode_edges = []
        episode_rewards = []

        max_steps = 200
        step = 0

        direction_keys = list(direction_configs[junction_id].keys())

        vehicles, tm = spawn_directional_vehicles(
            world, client, TM_PORT, direction_configs[junction_id]
        )
        
        if vehicles:
            tm.set_synchronous_mode(True)
            configure_traffic_manager(tm, vehicles)
        
        collision_flags = {v.id: False for v in vehicles}
        sensors = attach_collision_sensors(world, vehicles, collision_flags)


        try:
            while step < max_steps:
                
                world.tick()
                time.sleep(0.005)
                
                alive_vehicles = [v for v in vehicles if v.is_alive]
                if len(alive_vehicles) < len(vehicles):
                    print("Warning: vehicle destroyed mid-episode")

                node_features = extract_graph_state(alive_vehicles, intersection_loc, direction_keys)
                #node_features = extract_graph_state(vehicles, intersection_loc, direction_keys)
                edges = construct_multi_relational_edges(alive_vehicles, intersection_loc)

                
                reward = compute_total_reward(world, alive_vehicles, intersection_loc, collision_flags)

                #print(reward)

                episode_states.append(node_features)
                episode_edges.append(edges)
                episode_rewards.append(reward)

                step += 1
            print("Episode is Done")
        except KeyboardInterrupt:
            print("\nCleaning up...")
            destroy_vehicles(vehicles)
            break  # Exit the episode loop on Ctrl+C

        for sensor in sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()
        
        tm.set_synchronous_mode(False)
        for v in vehicles:
            if v.is_alive:
                v.set_autopilot(False)
        world.tick()  # Let vehicles stop before destroying
        destroy_vehicles(vehicles)
        #world.tick()  # Ensure cleanup is processed

        print("vehicle Destroyed")  

        gamma_discount = 0.99
        returns = []
        G = 0.0
        for r in reversed(episode_rewards):
            G = r + gamma_discount * G
            returns.insert(0, G)

        os.makedirs("dataset_3Way", exist_ok=True)
        np.save(f"dataset_3Way/episode_{episode_count}.npy", {
            "nodes": np.array(episode_states),
            "edges": np.array(episode_edges),
            "returns": np.array(returns)
        })

        

        print(f"Episode {episode_count+1} saved.")
        episode_count += 1

    
        
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
            
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()
