import carla
import torch
import numpy as np
import time
import os
import csv

from Utility_Carla import BIKES, TRUCKS, VANS, CARS


# ==== YOUR EXISTING IMPORTS ====
from Utility_RL import (
    CustomGraphModel,
    build_graph,
    spawn_directional_vehicles,
    get_junction_centroid,
    compute_total_reward,
    initialize_episode_metrics,
    attach_collision_sensors,
    compute_final_metrics
)

vehicle_arrival_time = {}  # vid -> timestamp when STOP alert first received
vehicle_strict_stop_time = {}

def update_vehicle_tracking(vehicles, intersection_loc, metrics, congestion_radius=20.0):
    current_time = time.time()
    current_ids = set()
    for v in vehicles:
        if not v.is_alive:
            continue
        vid = v.id
        loc = v.get_location()
        speed = np.linalg.norm([v.get_velocity().x, v.get_velocity().y, v.get_velocity().z])
        dist = loc.distance(intersection_loc)
        current_ids.add(vid)
        if vid not in metrics["entry_times"]:
            metrics["entry_times"][vid] = current_time
        metrics["total_speed"] += speed
        metrics["speed_samples"] += 1
    queue_count = sum(
        1 for v in vehicles
        if np.linalg.norm([v.get_velocity().x, v.get_velocity().y, v.get_velocity().z]) < 0.5 and
           v.get_location().distance(intersection_loc) < congestion_radius
    )
    metrics["queue_lengths"].append(queue_count)
    if queue_count > metrics["max_queue"]:
        metrics["max_queue"] = queue_count
    for vid in list(metrics["entry_times"].keys()):
        if vid not in current_ids and vid not in metrics["exit_times"]:
            metrics["exit_times"][vid] = current_time
            metrics["vehicles_passed"].add(vid)

import time
import carla

CONGESTION_RADIUS = 15.0
STOP_LINE_RADIUS  = 16.5
CLEAR_RADIUS      = 5.0
MAX_WAIT_TIME     = 8.0

# store waiting time per vehicle
vehicle_wait_dict = {}

vehicle_stop_state = {}

def hybrid_liveness_controller(vehicle,
                                alert,
                                policy_control,
                                intersection_center,
                                world):

    location = vehicle.get_location()
    distance = location.distance(intersection_center)

    vehicle_id = vehicle.id

    # initialize waiting timer
    if vehicle_id not in vehicle_wait_dict:
        vehicle_wait_dict[vehicle_id] = {
            "start_wait": None,
            "inside_zone": False
        }

    # detect if vehicle entered congestion zone
    if distance <= CONGESTION_RADIUS:
        vehicle_wait_dict[vehicle_id]["inside_zone"] = True

    # ---------------------------------------------
    # CASE 1: STOP ALERT
    # ---------------------------------------------
    if alert == "STOP":

        # If vehicle hasn't crossed stop line
        if distance > STOP_LINE_RADIUS:
            control = carla.VehicleControl()
            control.throttle = 0.0
            control.brake = 1.0
            control.hand_brake = True
            return control

        # If already inside intersection, clear it safely
        else:
            control = carla.VehicleControl()
            control.throttle = 0.1
            control.brake = 0.0
            return control

    # ---------------------------------------------
    # CASE 2: PROCEED ALERT
    # ---------------------------------------------
    elif alert == "MOVE":

        # If someone else is still inside core zone
        actors = world.get_actors().filter("*vehicle*")
        for other in actors:
            if other.id == vehicle_id:
                continue

            other_loc = other.get_location()
            other_dist = other_loc.distance(intersection_center)

            if other_dist < CLEAR_RADIUS:
                # someone still clearing → wait
                control = carla.VehicleControl()
                control.throttle = 0.0
                control.brake = 1.0
                return control

        # Deadlock prevention
        if vehicle_wait_dict[vehicle_id]["start_wait"] is None:
            vehicle_wait_dict[vehicle_id]["start_wait"] = time.time()

        waited = time.time() - vehicle_wait_dict[vehicle_id]["start_wait"]

        if waited > MAX_WAIT_TIME:
            # force movement
            control = carla.VehicleControl()
            control.throttle = 0.35
            control.brake = 0.0
            return control

        # Otherwise allow PPO policy output
        return policy_control

    # ---------------------------------------------
    # DEFAULT FALLBACK
    # ---------------------------------------------
    return policy_control



def set_spectator_top_view(world, location, height=120.0):
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla.Location(location.x, location.y, location.z + height),
        carla.Rotation(pitch=-90.0)
    )
    spectator.set_transform(transform)

def configure_traffic_manager(tm, vehicles):
    for vehicle in vehicles:
        tm.ignore_lights_percentage(vehicle, 100)
        tm.ignore_signs_percentage(vehicle, 100)
        tm.ignore_vehicles_percentage(vehicle, 75)
        tm.auto_lane_change(vehicle, True)
        tm.distance_to_leading_vehicle(vehicle, 4.0)
        tm.vehicle_percentage_speed_difference(vehicle, -75)

def save_episode_metrics(csv_file, episode, episode_reward, final_metrics):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "episode",
                "episode_reward",
                "throughput",
                "avg_delay",
                "avg_queue",
                "max_queue",
                "collisions",
                "avg_speed"
            ])

        writer.writerow([
            episode,
            episode_reward,
            final_metrics.get("throughput", 0),
            final_metrics.get("avg_delay", 0),
            final_metrics.get("avg_queue", 0),
            final_metrics.get("max_queue", 0),
            final_metrics.get("collisions", 0),
            final_metrics.get("avg_speed", 0)
        ])

# ==== CONFIG ====
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town07"
TM_PORT = 8000
JUNCTION_ID = 68  # 4-way

MODEL_DIR = "RL_Model_4_WAY"
MODEL_EPISODE = 400                  # Change this if you want another checkpoint
MODEL_PATH = os.path.join(MODEL_DIR, f"ppo_ep{MODEL_EPISODE}.pth")

NUM_DEMO_EPISODES = 5                # How many episodes to show
MAX_STEPS = 900
SHOW_ALERTS = True  

DEADLOCK_SPEED_THRESHOLD = 0.5   # m/s
DEADLOCK_TIME = 20               # steps (~1 sec)
deadlock_counter = 0                 

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ============================
# PREPROCESS OBS
# ============================
def obs_to_tensor(obs_dict):
    return {
        k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        for k, v in obs_dict.items()
    }

# ============================
# DRAW ALERT (visual feedback)
# ============================
def draw_vehicle_alert(world, vehicle, alert_type):
    if not SHOW_ALERTS:
        return
    loc = vehicle.get_location()
    text_loc = carla.Location(loc.x, loc.y, loc.z + 2.5)
    if alert_type == "STOP":
        color = carla.Color(255, 0, 0)
    elif alert_type == "SLOW":
        color = carla.Color(255, 255, 0)
    else:
        color = carla.Color(0, 255, 0)
    world.debug.draw_string(
        text_loc, alert_type,
        draw_shadow=True, color=color,
        life_time=0.05, persistent_lines=False
    )

import math

def draw_radius_circle(world, center, radius, color, z_offset=0.5, segments=72):
    points = []

    for i in range(segments + 1):
        angle = 2 * math.pi * i / segments
        x = center.x + radius * math.cos(angle)
        y = center.y + radius * math.sin(angle)
        z = center.z + z_offset
        points.append(carla.Location(x, y, z))

    for i in range(len(points) - 1):
        world.debug.draw_line(
            points[i],
            points[i + 1],
            thickness=0.2,
            color=color,
            life_time=40.0,
            persistent_lines=False
        )

# ============================
# CONGESTION CONTROL LAYER
# ============================

CONGESTION_RADIUS = 15.0
QUEUE_LOOKAHEAD = 30.0
ALLOW_DIRECTIONS = 3

import math

def get_vehicle_direction(vehicle, intersection_loc):
    loc = vehicle.get_location()
    dx = loc.x - intersection_loc.x
    dy = loc.y - intersection_loc.y
    angle = math.atan2(dy, dx) * 180 / math.pi  # angle in degrees

    # Adjust these ranges based on your map
    if -45 <= angle < 45:
        return "E"
    elif 45 <= angle < 135:
        return "N"
    elif -135 <= angle < -45:
        return "S"
    else:
        return "W"
'''
def get_vehicle_direction(vehicle, intersection_loc):
    loc = vehicle.get_location()
    dx = loc.x - intersection_loc.x
    dy = loc.y - intersection_loc.y

    # Determine incoming direction
    if abs(dx) > abs(dy):
        return "E" if dx < 0 else "W"
    else:
        return "N" if dy < 0 else "S"
'''

def congestion_direction_selection(vehicles, actions_dict, intersection_loc):
    directions = ["N", "S", "E", "W"]

    # Stats
    alert_count = {d: 0 for d in directions}
    queue_count = {d: 0 for d in directions}

    for v in vehicles:
        if not v.is_alive:
            continue

        dist = v.get_location().distance(intersection_loc)
        d = get_vehicle_direction(v, intersection_loc)

        if dist <= CONGESTION_RADIUS:
            # count non-move alerts
            if actions_dict.get(v.id, 2) != 2:
                alert_count[d] += 1

        if dist <= QUEUE_LOOKAHEAD:
            queue_count[d] += 1

    # Priority score
    priority = {}
    for d in directions:
        priority[d] = (1.0 / (1 + alert_count[d])) + 0.3 * queue_count[d]

    # Select top directions
    allowed_dirs = sorted(priority, key=priority.get, reverse=True)[:ALLOW_DIRECTIONS]

    return allowed_dirs

def hard_stop_vehicle(vehicle, tm):
    tm.vehicle_percentage_speed_difference(vehicle, 100)
    vehicle.apply_control(
        carla.VehicleControl(
            throttle=0.0,
            brake=1.0,
            hand_brake=True
        )
    )


def detect_deadlock(vehicles, intersection_loc):
    inside = []
    for v in vehicles:
        if not v.is_alive:
            continue
        dist = v.get_location().distance(intersection_loc)
        if dist <= CONGESTION_RADIUS:
            speed = v.get_velocity()
            speed_mag = (speed.x**2 + speed.y**2)**0.5
            inside.append(speed_mag)

    if len(inside) < 2:
        return False

    # All nearly stopped
    return all(s < DEADLOCK_SPEED_THRESHOLD for s in inside)

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

        world.debug.draw_line(start, end, thickness=0.2, color=color_map[dir],
                              life_time=40, persistent_lines=True)
        world.debug.draw_string(end, dir, color=color_map[dir],
                                life_time=40, persistent_lines=True)
        

def is_moving_away(vehicle, intersection_loc):
    loc = vehicle.get_location()
    vel = vehicle.get_velocity()

    rx = loc.x - intersection_loc.x
    ry = loc.y - intersection_loc.y

    dot = rx * vel.x + ry * vel.y

    return dot > 0  # positive → moving away

def is_vehicle_strictly_stopped(vehicle, intersection_loc, stop_line_radius, speed_thresh=0.2):
    dist = vehicle.get_location().distance(intersection_loc)
    speed = vehicle.get_velocity()
    speed_mag = (speed.x**2 + speed.y**2 + speed.z**2)**0.5
    return dist >= (stop_line_radius-2.0) and (speed_mag < speed_thresh or speed_mag < 0.2)


non_stopped_inside = None

def get_vehicle_class(vehicle):
    type_id = vehicle.type_id.lower()
    if type_id in BIKES:
        return "bike"
    elif type_id in TRUCKS:
        return "truck"
    elif type_id in VANS:
        return "van"
    else:
        return "car"   # default

def get_priority_value(v):
    vclass = get_vehicle_class(v)
    if vclass == "truck":
        return 4
    elif vclass == "van":
        return 3
    elif vclass == "car":
        return 2
    else:
        return 1

vehicle_arrival_to_stop_time = {}  # vid -> arrival-to-stop time


released_vehicle_id = None































# ============================
# MAIN DEMONSTRATION
# ============================
def main():
    # ============================
    # LOAD MODEL
    # ============================
    global non_stopped_inside, released_vehicle_id
    model = CustomGraphModel(None, None, 3, {}, "ppo_graph")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded trained model from episode {MODEL_EPISODE}: {MODEL_PATH}")

    os.makedirs("2_Logs_Town07_3_WAY", exist_ok=True)
    csv_file = f"2_Logs_Town07_3_WAY/ppo_4way_metrics_RGCNN_withSC_{MODEL_EPISODE}.csv"

    # ============================
    # CONNECT CARLA
    # ============================
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(50.0)
    world = client.load_world(TOWN)

    intersection_loc = get_junction_centroid(world.get_map(), JUNCTION_ID)

    set_spectator_top_view(world, intersection_loc)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = False
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    # Visual zones
    #draw_radius_circle(world, intersection_loc, radius=50.0,color=carla.Color(0, 255, 255), z_offset=0.1)
    #draw_radius_circle(world, intersection_loc, radius=20.0,color=carla.Color(255, 0, 255), z_offset=0.2)
    #draw_direction_lines(world, intersection_loc, ["W", "N", "E", "S"], length=30.0, z=2.0)

    # ============================
    # EPISODES
    # ============================
    for episode in range(NUM_DEMO_EPISODES):

        print(f"\n===== Demo Episode {episode+1}/{NUM_DEMO_EPISODES} =====")
        episode_start = time.time()

        # Spawn vehicles
        direction_configs = {
            "E":  (82, TRUCKS, 60),   # East: Truck, spawn id 194, speed 60
            "S":  (49, VANS, 68),      # South: Van, spawn id 44, speed 68
            "W":  (76, BIKES, 55),     # West: Bike, spawn id 30, speed 55
            "N":  (13, CARS, 48)  
        }

        vehicles, _ = spawn_directional_vehicles(
            world, client, TM_PORT, direction_configs
        )

        configure_traffic_manager(tm, vehicles)

        metrics = initialize_episode_metrics()
        collision_flags = {v.id: False for v in vehicles}
        sensors = attach_collision_sensors(world, vehicles, metrics)

        episode_reward = 0.0

        # Warmup
        for _ in range(10):
            world.tick()

        # ============================
        # MAIN CONTROL LOOP
        # ============================
        # Initialize deadlock counter (before loop)
        deadlock_counter = 0

        #waiting_printed_once = True
        released_vehicle_ids = set()
        currently_released_vid = None
        released_vid_entered_zone = False  # ← track if it entered the junction yet


        for step in range(MAX_STEPS):
            
            alive_vehicles = [v for v in vehicles if v.is_alive]
            if not alive_vehicles:
                print("All vehicles finished.")
                break

            

            # =====================================================
            # STEP 1: RL ACTION INFERENCE
            # =====================================================
            actions_dict = {}

            for v in alive_vehicles:
                obs = build_graph(
                    alive_vehicles,
                    intersection_loc,
                    ["W", "N", "E", "S"],
                    ego_vehicle=v
                )

                obs_tensor = obs_to_tensor(obs)

                with torch.no_grad():
                    logits, _ = model({"obs": obs_tensor}, [], None)

                action = torch.argmax(logits, dim=-1).item()
                actions_dict[v.id] = action

            

            
            # --- NEW LOGIC: Force STOP for first 2, MOVE for the rest ---
            stop_vehicles = [v for v in alive_vehicles if actions_dict.get(v.id, 2) == 0]

            if len(stop_vehicles) >= 3:
                # Pick the first 3 vehicles that got STOP
                forced_stop_ids = set(v.id for v in stop_vehicles[:3])
                # The rest are forced MOVE
                forced_move_ids = set(v.id for v in alive_vehicles) - forced_stop_ids
            else:
                forced_stop_ids = set()
                forced_move_ids = set()

            # --- Identify vehicles that did NOT receive STOP ---
            non_stop_vehicles = [v for v in alive_vehicles if actions_dict.get(v.id, 2) != 0]

            # If there are any such vehicles, find the one closest to the intersection
            closest_vid = None
            if non_stop_vehicles and currently_released_vid is None:
                distances = [(v.id, v.get_location().distance(intersection_loc)) for v in non_stop_vehicles]
                # Find the vehicle with the minimum distance
                closest_vid = min(distances, key=lambda x: x[1])[0]

            non_stopped_inside = False  # Reset for this step




            
            # =====================================================
            # STEP 2: PHASE-1 – FIRST STOP CONTROL
            # =====================================================

            for v in alive_vehicles:

                vid = v.id
                action = actions_dict[vid]   # 0=STOP, 1=SLOW, 2=MOVE (assuming your mapping)

                tempD = v.get_location().distance(intersection_loc) 
                if tempD > 50.0:
                    continue
                
                if is_moving_away(v, intersection_loc) and tempD > CONGESTION_RADIUS:
                    tm.vehicle_percentage_speed_difference(v, 0)
                    #v.apply_control(carla.VehicleControl(throttle=0.5, brake=0.0, hand_brake=False))
                    draw_vehicle_alert(world, v, "MOVE")
                    continue

                vid = v.id
                # --- Apply forced STOP/MOVE logic ---
                # --- Apply forced STOP/MOVE logic ---
                if vid in forced_stop_ids:
                    action = 0  # Force STOP
                elif vid in forced_move_ids:
                    action = 2  # Force MOVE
                else:
                    # New logic: For non-STOP vehicles, only the closest proceeds, others STOP
                    if actions_dict[vid] != 0:
                        if vid == closest_vid:
                            action = 2  # Allow to MOVE
                        else:
                            action = 0  # Force STOP
                    else:
                        action = actions_dict[vid]

                # Initialize
                if vid not in vehicle_stop_state:
                    vehicle_stop_state[vid] = False

                loc = v.get_location()
                dist = loc.distance(intersection_loc)

                if action == 0 and (dist > STOP_LINE_RADIUS) and (vehicle_stop_state[vid] == False):
                    if vid in released_vehicle_ids:   # ← ADD THIS GUARD
                        pass                          # Don't re-stop a released vehicle
                    else:
                        if vid not in vehicle_arrival_time:
                            vehicle_arrival_time[vid] = time.time()
                        vehicle_stop_state[vid] = True

                # If vehicle is strictly stopped (your strict stop logic)
                if vehicle_stop_state.get(vid, False) and is_vehicle_strictly_stopped(v, intersection_loc, STOP_LINE_RADIUS):
                    if vid not in vehicle_strict_stop_time:
                        vehicle_strict_stop_time[vid] = time.time()
                        arrival_to_stop = vehicle_strict_stop_time[vid] - vehicle_arrival_time.get(vid, vehicle_strict_stop_time[vid])
                        print(f"Vehicle {vid} arrival-to-stop time: {arrival_to_stop:.2f} seconds")  # Mark as must stop before stop line
                        vehicle_arrival_to_stop_time[vid] = arrival_to_stop



                # =================================================
                # If STOP received once before stop line
                # =================================================
                if vehicle_stop_state[vid]:
                    if dist > STOP_LINE_RADIUS:
                        # Smooth stop
                        max_dist = STOP_LINE_RADIUS + 15.0
                        min_dist = STOP_LINE_RADIUS + 10.0
                        if dist > max_dist:
                            speed_diff = 0
                        elif dist <= min_dist:
                            speed_diff = 100
                        else:
                            speed_diff = int(100 * (max_dist - dist) / (max_dist - min_dist))
                        tm.vehicle_percentage_speed_difference(v, speed_diff)
                        draw_vehicle_alert(world, v, "STOP")
                        # --- Add strict brake when close to stop line ---
                        if dist <= STOP_LINE_RADIUS + 10.0:
                            v.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
                        continue
                    else:
                        # Crossed stop line, release to RL
                        vehicle_stop_state[vid] = False
                        released_vehicle_ids.discard(vid)

                else:
                    if action == 1:
                        draw_vehicle_alert(world, v, "SLOW")
                        tm.vehicle_percentage_speed_difference(v, 50)
                    else:
                        draw_vehicle_alert(world, v, "MOVE")
                        tm.vehicle_percentage_speed_difference(v, 0)


            # =====================================================
            # STEP 2: PHASE-3 – Release Control (One at a time)
            # =====================================================

            strictly_stopped_vehicles = []
            for v in alive_vehicles:
                if vehicle_stop_state.get(v.id, False):
                    if is_vehicle_strictly_stopped(v, intersection_loc, STOP_LINE_RADIUS):
                        strictly_stopped_vehicles.append(v)

            # Check if any non-stopped vehicle is still inside the congestion zone
            non_stopped_inside = False
            for v in alive_vehicles:
                dist = v.get_location().distance(intersection_loc)
                if dist < CONGESTION_RADIUS and not vehicle_stop_state.get(v.id, False):
                    tm.vehicle_percentage_speed_difference(v, 0)
                    draw_vehicle_alert(world, v, "MOVE")
                    non_stopped_inside = True

            # Check if the currently released vehicle has cleared the intersection
            if currently_released_vid is not None:
                released_still_alive = any(v.id == currently_released_vid for v in alive_vehicles)
                if released_still_alive:
                    released_v = next(v for v in alive_vehicles if v.id == currently_released_vid)
                    released_dist = released_v.get_location().distance(intersection_loc)

                    # Phase 1: wait for it to enter the congestion zone
                    if not released_vid_entered_zone and released_dist < CONGESTION_RADIUS:
                        released_vid_entered_zone = True

                    # Phase 2: only clear after it has entered AND then exited
                    if released_vid_entered_zone and released_dist > CONGESTION_RADIUS:
                        print(f"Vehicle {currently_released_vid} has cleared. Ready for next release.")
                        currently_released_vid = None
                        released_vid_entered_zone = False
                else:
                    # Vehicle left the scene entirely
                    currently_released_vid = None
                    released_vid_entered_zone = False

            # Only release a new vehicle if:
            # 1. No vehicle is currently being processed through
            # 2. The intersection congestion zone is clear of non-stopped moving vehicles
            if currently_released_vid is None and not non_stopped_inside and strictly_stopped_vehicles:
                to_release = sorted(strictly_stopped_vehicles, key=get_priority_value, reverse=True)
                cats = [get_vehicle_class(v) for v in to_release]
                to_release = to_release[0]
                if len(cats) != len(set(cats)):
                    print("Duplicate categories detected. Switching to arrival time based release.")
                    to_release = sorted(strictly_stopped_vehicles,
                                        key=lambda v: vehicle_arrival_to_stop_time.get(v.id, float('inf')))[0]

                currently_released_vid = to_release.id
                released_vehicle_ids.add(to_release.id)
                vehicle_stop_state[to_release.id] = False
                tm.vehicle_percentage_speed_difference(to_release, 0)
                draw_vehicle_alert(world, to_release, "MOVE")
                rel_dir = get_vehicle_direction(to_release, intersection_loc)
                print(f"Direction Zone: {rel_dir} → Releasing Vehicle {to_release.id} ({get_vehicle_class(to_release)})") 
            

            # =====================================================
            # STEP 6: SIMULATION STEP
            # =====================================================
            world.tick()

            update_vehicle_tracking(alive_vehicles, intersection_loc, metrics)

            reward = compute_total_reward(
                world,
                alive_vehicles,
                intersection_loc,
                collision_flags
            )

            episode_reward += reward

        vehicle_stop_state.clear()  # Clear state for next episode
        currently_released_vid = None   # ← add this
        released_vid_entered_zone = False   # ← add this
        released_vehicle_ids.clear()    # ← add this too

        # ============================
        # EPISODE END
        # ============================
        duration = time.time() - episode_start
        final_metrics = compute_final_metrics(metrics, duration)
        save_episode_metrics(csv_file, episode + 1, episode_reward, final_metrics)

        print("Reward:", episode_reward)
        print("Metrics:", final_metrics)

        # Cleanup
        for s in sensors:
            if s.is_alive:
                s.stop()
                s.destroy()

        for v in vehicles:
            if v.is_alive:
                v.destroy()
        time.sleep(3.0)

    # Restore world
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)

    print("Demo finished.")
    

if __name__ == "__main__":
    main()