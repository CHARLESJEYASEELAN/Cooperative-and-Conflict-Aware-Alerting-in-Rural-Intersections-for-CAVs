import carla
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import RGCNConv, global_mean_pool
import random
import time
import os
import csv
import traceback
import math

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.modelv2 import restore_original_dimensions
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from ray import tune
import ray

from Utility_Carla import BIKES, TRUCKS, VANS, CARS

# ==== CONFIG ====
CARLA_HOST = "localhost"
CARLA_PORT = 2000
TOWN = "Town03"
TM_PORT = 8000
JUNCTION_ID = 1352
MAX_EPISODES = 5
MAX_EPISODE_STEPS = 400
INTERSECTION_RADIUS = 60.0
DESPAWN_RADIUS = 80.0
DEADLOCK_SPEED = 0.5
DEADLOCK_SECONDS = 5.0
CONGESTION_TIMEOUT = 20.0

FEAT_DIM = 12 #12 for 3-way and 13 for 4-way
NUM_RELATIONS = 3
HIDDEN_DIM = 64
MAX_NODES = 9
MAX_EDGES = 72
DIRECTIONS = ["E", "N", "S"]

SEED = 42  # change to 123, 999



obs_space = spaces.Dict({
    "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_NODES, FEAT_DIM)),
    "edge_index": spaces.Box(low=-np.inf, high=np.inf, shape=(2, MAX_EDGES)),
    "edge_types": spaces.Box(low=0, high=NUM_RELATIONS-1, shape=(MAX_EDGES,)),
    "num_nodes": spaces.Box(low=0, high=MAX_NODES, shape=()),
    "num_edges": spaces.Box(low=0, high=MAX_EDGES, shape=()),
    "ego_index": spaces.Box(low=0, high=MAX_NODES-1, shape=())
})
act_space = spaces.Discrete(3)  # Stop, SlowDown, Move

# ==== MODEL ====
class CustomGraphModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.conv1 = RGCNConv(FEAT_DIM, HIDDEN_DIM, NUM_RELATIONS)
        self.conv2 = RGCNConv(HIDDEN_DIM, HIDDEN_DIM, NUM_RELATIONS)
        self.actor_lin = nn.Linear(HIDDEN_DIM, num_outputs)
        self.value_lin = nn.Linear(HIDDEN_DIM, 1)
        self.cur_value = None

    def forward(self, input_dict, state, seq_lens):
        obs = restore_original_dimensions(input_dict["obs"], self.obs_space, "torch")
        batch_size = obs["node_features"].shape[0]
        datas, valid_indices = [], []
        for i in range(batch_size):
            num_n = int(obs["num_nodes"][i].item())
            if num_n == 0: continue
            x = obs["node_features"][i, :num_n, :]
            ego_idx = int(obs["ego_index"][i].item())
            num_e = int(obs["num_edges"][i].item())
            edge_idx = obs["edge_index"][i, :, :num_e].long()
            edge_t = obs["edge_types"][i, :num_e].long()
            datas.append(Data(x=x, edge_index=edge_idx, edge_type=edge_t, ego_index=ego_idx))
            valid_indices.append(i)
        if len(datas) == 0:
            logits = torch.zeros((batch_size, self.num_outputs), device=input_dict["obs"]["node_features"].device)
            self.cur_value = torch.zeros(batch_size, device=logits.device)
            return logits, []
        batch = Batch.from_data_list(datas)
        x = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_type))
        x = F.relu(self.conv2(x, batch.edge_index, batch.edge_type))
        pooled = global_mean_pool(x, batch.batch)
        values = self.value_lin(pooled).squeeze(-1)
        logits_list = []
        for g in range(batch.num_graphs):
            ego_local = datas[g].ego_index
            ego_global = batch.ptr[g] + ego_local
            ego_embed = x[ego_global]
            logits_list.append(self.actor_lin(ego_embed))
        logits_valid = torch.stack(logits_list)
        logits = torch.zeros((batch_size, self.num_outputs), device=logits_valid.device)
        value_full = torch.zeros(batch_size, device=values.device)
        for idx, orig_i in enumerate(valid_indices):
            logits[orig_i] = logits_valid[idx]
            value_full[orig_i] = values[idx]
        self.cur_value = value_full
        return logits, []

    def value_function(self):
        return self.cur_value

def load_pretrained(model, pretrained_path):
    if os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path)
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")
    else:
        print(f"Pretrained file not found: {pretrained_path}")

# ==== UTILS ====
def get_junction_centroid(carla_map, junction_id):
    for wp1, wp2 in carla_map.get_topology():
        if wp1.is_junction and wp1.get_junction().id == junction_id:
            bbox = wp1.get_junction().bounding_box
            loc = bbox.location
            return carla.Location(loc.x, loc.y, loc.z + 1.0)
    return None

def set_spectator_top_view(world, location, height=90.0):
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla.Location(location.x, location.y, location.z + height),
        carla.Rotation(pitch=-90.0)
    )
    spectator.set_transform(transform)

def spawn_directional_vehicles(world, client, tm_port, direction_config):
    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    tm = client.get_trafficmanager(tm_port)
    tm.set_synchronous_mode(False)
    vehicles = []
    for direction, (spawn_id, vehicle_list, speed_mph) in direction_config.items():
        if spawn_id >= len(spawn_points): continue
        allowed_blueprints = [bp for bp in blueprint_library.filter('vehicle.*') if bp.id in vehicle_list]
        if not allowed_blueprints: continue
        transform = spawn_points[spawn_id]
        if world.get_actors().filter('vehicle.*') and any(
            v.get_transform().location.distance(transform.location) < 4.0
            for v in world.get_actors().filter('vehicle.*')
        ): continue
        blueprint = random.choice(allowed_blueprints)
        vehicle = world.try_spawn_actor(blueprint, transform)
        if vehicle is None: continue
        vehicle.set_autopilot(True, tm_port)
        speed_kph = speed_mph * 1.60934
        tm.vehicle_percentage_speed_difference(vehicle, 100 - (speed_kph / 90 * 100))
        vehicles.append(vehicle)
    return vehicles, tm

def destroy_vehicles(vehicles):
    for v in vehicles:
        try:
            if v is not None and v.is_alive:
                v.destroy()
        except RuntimeError: pass
    vehicles.clear()

def build_graph(vehicles, intersection_loc, direction_keys, ego_vehicle=None):
    node_features, ego_index = [], None
    for idx, veh in enumerate(vehicles):
        loc = veh.get_location()
        vel = veh.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        dist = loc.distance(intersection_loc)
        yaw = veh.get_transform().rotation.yaw
        dir_onehot = np.zeros(len(direction_keys))
        dir_onehot[idx % len(direction_keys)] = 1.0
        blueprint = veh.type_id.lower()
        if blueprint in TRUCKS: type_idx = 1
        elif blueprint in VANS: type_idx = 2
        elif blueprint in BIKES: type_idx = 3
        else: type_idx = 0
        type_onehot = np.zeros(4)
        type_onehot[type_idx] = 1.0
        features = np.array([loc.x, loc.y, speed, dist, yaw])
        full_feature = np.concatenate([features, dir_onehot, type_onehot])
        node_features.append(full_feature)
        if ego_vehicle is not None and veh == ego_vehicle:
            ego_index = idx
    n = len(vehicles)
    edge_indices, edge_types = [], []
    epsilon = 0.1
    for i in range(n):
        for j in range(n):
            if i == j: continue
            vi, vj = vehicles[i], vehicles[j]
            loc_i, loc_j = vi.get_location(), vj.get_location()
            speed_i = np.sqrt(vi.get_velocity().x**2 + vi.get_velocity().y**2 + vi.get_velocity().z**2)
            speed_j = np.sqrt(vj.get_velocity().x**2 + vj.get_velocity().y**2 + vj.get_velocity().z**2)
            distance = loc_i.distance(loc_j)
            if distance < 40:
                edge_indices.append([i, j])
                edge_types.append(0)
            delta_theta = abs(vi.get_transform().rotation.yaw - vj.get_transform().rotation.yaw)
            delta_theta = np.radians(delta_theta)
            H_ij = abs(np.sin(delta_theta))
            if H_ij > 0.5:
                edge_indices.append([i, j])
                edge_types.append(1)
            v_rel = abs(speed_i - speed_j)
            TTC = distance / (v_rel + epsilon)
            if TTC < 5.0:
                edge_indices.append([i, j])
                edge_types.append(2)
    node_features = np.array(node_features)
    if len(node_features) > MAX_NODES:
        node_features = node_features[:MAX_NODES]
    padded_nodes = np.zeros((MAX_NODES, FEAT_DIM))
    padded_nodes[:len(node_features)] = node_features
    num_nodes = len(node_features)
    edge_indices = np.array(edge_indices).T if edge_indices else np.zeros((2, 0))
    if edge_indices.shape[1] > MAX_EDGES:
        edge_indices = edge_indices[:, :MAX_EDGES]
    padded_edges = np.zeros((2, MAX_EDGES))
    padded_edges[:, :edge_indices.shape[1]] = edge_indices
    num_edges = edge_indices.shape[1]
    edge_types = np.array(edge_types)
    if len(edge_types) > MAX_EDGES:
        edge_types = edge_types[:MAX_EDGES]
    padded_types = np.zeros(MAX_EDGES)
    padded_types[:len(edge_types)] = edge_types
    return {
        "node_features": padded_nodes.astype(np.float32),
        "edge_index": padded_edges.astype(np.float32),
        "edge_types": padded_types.astype(np.float32),
        "num_nodes": np.array(num_nodes, dtype=np.float32),
        "num_edges": np.array(num_edges, dtype=np.float32),
        "ego_index": np.array(ego_index if ego_index is not None else 0, dtype=np.float32)
    }

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

def save_episode_to_csv(filename, episode_index, metrics, episode_reward):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow([
                "episode", "throughput", "avg_delay", "avg_queue",
                "max_queue", "collisions", "avg_speed", "episode_reward"
            ])
        writer.writerow([
            episode_index, metrics.get("throughput", 0), metrics.get("avg_delay", 0), metrics.get("avg_queue", 0),
            metrics.get("max_queue", 0), metrics.get("collisions", 0), metrics.get("avg_speed", 0), episode_reward
        ])



def initialize_episode_metrics():
    return {
        "entry_times": {},
        "exit_times": {},
        "vehicles_passed": set(),
        "queue_lengths": [],
        "max_queue": 0,
        "total_speed": 0.0,
        "speed_samples": 0,
        "collisions": 0,
        "collision_pairs": set()
    }

def update_vehicle_tracking(vehicles, intersection_loc, metrics, congestion_radius=15.0):
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

def compute_final_metrics(metrics, episode_duration):
    free_flow_speed = 13.8  # m/s
    path_length = 120.0     # meters
    free_flow_time = path_length / free_flow_speed
    delays = []
    for vid in metrics["vehicles_passed"]:
        entry = metrics["entry_times"].get(vid, None)
        exit_t = metrics["exit_times"].get(vid, None)
        if entry and exit_t:
            travel_time = exit_t - entry
            delay = max(0, travel_time - free_flow_time)
            delays.append(delay)
    avg_delay = np.mean(delays) if delays else 0
    throughput = len(metrics["vehicles_passed"]) / episode_duration if episode_duration > 0 else 0
    avg_queue = np.mean(metrics["queue_lengths"]) if metrics["queue_lengths"] else 0
    max_queue = metrics["max_queue"]
    avg_speed = metrics["total_speed"] / metrics["speed_samples"] if metrics["speed_samples"] > 0 else 0
    return {
        "throughput": throughput,
        "avg_delay": avg_delay,
        "avg_queue": avg_queue,
        "max_queue": max_queue,
        "collisions": metrics["collisions"],
        "avg_speed": avg_speed
    }

def attach_collision_sensors(world, vehicles, metrics):
    blueprint_library = world.get_blueprint_library()
    collision_bp = blueprint_library.find('sensor.other.collision')
    sensors = []
    for vehicle in vehicles:
        sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=vehicle)
        def callback(event, metrics=metrics):
            try:
                other_actor = event.other_actor
                if other_actor is not None and "vehicle" in other_actor.type_id:
                    pair = tuple(sorted([event.actor.id, other_actor.id]))
                    if pair not in metrics["collision_pairs"]:
                        metrics["collision_pairs"].add(pair)
                        metrics["collisions"] += 1
            except RuntimeError:
                pass
        sensor.listen(callback)
        sensors.append(sensor)
    return sensors


def configure_traffic_manager(tm, vehicles):
    for vehicle in vehicles:
        tm.ignore_lights_percentage(vehicle, 100)
        tm.ignore_signs_percentage(vehicle, 100)
        tm.ignore_vehicles_percentage(vehicle, 75)
        tm.auto_lane_change(vehicle, True)
        tm.distance_to_leading_vehicle(vehicle, 5.0)
        tm.vehicle_percentage_speed_difference(vehicle, -75)


def draw_vehicle_alert(world, vehicle, alert_type, color=None, life_time=0.5):
    """
    Draws a text alert above the vehicle in CARLA.
    Args:
        world: carla.World
        vehicle: carla.Vehicle
        alert_type: str ("STOP", "SLOW", "MOVE", etc.)
        color: carla.Color or None (default: auto by alert_type)
        life_time: float (seconds)
    """
    if color is None:
        if alert_type.upper() == "STOP":
            color = carla.Color(255, 0, 0)      # Red
        elif alert_type.upper() == "SLOW":
            color = carla.Color(255, 255, 0)    # Yellow
        elif alert_type.upper() == "MOVE":
            color = carla.Color(0, 255, 0)      # Green
        else:
            color = carla.Color(0, 191, 255)    # Default: DeepSkyBlue

    loc = vehicle.get_location()
    text_loc = carla.Location(loc.x, loc.y, loc.z + 2.5)
    world.debug.draw_string(
        text_loc,
        f"{alert_type}",
        draw_shadow=True,
        color=color,
        life_time=life_time,
        persistent_lines=False
    )


class CarlaRLWrapper(MultiAgentEnv):
    def __init__(self, config):
        # Create CARLA connection inside the worker
        self.client = carla.Client(CARLA_HOST, CARLA_PORT)  # <-- store client
        self.client.set_timeout(50.0)

        self.world = self.client.get_world()
        self.tm = self.client.get_trafficmanager(TM_PORT)

        carla_map = self.world.get_map()
        self.intersection_loc = get_junction_centroid(carla_map, JUNCTION_ID)

        self.vehicles = []
        self.metrics = None
        self.collision_sensors = []

    def reset(self, *, seed=None, options=None):
        self._cleanup()
        self.step_count = 0
        self.start_time = time.time()
        direction_configs = {
            "E": (253, VANS, 45),
            "N": (43, CARS, 62),
            "S": (223, CARS, 55)
        }

        self.vehicles, _ = spawn_directional_vehicles(
            self.world, self.client, TM_PORT, direction_configs
        )

        configure_traffic_manager(self.tm, self.vehicles)
        self.metrics = initialize_episode_metrics()
        self.collision_sensors = attach_collision_sensors(
            self.world, self.vehicles, self.metrics
        )

        obs = {}
        for v in self.vehicles:
            obs[f"veh_{v.id}"] = build_graph(
                self.vehicles,
                self.intersection_loc,
                DIRECTIONS,
                ego_vehicle=v
            )

        return obs, {}

    def step(self, action_dict):
        # Apply actions
        for agent_id, action in action_dict.items():
            veh_id = int(agent_id.split("_")[1])
            vehicle = next((v for v in self.vehicles if v.id == veh_id), None)

            if vehicle and vehicle.is_alive:
                if action == 0:
                    vehicle.apply_control(carla.VehicleControl(brake=1.0))
                    draw_vehicle_alert(self.world, vehicle, "STOP")
                elif action == 1:
                    self.tm.vehicle_percentage_speed_difference(vehicle, 50)
                    draw_vehicle_alert(self.world, vehicle, "SLOW")
                elif action == 2:
                    self.tm.vehicle_percentage_speed_difference(vehicle, 0)
                    draw_vehicle_alert(self.world, vehicle, "MOVE")

        self.world.tick()

        update_vehicle_tracking(self.vehicles, self.intersection_loc, self.metrics)

        reward = compute_total_reward(
            self.world, self.vehicles, self.intersection_loc, {}
        )

        obs, rewards, dones = {}, {}, {}

        for v in self.vehicles:
            if v.is_alive:
                aid = f"veh_{v.id}"
                obs[aid] = build_graph(
                    self.vehicles,
                    self.intersection_loc,
                    DIRECTIONS,
                    ego_vehicle=v
                )
                rewards[aid] = reward
                dones[aid] = False
        self.step_count += 1
        done = self.step_count > MAX_EPISODE_STEPS
        dones["__all__"] = done

        truncateds = {aid: False for aid in obs}
        truncateds["__all__"] = False

        infos = {aid: {} for aid in obs}
        if done:
            self._cleanup()
            episode_duration = time.time() - self.start_time
            final_metrics = compute_final_metrics(self.metrics, episode_duration)
            infos["__common__"] = final_metrics

        return obs, rewards, dones, infos, truncateds
    
    def _cleanup(self):
        # Destroy sensors
        for s in self.collision_sensors:
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except:
                pass
        self.collision_sensors = []

        # Destroy vehicles
        for v in self.vehicles:
            try:
                if v.is_alive:
                    v.destroy()
            except:
                pass
        self.vehicles = []

# ==== MAIN LOOP ====
def main():
    ray.init(ignore_reinit_error=True, include_dashboard=False)
    os.makedirs("Logs", exist_ok=True)
    os.makedirs("RL_Models_3WAY", exist_ok=True)
    pretrained_path = "./Models/rgcn_critic_3WAY.pth"
    csv_filename = "./Logs/rgcnn_ppo_metrics_3_WAY.csv"

    # 1. CARLA setup
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(50.0)
    world = client.load_world(TOWN)
    carla_map = world.get_map()
    intersection_loc = get_junction_centroid(carla_map, JUNCTION_ID)
    set_spectator_top_view(world, intersection_loc)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)


    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)


    tune.register_env("carla_multi_agent", lambda config: CarlaRLWrapper(config))
    config = PPOConfig() \
        .framework("torch") \
        .multi_agent(
            policies={"shared_policy": PolicySpec(None, obs_space, act_space)},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy"
        ) \
        .training(
            model={"custom_model": CustomGraphModel},
            train_batch_size=2000,
            minibatch_size=64,
            lr=5e-5,
            vf_loss_coeff=0.5,
            entropy_coeff=0.005,
            kl_coeff=0.5,
            lambda_=0.95
        ) \
        .resources(num_gpus=1 if torch.cuda.is_available() else 0).api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        ).env_runners(
            num_env_runners=0
        ).environment(env="carla_multi_agent")
    
    trainer = config.build_algo()
    policy = trainer.get_policy("shared_policy")
    model = policy.model


    state_dict = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded pretrained model from {pretrained_path}")

    # 3. Episode loop
    for episode in range(MAX_EPISODES):
        print(f"\n========== EPISODE {episode + 1} ==========")
        result = trainer.train()
        print(f"Episode {episode+1}")
        print("Reward:", result["episode_reward_mean"])
        checkpoint = trainer.save("RL_Models_3WAY")
        print("Saved:", checkpoint)

        if "custom_metrics" in result:
            metrics = result["custom_metrics"]
        else:
            metrics = {}
        save_episode_to_csv(csv_filename, episode + 1, metrics, result["episode_reward_mean"])
            
        
    trainer.stop()
    print("Fine-tuning completed.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("An error occurred:")
        traceback.print_exc()