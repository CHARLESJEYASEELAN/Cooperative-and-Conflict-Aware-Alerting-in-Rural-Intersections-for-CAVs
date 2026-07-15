import carla
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import math
from torch.distributions import Categorical
import csv
import tqdm
from tqdm import trange

from Utility_Carla import BIKES, TRUCKS, VANS, CARS

# ==== YOUR EXISTING IMPORTS ====
from Utility_RL import (
    CustomGraphModel,
    build_graph,
    spawn_directional_vehicles,
    get_junction_centroid,
    compute_total_reward,
    initialize_episode_metrics,
    update_vehicle_tracking,
    attach_collision_sensors,
    compute_final_metrics
)

SEED = 999  # change to 123, 999



# Note: Assuming Utility_RL.py has all the necessary functions as in your previous scripts.
# If collision_flags need to be updated in sensors, ensure the callback sets collision_flags[v.id] = True
# and increments metrics['collisions'].

def configure_traffic_manager(tm, vehicles):
    for vehicle in vehicles:
        tm.ignore_lights_percentage(vehicle, 100)
        tm.ignore_signs_percentage(vehicle, 100)
        tm.ignore_vehicles_percentage(vehicle, 85)
        tm.auto_lane_change(vehicle, True)
        tm.distance_to_leading_vehicle(vehicle, 4.0)
        tm.vehicle_percentage_speed_difference(vehicle, -85)

def save_episode_metrics(csv_file, episode, episode_reward, final_metrics):
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode='a', newline='') as f:
        writer = csv.writer(f)

        # Header
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
TOWN = "Town03"
TM_PORT = 8000
JUNCTION_ID = 238  # 4-way

CHECKPOINT_SWITCH = True  # Whether to load from checkpoint or start fresh

START_EPISODE = 100
MAX_EPISODES = 300  # Test with 5, can increase to 100-200
MAX_STEPS = 400

GAMMA = 0.99
LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 5e-5
PPO_EPOCHS = 2
BATCH_SIZE = 256  # Not used in this mini-batch free version, but placeholder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(DEVICE)

# ============================
# PPO MEMORY
# ============================
class Memory:
    def __init__(self):
        self.obs = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def clear(self):
        self.__init__()

# ============================
# PREPROCESS OBS
# ============================
def obs_to_tensor(obs_dict):
    return {
        k: torch.tensor(v, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        for k, v in obs_dict.items()
    }

# ============================
# ADVANTAGE (GAE)
# ============================
def compute_gae(memory, next_value):
    rewards = memory.rewards
    values = memory.values + [next_value]
    dones = memory.dones

    gae = 0
    returns = []
    advantages = []

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step+1] * (1-dones[step]) - values[step]
        gae = delta + GAMMA * LAMBDA * (1-dones[step]) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[step])

    return torch.tensor(returns, dtype=torch.float32).to(DEVICE), torch.tensor(advantages, dtype=torch.float32).to(DEVICE)

# ============================
# PPO UPDATE
# ============================
def ppo_update(model, optimizer, memory):
    if not memory.actions:  # Safety check: skip if no data
        print("No transitions in memory — skipping PPO update")
        return

    # Compute returns and advantages once (full memory)
    returns, advantages = compute_gae(memory, next_value=0)

    actions = torch.tensor(memory.actions, dtype=torch.long).to(DEVICE)
    old_logprobs = torch.tensor(memory.logprobs, dtype=torch.float32).to(DEVICE)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    num_transitions = len(memory.actions)
    indices = np.arange(num_transitions)

    # Prepare batched obs for all transitions (each obs is a dict of tensors)
    obs_keys = list(memory.obs[0].keys())
    batched_obs_all = {k: torch.cat([obs[k] for obs in memory.obs], dim=0) for k in obs_keys}

    print("Starting PPO update")
    for epoch in tqdm.tqdm(range(PPO_EPOCHS), desc="PPO Epochs"):
        np.random.shuffle(indices)  # Shuffle for better generalization

        for start_idx in range(0, num_transitions, BATCH_SIZE):
            end_idx = min(start_idx + BATCH_SIZE, num_transitions)
            batch_indices = indices[start_idx:end_idx]

            # Slice batched obs for this mini-batch
            batch_obs = {k: v[batch_indices] for k, v in batched_obs_all.items()}
            batch_actions = actions[batch_indices]
            batch_old_logprobs = old_logprobs[batch_indices]
            batch_advantages = advantages[batch_indices]
            batch_returns = returns[batch_indices]

            # Forward pass for the batch
            logits, _ = model({"obs": batch_obs}, [], None)
            values = model.value_function()

            dist = Categorical(logits=logits)
            logprobs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()

            ratios = torch.exp(logprobs - batch_old_logprobs)

            surr1 = ratios * batch_advantages
            surr2 = torch.clamp(ratios, 1 - CLIP_EPS, 1 + CLIP_EPS) * batch_advantages

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (batch_returns - values).pow(2).mean()

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

    print(f"PPO update completed for {num_transitions} transitions "
          f"({PPO_EPOCHS} epochs, batch size {BATCH_SIZE})")
    

def draw_vehicle_alert(world, vehicle, alert_type, color=None, life_time=0.02):
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


def cleanup_far_vehicles(world, intersection_loc, despawn_radius, vehicles, sensors):
    """
    Destroys vehicles and their sensors that are farther than despawn_radius from the intersection.
    Modifies vehicles and sensors lists in-place.
    """
    to_remove = []
    for i, v in enumerate(vehicles):
        if not v.is_alive:
            to_remove.append(i)
            continue
        loc = v.get_location()
        if loc.distance(intersection_loc) > despawn_radius:
            try:
                if v.is_alive:
                    v.destroy()
            except:
                pass
            try:
                if sensors[i].is_alive:
                    sensors[i].stop()
                    sensors[i].destroy()
            except:
                pass
            to_remove.append(i)
    # Remove from lists (in reverse order to avoid index shift)
    for idx in reversed(to_remove):
        del vehicles[idx]
        del sensors[idx]
def set_spectator_top_view(world, location, height=90.0):
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla.Location(location.x, location.y, location.z + height),
        carla.Rotation(pitch=-90.0)
    )
    spectator.set_transform(transform)
# ============================
# MAIN TRAIN LOOP
# ============================
def main():

    os.makedirs("1_Logs_RGCNN_4_WAY", exist_ok=True)
    csv_file = "1_Logs_RGCNN_4_WAY/ppo_4way_metrics_RGCNN.csv"

    # ==== CARLA ====
    client = carla.Client(CARLA_HOST, CARLA_PORT)
    client.set_timeout(50.0)
    world = client.load_world(TOWN)

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.no_rendering_mode = True  # Disable rendering for faster training
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    tm = client.get_trafficmanager(TM_PORT)
    tm.set_synchronous_mode(True)

    intersection_loc = get_junction_centroid(world.get_map(), JUNCTION_ID)
    set_spectator_top_view(world, intersection_loc)
    # ==== MODEL ====
    model = CustomGraphModel(None, None, 3, {}, "ppo_graph").to(DEVICE)  # Adjust init if needed

    if CHECKPOINT_SWITCH:
        print("Switched to checkpoint loading mode.")
        checkpoint_path = f"1_RL_Models_4WAY_RGCNN/ppo_ep{START_EPISODE}.pth"
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE), strict=False)
        print(f"Loaded checkpoint from episode {START_EPISODE}")
    else:
        print("Starting training from scratch with pretrained critic.")
        pretrained_path = "./Models/rgcn_critic_4WAY.pth"
        model.load_state_dict(torch.load(pretrained_path, map_location=DEVICE), strict=False)
        print("Loaded pretrained critic for 4-way")

    optimizer = optim.Adam(model.parameters(), lr=LR)

    # ==== TRAIN ====
    for episode in range(START_EPISODE, START_EPISODE+MAX_EPISODES):

        memory = Memory()
        episode_start_time = time.time()

        # Spawn vehicles for 4-way
        direction_configs = {
            "E":  (194, TRUCKS, 60),   # East: Truck, spawn id 194, speed 60
            "S":  (44, VANS, 68),      # South: Van, spawn id 44, speed 68
            "W":  (30, BIKES, 55),     # West: Bike, spawn id 30, speed 55
            "N":  (165, CARS, 48)      # North: Car, spawn id 52, speed 48
        }

        vehicles, _ = spawn_directional_vehicles(world, client, TM_PORT, direction_configs)
        configure_traffic_manager(tm, vehicles)

        metrics = initialize_episode_metrics()
        collision_flags = {v.id: False for v in vehicles}
        sensors = attach_collision_sensors(world, vehicles, metrics)  # Assume callback updates metrics['collisions'] and optionally collision_flags

        episode_reward = 0
        print(f"Starting Episode {episode+1} with {len(vehicles)} vehicles.")

        spawnTime = time.time() - episode_start_time
        print(f"Spawned vehicles and sensors in {spawnTime:.2f} seconds.")

        for step in trange(MAX_STEPS, desc=f"Episode {episode+1} Steps", leave=False):

            actions_dict = {}

            for v in vehicles:
                if not v.is_alive:
                    continue

                obs = build_graph(vehicles, intersection_loc, ["E", "S", "W", "N"], ego_vehicle=v)
                obs_tensor = obs_to_tensor(obs)

                logits, _ = model({"obs": obs_tensor}, [], None)  # Adjust forward if needed
                value = model.value_function().item()

                dist = Categorical(logits=logits)
                action = dist.sample()
                logprob = dist.log_prob(action)

                memory.obs.append(obs_tensor)
                memory.actions.append(action.item())
                memory.logprobs.append(logprob.item())
                memory.values.append(value)
                memory.dones.append(0)

                actions_dict[v.id] = action.item()

            # Apply actions
            for v in vehicles:
                if v.id not in actions_dict or not v.is_alive:
                    continue
                a = actions_dict[v.id]
                if a == 0:  # STOP
                    v.apply_control(carla.VehicleControl(brake=1.0))
                    #draw_vehicle_alert(world, v, "STOP")
                elif a == 1:  # SLOW
                    tm.vehicle_percentage_speed_difference(v, 50)
                    #draw_vehicle_alert(world, v, "SLOW")
                else:  # MOVE
                    tm.vehicle_percentage_speed_difference(v, 0)
                    #draw_vehicle_alert(world, v, "MOVE")

            world.tick()

            # Update tracking (for metrics)
            alive_vehicles = [v for v in vehicles if v.is_alive]
            update_vehicle_tracking(alive_vehicles, intersection_loc, metrics)  # Assume radius=15.0 or whatever

            # Compute reward
            reward = compute_total_reward(world, alive_vehicles, intersection_loc, collision_flags)

            num_agents = len(actions_dict)
            for _ in range(num_agents):
                memory.rewards.append(reward)
            episode_reward += reward

            # Optional: Cleanup far vehicles to prevent clutter
            #cleanup_far_vehicles(world, intersection_loc, 80.0, vehicles, sensors)

        stepTime = time.time() - episode_start_time
        print(f"Completed {MAX_STEPS} steps in {stepTime:.2f} seconds.")

        # ==== PPO UPDATE ====
        ppo_update(model, optimizer, memory)
        torch.cuda.empty_cache()  # Clear GPU cache after update to free memory
        memory.clear()  # Clear memory after PPO update

        # ==== CLEANUP ====
        for s in sensors:
            try:
                if s.is_alive:
                    s.stop()
                    s.destroy()
            except Exception as e:
                print(f"Error destroying sensor: {e}")

        # Destroy vehicles
        for v in vehicles:
            try:
                if v.is_alive:
                    v.destroy()
            except Exception as e:
                print(f"Error destroying vehicle: {e}")
        torch.cuda.empty_cache()  # Clear GPU cache after cleanup

        episode_duration = time.time() - episode_start_time

        final_metrics = compute_final_metrics(metrics, episode_duration)
        #os.makedirs(os.path.dirname(csv_file), exist_ok=True)
        save_episode_metrics(
            csv_file,
            episode + 1,
            episode_reward,
            final_metrics
        )

        memoryTime = time.time() - episode_start_time
        print(f"Saved metrics/memory in {memoryTime:.2f} seconds.")

        print(f"Episode {episode+1} | Reward: {episode_reward:.2f}")
        print("Metrics:", final_metrics)
        print(f"Episode duration: {episode_duration:.2f} seconds")

        if (episode+1) % 1 == 0:
            os.makedirs("1_RL_Models_4WAY_RGCNN", exist_ok=True)
            torch.save(model.state_dict(), f"1_RL_Models_4WAY_RGCNN/ppo_ep{episode+1}.pth")
        
    # Final cleanup
    settings = world.get_settings()
    settings.synchronous_mode = False
    settings.fixed_delta_seconds = None
    world.apply_settings(settings)
    print("Training completed.")

if __name__ == "__main__":
    main()