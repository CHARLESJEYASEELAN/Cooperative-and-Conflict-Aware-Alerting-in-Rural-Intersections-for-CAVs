# CARLA vehicle blueprint categories
import carla

BIKES = [
    "vehicle.harley-davidson.low_rider",# Motorcycle
    "vehicle.kawasaki.ninja",           # Motorcycle
    "vehicle.yamaha.yzf"                # Motorcycle
]

TRUCKS = [
    "vehicle.carlamotors.european_hgv", # Heavy goods vehicle (truck)
    "vehicle.tesla.cybertruck",         # Pickup truck
    "vehicle.mitsubishi.fusorosa"       # Truck
]

VANS = [
    "vehicle.mercedes.sprinter",        # Van
    "vehicle.carlamotors.carlacola",    # Van
    "vehicle.micro.microlino"           # Small van
]

CARS = [
    "vehicle.audi.a2",
    "vehicle.audi.etron",
    "vehicle.audi.tt",
    "vehicle.bmw.grandtourer",
    "vehicle.chevrolet.impala",
    "vehicle.citroen.c3",
    "vehicle.dodge.charger_2020",
    "vehicle.dodge.charger_police",
    "vehicle.dodge.charger_police_2020",
    "vehicle.ford.crown",
    "vehicle.ford.mustang",
    "vehicle.ford.ambulance",
    "vehicle.jeep.wrangler_rubicon",
    "vehicle.lincoln.mkz_2017",
    "vehicle.lincoln.mkz_2020",
    "vehicle.lincoln.mkz_2020",
    "vehicle.mercedes.coupe",
    "vehicle.mercedes.coupe_2020",
    "vehicle.mini.cooper_s",
    "vehicle.mini.cooper_s_2021",
    "vehicle.nissan.micra",
    "vehicle.nissan.patrol",
    "vehicle.nissan.patrol_2021",
    "vehicle.seat.leon",
    "vehicle.toyota.prius",
    "vehicle.volkswagen.t2",
    "vehicle.volkswagen.t2_2021"
]


# ------------------------------------------------------------
# RELIABLE JUNCTION CENTROID (Using topology + bounding box)
# ------------------------------------------------------------
def get_junction_centroid(carla_map, junction_id):
    topology = carla_map.get_topology()

    for wp1, wp2 in topology:
        if wp1.is_junction and wp1.get_junction().id == junction_id:
            junction = wp1.get_junction()
            bbox = junction.bounding_box
            loc = bbox.location
            return carla.Location(loc.x, loc.y, loc.z + 1.0)

    print(f"[ERROR] Junction ID {junction_id} not found.")
    return None

# -----------------------------------------------------------
# DRAW ALL JUNCTION IDS (For debugging)
# ------------------------------------------------------------
def draw_intersection_labels(world, carla_map):
    topology = carla_map.get_topology()
    drawn_ids = set()

    for wp1, wp2 in topology:
        if wp1.is_junction:
            junction = wp1.get_junction()
            junction_id = junction.id

            if junction_id not in drawn_ids:
                drawn_ids.add(junction_id)
                loc = junction.bounding_box.location

                world.debug.draw_string(
                    carla.Location(loc.x, loc.y, loc.z + 3.0),
                    f"Junction ID: {junction_id}",
                    life_time=180.0,
                    color=carla.Color(0, 0, 255),
                    persistent_lines=True
                )

    print("Available Junction IDs:", sorted(drawn_ids))

# ------------------------------------------------------------
# DRAW DIRECTION LINES (optional helper)
# ------------------------------------------------------------
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
# MARK SPAWN POINTS (helpful for choosing indices)
# ------------------------------------------------------------
def draw_spawn_point_markers(world, carla_map):
    spawn_points = carla_map.get_spawn_points()
    for idx, sp in enumerate(spawn_points):
        loc = sp.location
        world.debug.draw_box(
            box=carla.BoundingBox(loc, carla.Vector3D(0.5, 0.5, 1.0)),
            rotation=sp.rotation,
            thickness=0.1,
            color=carla.Color(0, 255, 0),
            life_time=180.0,
            persistent_lines=True
        )
        world.debug.draw_string(
            carla.Location(loc.x, loc.y, loc.z + 2.0),
            str(idx),
            draw_shadow=False,
            color=carla.Color(0, 255, 0),
            life_time=180.0,
            persistent_lines=True
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
            life_time=0.2,
            persistent_lines=False
        )

# ------------------------------------------------------------
# TOP VIEW CAMERA
# ------------------------------------------------------------
def set_spectator_top_view(world, location, height=90.0):
    spectator = world.get_spectator()
    transform = carla.Transform(
        carla.Location(location.x, location.y, location.z + height),
        carla.Rotation(pitch=-90.0)
    )
    spectator.set_transform(transform)