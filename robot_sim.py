import pybullet as p
import pybullet_data
import time
import math
import heapq
import json
import os
import random
import sys

# Optional reproducibility: run with SIM_SEED=<int> to make the
# randomised traffic (and the Q-agent's ε-greedy exploration)
# repeatable. Unset, every run gets fresh traffic timing.
_seed = os.environ.get("SIM_SEED")
if _seed is not None:
    random.seed(int(_seed))
    print(f"Random seed fixed: SIM_SEED={_seed}")

# Evaluation-harness hooks (see evaluate.py):
#   SIM_HEADLESS=1     -> run without the GUI (p.DIRECT) at full speed,
#                         no real-time pacing
#   SIM_TIME_LIMIT=<s> -> abort the episode as a failure after this much
#                         SIMULATED time (0 or unset = no limit)
SIM_HEADLESS = os.environ.get("SIM_HEADLESS") == "1"
SIM_TIME_LIMIT = float(os.environ.get("SIM_TIME_LIMIT", "0"))

#   SIM_CAPTURE=<path.gif> -> render the run offscreen with PyBullet's
#                             software renderer and write an animated
#                             GIF (combine with SIM_HEADLESS=1; the
#                             route/trail are drawn as real geometry
#                             since debug lines don't render offscreen)
SIM_CAPTURE = os.environ.get("SIM_CAPTURE", "")
CAPTURE_FPS = 8          # captured frames per simulated second
CAPTURE_SPEEDUP = 2.0    # GIF playback speed vs simulated time
CAPTURE_SIZE = (640, 400)
CAPTURE_SCALE = 0.82     # downscale factor applied when writing the GIF
CAPTURE_COLORS = 128     # GIF palette size; below ~96 the traffic
                         # colours visibly shift (blue reads as teal)

#   SIM_MAP=0          -> suppress the 2D navigation map window that
#                         normally opens alongside the 3D GUI view
SIM_MAP = (not SIM_HEADLESS) and os.environ.get("SIM_MAP", "1") == "1"
MAP_UPDATE_EVERY = 24    # sim frames between map messages (~20 Hz)

# ---------------------------------------------------------
# 1. SET UP THE WORLD
# ---------------------------------------------------------
p.connect(p.DIRECT if SIM_HEADLESS else p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # hide the explorer/sliders panel
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
# Match the physics timestep to the main loop's own bookkeeping: the
# loop advances sim_time (and the kinematic cars) by 1/480 per step,
# but PyBullet's default step is 1/240 -- which made the cars move at
# half their configured speed relative to the robot, and every "per
# second" quantity (waiting penalty, collision cooldown) run at half
# rate. Stepping at 480Hz makes one loop iteration == 1/480s
# everywhere, and the time.sleep(1/480) pacing then runs in real time.
p.setTimeStep(1. / 480.)
p.loadURDF("plane.urdf")

# ---------------------------------------------------------
# AIRPORT ROAD NETWORK: straight segments meeting at 90-degree
# junctions, plus one diagonal connector, like a real terminal
# corridor layout instead of one single straight lane
# ---------------------------------------------------------
def draw_road_segment(x1, y1, x2, y2, width=4):
    """Draws a gray road rectangle between two points, with a yellow
    dashed center line running along its length."""
    length = math.hypot(x2 - x1, y2 - y1)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    angle = math.atan2(y2 - y1, x2 - x1)
    orientation = p.getQuaternionFromEuler([0, 0, angle])

    road_visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[length / 2, width / 2, 0.01],
        rgbaColor=[0.55, 0.55, 0.58, 1]
    )
    p.createMultiBody(0, -1, road_visual, basePosition=[mid_x, mid_y, 0.005],
                       baseOrientation=orientation)

    # Dashed center line along this segment
    dash_length = 0.4
    dash_gap = 0.4
    dist = dash_length / 2
    while dist < length - dash_length / 2:
        t = dist / length
        dash_x = x1 + (x2 - x1) * t
        dash_y = y1 + (y2 - y1) * t
        dash_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[dash_length / 2, 0.05, 0.012],
            rgbaColor=[0.95, 0.8, 0.1, 1]
        )
        p.createMultiBody(0, -1, dash_visual, basePosition=[dash_x, dash_y, 0.01],
                           baseOrientation=orientation)
        dist += dash_length + dash_gap

# Road network: start corridor -> 90-degree junction -> diagonal
# shortcut -> final straight stretch into the gate
ROAD_SEGMENTS = [
    # Main taxiway (full width, bottom) -- this is where cars drive
    ((0, 0),  (28, 0)),
    # Service road (parallel, above)
    ((0, 8),  (28, 8)),
    # Cross-connectors linking taxiway to service road
    ((6,  0), (6,  8)),    # connector A
    ((14, 0), (14, 8)),    # connector B
    ((22, 0), (22, 8)),    # connector C
    # Apron perimeter
    ((4,  12), (24, 12)),  # apron front
    ((4,  18), (24, 18)),  # apron back
    ((4,  12), (4,  18)),  # apron left wall
    ((24, 12), (24, 18)),  # apron right wall
    # Gate connectors (service road -> apron)
    ((8,  8),  (8,  12)),
    ((14, 8),  (14, 12)),
    ((20, 8),  (20, 12)),
    # Apron interior taxilines
    ((8,  12), (8,  18)),
    ((14, 12), (14, 18)),
    ((20, 12), (20, 18)),
    ((4,  15), (24, 15)),  # apron midline
    # Gate bay connector: from apron midline up to the gate bay entrance
    ((14, 15), (14, 20)),  # approach taxiline to gate bay
    # Gate bay interior: single centreline through the bay
    ((14, 20), (14, 26)),  # inside the gate bay
]

for (start, end) in ROAD_SEGMENTS:
    draw_road_segment(start[0], start[1], end[0], end[1])

# Free-roam camera state (controlled with arrow keys)
camera_yaw = 50
camera_pitch = -40
camera_distance = 28
camera_target = [14, 12, 0]

p.resetDebugVisualizerCamera(
    cameraDistance=camera_distance,
    cameraYaw=camera_yaw,
    cameraPitch=camera_pitch,
    cameraTargetPosition=camera_target
)

def update_camera_from_keys():
    """Reads arrow key presses each frame and moves the camera target
    around the scene. Up/Down moves forward/back along the camera's
    facing direction, Left/Right strafes sideways.

    Before applying any keyboard movement, we read PyBullet's actual
    current camera state via getDebugVisualizerCamera(). This picks up
    any changes made by mouse dragging/zooming, so keyboard and mouse
    control can coexist without one overwriting the other.
    """
    global camera_yaw, camera_pitch, camera_distance, camera_target

    # Read the camera's true current state (reflects any mouse input)
    cam_info = p.getDebugVisualizerCamera()
    # cam_info indices: 8=yaw, 9=pitch, 10=distance, 11=target
    camera_yaw = cam_info[8]
    camera_pitch = cam_info[9]
    camera_distance = cam_info[10]
    camera_target = list(cam_info[11])

    keys = p.getKeyboardEvents()
    move_speed = 0.1
    key_pressed = False

    # Forward direction based on current yaw
    yaw_rad = math.radians(camera_yaw)
    forward = [-math.sin(yaw_rad), math.cos(yaw_rad)]
    right = [math.cos(yaw_rad), math.sin(yaw_rad)]

    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
        camera_target[0] += forward[0] * move_speed
        camera_target[1] += forward[1] * move_speed
        key_pressed = True
    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
        camera_target[0] -= forward[0] * move_speed
        camera_target[1] -= forward[1] * move_speed
        key_pressed = True
    if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
        camera_target[0] -= right[0] * move_speed
        camera_target[1] -= right[1] * move_speed
        key_pressed = True
    if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
        camera_target[0] += right[0] * move_speed
        camera_target[1] += right[1] * move_speed
        key_pressed = True

    # Only force the camera position when an arrow key was pressed.
    # Since we just read the camera's real current state above, this
    # applies the movement on top of wherever the mouse last left it,
    # instead of snapping back to a stale remembered position.
    if key_pressed:
        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=camera_yaw,
            cameraPitch=camera_pitch,
            cameraTargetPosition=camera_target
        )

robot = p.loadURDF("husky/husky.urdf", [0, 0, 0.1])

# Let the robot fully settle onto the ground under gravity before any
# driving begins. Without this, the very first stepSimulation() call
# resolves the initial fall + ground contact all at once, and if the
# four wheels don't all touch down in perfect sync, the chassis can
# pick up a small unwanted yaw (rotation) right at touchdown -- which,
# with no compass-heading correction (only relative-to-waypoint
# correction), can show up as the robot bowing off to one side during
# the first leg of the drive instead of going straight.
for joint in [2, 3, 4, 5]:
    p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=15)
for _ in range(240):  # 0.5s at 480Hz -- plenty for a rigid body to settle
    p.stepSimulation()

# Obstacle positions placed near the road network (slightly off the
# centerline, like cones placed along a real corridor)
# No static cone obstacles in this version -- traffic cars on the
# main taxiway are the dynamic obstacles the robot must navigate around.
obstacle_positions = []

cone_body_ids = []
for (ox, oy) in obstacle_positions:
    # Traffic cone: a tapered cylinder collision shape (PyBullet doesn't
    # have a true cone primitive, so we approximate one with a narrow
    # cylinder for the body) plus a wider flat base disc, both bright
    # safety-yellow like a real airport floor cone.
    cone_collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=0.5)
    cone_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.2, length=0.5,
        rgbaColor=[1.0, 0.75, 0.0, 1]
    )
    cone_id = p.createMultiBody(1, cone_collision, cone_visual, basePosition=[ox, oy, 0.25])
    cone_body_ids.append(cone_id)

    # Flat wider base disc so it reads visually as a cone sitting on
    # the floor, with an orange-red reflective stripe ring
    base_visual = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.32, length=0.04,
        rgbaColor=[1.0, 0.4, 0.0, 1]
    )
    p.createMultiBody(0, -1, base_visual, basePosition=[ox, oy, 0.02])

start_pos = (0, 0)       # depot at far left of main taxiway
target_pos = (14, 23)    # inside the gate bay

# ---------------------------------------------------------
# LEARNED OBSTACLE MEMORY (persists across runs in a JSON file)
# ---------------------------------------------------------
# Every time the robot physically collides with something, the
# collision location gets appended here and saved to disk. On the
# NEXT run of the script, these remembered locations are loaded back
# in and treated as additional obstacles during path planning, so the
# robot avoids them from the start instead of re-discovering them by
# crashing into them again.
MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "learned_obstacles.json")

def load_learned_obstacles():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, "r") as f:
                data = json.load(f)
                return [tuple(pt) for pt in data.get("learned_hits", [])]
        except (json.JSONDecodeError, KeyError):
            return []
    return []

def save_learned_obstacles(points):
    with open(MEMORY_FILE, "w") as f:
        json.dump({"learned_hits": [list(pt) for pt in points]}, f, indent=2)

learned_obstacles = load_learned_obstacles()
if learned_obstacles:
    print(f"Loaded {len(learned_obstacles)} previously learned obstacle "
          f"location(s) from earlier runs.")

# Mark learned-obstacle locations on the floor in dark red, so you can
# visually see what the robot "remembers" from past collisions
for (lx, ly) in learned_obstacles:
    learned_marker = p.createVisualShape(
        p.GEOM_CYLINDER, radius=0.35, length=0.015,
        rgbaColor=[0.6, 0.05, 0.05, 0.7]
    )
    p.createMultiBody(0, -1, learned_marker, basePosition=[lx, ly, 0.02])

# ---------------------------------------------------------
# GATE BAY: walled enclosure at y=20-26, centred on x=14
# Three walls (left, right, back) + open front at y=20.
# No cars operate inside -- it's an isolated delivery zone.
# ---------------------------------------------------------
def make_wall(cx, cy, cz, hx, hy, hz, color=[0.75, 0.75, 0.78, 1]):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                               rgbaColor=color)
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
    return p.createMultiBody(0, col, vis, basePosition=[cx, cy, cz])

wall_body_ids = []

WALL_T = 0.2   # wall thickness half-extent
WALL_H = 1.2   # wall height half-extent
BAY_X  = 14.0  # bay centre x
BAY_Y0 = 20.0  # bay entrance y
BAY_Y1 = 26.0  # bay back wall y
BAY_W  = 4.0   # bay half-width (total width = 8m)

# Left wall
wall_body_ids.append(make_wall(BAY_X - BAY_W, (BAY_Y0 + BAY_Y1) / 2, WALL_H,
                                WALL_T, (BAY_Y1 - BAY_Y0) / 2, WALL_H))
# Right wall
wall_body_ids.append(make_wall(BAY_X + BAY_W, (BAY_Y0 + BAY_Y1) / 2, WALL_H,
                                WALL_T, (BAY_Y1 - BAY_Y0) / 2, WALL_H))
# Back wall
wall_body_ids.append(make_wall(BAY_X, BAY_Y1, WALL_H,
                                BAY_W + WALL_T, WALL_T, WALL_H))

# Gate sign above the entrance
gate_sign = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[1.5, 0.05, 0.35],
    rgbaColor=[0.05, 0.4, 0.15, 1])
p.createMultiBody(0, -1, gate_sign,
                   basePosition=[BAY_X, BAY_Y0, 2.8])

# Ground marker inside the bay
gate_floor = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[BAY_W - 0.3, (BAY_Y1 - BAY_Y0) / 2 - 0.3, 0.01],
    rgbaColor=[0.1, 0.55, 0.25, 0.5])
p.createMultiBody(0, -1, gate_floor,
                   basePosition=[BAY_X, (BAY_Y0 + BAY_Y1) / 2, 0.012])

# Target post inside the bay
gate_post = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.06, 0.06, 0.9], rgbaColor=[0.2, 0.2, 0.2, 1])
p.createMultiBody(0, -1, gate_post,
                   basePosition=[target_pos[0], target_pos[1], 0.9])

# ---------------------------------------------------------
# PARKED PLANE: static aircraft docked at the gate bay,
# sitting to the right of the bay (x > BAY_X + BAY_W).
# Built from simple box shapes: fuselage, wings, tail.
# ---------------------------------------------------------
PLANE_X  = BAY_X + BAY_W + 5.0   # fuselage centre x (right of bay)
PLANE_Y  = (BAY_Y0 + BAY_Y1) / 2  # fuselage centre y (aligned with bay)
PLANE_Z  = 0.6                     # fuselage centre height
PLANE_COL = [0.92, 0.92, 0.95, 1]  # near-white aluminium colour

def make_plane_part(cx, cy, cz, hx, hy, hz, color=PLANE_COL):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz],
                               rgbaColor=color)
    p.createMultiBody(0, -1, vis, basePosition=[cx, cy, cz])

# Fuselage (long tube along x-axis)
make_plane_part(PLANE_X, PLANE_Y, PLANE_Z, 6.0, 0.7, 0.7)
# Wings (wide flat surface perpendicular to fuselage)
make_plane_part(PLANE_X, PLANE_Y, PLANE_Z + 0.1, 2.5, 4.5, 0.12)
# Tail fin (vertical)
make_plane_part(PLANE_X + 5.2, PLANE_Y, PLANE_Z + 1.0, 0.1, 0.6, 1.0)
# Horizontal tail (small horizontal stabiliser)
make_plane_part(PLANE_X + 5.2, PLANE_Y, PLANE_Z + 0.35, 0.8, 1.8, 0.08)
# Nose cone (slightly narrower box at the front)
make_plane_part(PLANE_X - 6.4, PLANE_Y, PLANE_Z, 0.5, 0.45, 0.45,
                color=[0.85, 0.85, 0.88, 1])
# Engine pods (two, under the wings)
make_plane_part(PLANE_X - 0.5, PLANE_Y + 3.0, PLANE_Z - 0.4, 1.2, 0.4, 0.35,
                color=[0.6, 0.6, 0.65, 1])
make_plane_part(PLANE_X - 0.5, PLANE_Y - 3.0, PLANE_Z - 0.4, 1.2, 0.4, 0.35,
                color=[0.6, 0.6, 0.65, 1])
# Windows strip along fuselage (thin dark band)
make_plane_part(PLANE_X - 1.0, PLANE_Y, PLANE_Z + 0.5, 4.0, 0.05, 0.12,
                color=[0.3, 0.45, 0.65, 1])
# Airline stripe along fuselage
make_plane_part(PLANE_X - 1.0, PLANE_Y, PLANE_Z + 0.25, 4.5, 0.71, 0.05,
                color=[0.05, 0.25, 0.65, 1])

# ---------------------------------------------------------
# 2. BUILD A GRID MAP FOR A*
# ---------------------------------------------------------
GRID_SIZE = 0.5          # each grid cell is 0.5m x 0.5m
GRID_W, GRID_H = 66, 62  # expanded to cover gate bay at y=20-26
ORIGIN_X, ORIGIN_Y = -2, -3

def world_to_grid(x, y):
    gx = int((x - ORIGIN_X) / GRID_SIZE)
    gy = int((y - ORIGIN_Y) / GRID_SIZE)
    return gx, gy

def grid_to_world(gx, gy):
    x = gx * GRID_SIZE + ORIGIN_X
    y = gy * GRID_SIZE + ORIGIN_Y
    return x, y

# Mark grid cells blocked if they fall within an obstacle's radius.
# This now includes both the known cone positions AND any locations
# learned from collisions in previous runs.
OBSTACLE_RADIUS = 0.7   # safety margin around each obstacle. Lowered
                         # from 0.8 to fit safely within the tightened
                         # ROAD_HALF_WIDTH corridor above -- at 0.8, an
                         # obstacle near the corridor's edge could block
                         # the entire road width at that point, leaving
                         # A* with no way through at all.
LEARNED_RADIUS = 0.7    # slightly tighter margin for learned points (single collision point, not a known cone footprint)
blocked = set()
for (ox, oy) in obstacle_positions:
    ogx, ogy = world_to_grid(ox, oy)
    cell_radius = int(OBSTACLE_RADIUS / GRID_SIZE) + 1
    for dx in range(-cell_radius, cell_radius + 1):
        for dy in range(-cell_radius, cell_radius + 1):
            gx, gy = ogx + dx, ogy + dy
            wx, wy = grid_to_world(gx, gy)
            if math.hypot(wx - ox, wy - oy) <= OBSTACLE_RADIUS:
                blocked.add((gx, gy))

for (lx, ly) in learned_obstacles:
    lgx, lgy = world_to_grid(lx, ly)
    cell_radius = int(LEARNED_RADIUS / GRID_SIZE) + 1
    for dx in range(-cell_radius, cell_radius + 1):
        for dy in range(-cell_radius, cell_radius + 1):
            gx, gy = lgx + dx, lgy + dy
            wx, wy = grid_to_world(gx, gy)
            if math.hypot(wx - lx, wy - ly) <= LEARNED_RADIUS:
                blocked.add((gx, gy))

# Block grid cells that are too far from any road segment, so the
# A* path is forced to stay on the road network rather than cutting
# across open ground between segments.
def distance_to_segment(px, py, x1, y1, x2, y2):
    """Shortest distance from point (px,py) to line segment (x1,y1)-(x2,y2)."""
    seg_dx, seg_dy = x2 - x1, y2 - y1
    seg_len_sq = seg_dx ** 2 + seg_dy ** 2
    if seg_len_sq == 0:
        return math.hypot(px - x1, py - y1)
    t = max(0, min(1, ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq))
    proj_x = x1 + t * seg_dx
    proj_y = y1 + t * seg_dy
    return math.hypot(px - proj_x, py - proj_y)

def closest_point_on_segment(px, py, x1, y1, x2, y2):
    """Like distance_to_segment, but returns the projected point itself
    (x, y) rather than just the distance."""
    seg_dx, seg_dy = x2 - x1, y2 - y1
    seg_len_sq = seg_dx ** 2 + seg_dy ** 2
    if seg_len_sq == 0:
        return (x1, y1)
    t = max(0, min(1, ((px - x1) * seg_dx + (py - y1) * seg_dy) / seg_len_sq))
    return (x1 + t * seg_dx, y1 + t * seg_dy)

def snap_to_road_centerline(px, py):
    """Find the nearest point on ANY road segment's true centerline to
    (px, py). Used only for drawing the yellow overlay tightly onto the
    road art -- A*'s actual grid-based waypoints (used for driving and
    obstacle avoidance) are left untouched."""
    best_point = (px, py)
    best_dist = float("inf")
    for (seg_start, seg_end) in ROAD_SEGMENTS:
        cand = closest_point_on_segment(px, py, seg_start[0], seg_start[1],
                                         seg_end[0], seg_end[1])
        d = math.hypot(px - cand[0], py - cand[1])
        if d < best_dist:
            best_dist = d
            best_point = cand
    return best_point

ROAD_HALF_WIDTH = 1.0  # how far off the road centerline driving is still
                       # allowed. NOTE: this was originally 2.0, which was
                       # wide enough that A* could find a "shortcut" grid
                       # path cutting diagonally across the field near the
                       # 90-degree turn -- a diagonal line from partway
                       # down the first straight to partway up the second
                       # straight stayed within 2.0m of several road
                       # segments simultaneously, so A* (which only
                       # minimizes distance) took that cheaper diagonal
                       # instead of following the road's actual L-shape.
                       # That's what was causing the planned route (and
                       # therefore the robot itself) to visibly bow away
                       # from the road on what should have been a
                       # straight stretch. 1.0 is tight enough to force
                       # A* to track the real road shape (only cutting
                       # corners where the road itself bends), while
                       # still leaving enough width for OBSTACLE_RADIUS
                       # below to fit without blocking the whole corridor.
for gx in range(GRID_W):
    for gy in range(GRID_H):
        wx, wy = grid_to_world(gx, gy)
        on_any_road = False
        for (seg_start, seg_end) in ROAD_SEGMENTS:
            if distance_to_segment(wx, wy, seg_start[0], seg_start[1],
                                    seg_end[0], seg_end[1]) <= ROAD_HALF_WIDTH:
                on_any_road = True
                break
        if not on_any_road:
            blocked.add((gx, gy))

# ---------------------------------------------------------
# 2b. MOVING CARS (traffic on the roads)
# ---------------------------------------------------------
# Four cars drive back and forth along the horizontal roads. They are
# kinematic bodies -- position is set directly each frame rather than
# being physics-driven, so they move predictably and don't destabilise
# the simulation.

# One car per horizontal road, excluding the main taxiway (y=0): the
# robot drives ALONG that road from the depot to the connector-B turn
# at x=14, so a car there would meet it head-on in its own lane --
# and the robot has no same-lane avoidance behaviour, only the
# after-contact recovery. The Q-learning gap-acceptance agent instead
# lives at the SERVICE ROAD crossing (x=14, y=8), which the robot
# crosses perpendicular to this road's traffic.
# Each car bounces between its road's endpoints. Start position and
# direction are randomised per run (below), so every episode presents
# different traffic timing.
# Format: y, x_min, x_max, speed magnitude, colour
car_configs = [
    {"y": 8.0,  "xmin": 0.0,  "xmax": 28.0, "speed": 4.0, "color": [0.9, 0.2, 0.2, 1]},  # service road
    {"y": 12.0, "xmin": 4.0,  "xmax": 24.0, "speed": 3.5, "color": [0.2, 0.5, 0.9, 1]},  # apron front
    {"y": 15.0, "xmin": 4.0,  "xmax": 24.0, "speed": 5.0, "color": [0.9, 0.7, 0.1, 1]},  # apron midline
    {"y": 18.0, "xmin": 4.0,  "xmax": 24.0, "speed": 4.5, "color": [0.5, 0.2, 0.9, 1]},  # apron back
]
cars = []

def create_car(x, y, color):
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.8, 0.35, 0.25],
        rgbaColor=color)
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.8, 0.35, 0.25])
    body = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[x, y, 0.25])
    return body

for cfg in car_configs:
    start_x = random.uniform(cfg["xmin"], cfg["xmax"])
    speed = cfg["speed"] * (1 if random.random() < 0.5 else -1)
    body_id = create_car(start_x, cfg["y"], cfg["color"])
    cars.append({
        "x": start_x, "y": cfg["y"],
        "xmin": cfg["xmin"], "xmax": cfg["xmax"],
        "speed": speed, "body_id": body_id
    })
    print(f"Traffic: car on y={cfg['y']:.0f} starting at x={start_x:.1f}, "
          f"speed {speed:+.1f} m/s")

# Bodies the robot can physically hit that should trigger the
# reverse-and-rejoin recovery but must NOT be learned or added to the
# blocked map: cars move (their position at impact is stale the moment
# it's recorded) and the bay walls already sit outside the drivable
# road corridor, so neither belongs in persistent obstacle memory.
transient_obstacle_ids = [car["body_id"] for car in cars] + wall_body_ids

def update_cars(dt):
    """Advance each car along its road and bounce at the ends."""
    for car in cars:
        car["x"] += car["speed"] * dt
        if car["x"] > car["xmax"]:
            car["x"] = car["xmax"]
            car["speed"] = -abs(car["speed"])
        elif car["x"] < car["xmin"]:
            car["x"] = car["xmin"]
            car["speed"] = abs(car["speed"])
        p.resetBasePositionAndOrientation(
            car["body_id"],
            [car["x"], car["y"], 0.25],
            p.getQuaternionFromEuler([0, 0, 0]))

# ---------------------------------------------------------
# 2c. Q-LEARNING: gap acceptance at road crossings
# ---------------------------------------------------------
# The robot must decide at each crossing: WAIT or GO?
# This is a classic reinforcement learning problem solved with
# Q-learning -- a model-free RL algorithm that learns the value
# of (state, action) pairs purely from experience.
#
# STATE:  the gap distance to the nearest car at the crossing
#         point, discretised into 5 buckets, combined with whether
#         that car is APPROACHING or RECEDING -- 10 states total.
#         Direction matters as much as distance: a 6m gap behind a
#         receding car is completely safe, while the same 6m gap in
#         front of a fast approaching car is a near-miss. Without
#         the direction bit those two situations share one state,
#         and its learned Q-value converges to a useless average.
#
# ACTIONS: 0 = WAIT  (stay stopped, let the car pass)
#          1 = GO    (proceed through the crossing)
#
# REWARD:  +10 for crossing safely (GO and no collision)
#          -10 for a near-miss    (GO when a car was too close)
#           -1 per second waited  (small penalty for waiting,
#              so the robot is incentivised to find tighter gaps
#              rather than always waiting for a huge clear road)
#
# Q-TABLE: 10 states × 2 actions, stored as a flat list in JSON
#          and updated after every crossing decision using the
#          standard Q-learning update rule:
#          Q(s,a) += α * (r + γ * max(Q(s')) - Q(s,a))
#
# EXPLORATION: ε-greedy -- with probability ε the robot picks
#          a random action (explore), otherwise it picks the
#          action with the highest Q-value (exploit). ε decays
#          over runs so the robot explores freely early on and
#          converges to learned behaviour as data accumulates.
#
# Over multiple runs the Q-table converges: Q(large_gap, GO)
# becomes highly positive, Q(small_gap, GO) becomes negative,
# Q(small_gap, WAIT) becomes positive -- and the robot's
# crossing behaviour visibly improves.

Q_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "q_table.json")

# Gap buckets: distance from nearest car to the crossing point (metres)
GAP_BUCKETS = [3.0, 5.0, 7.0, 9.0]  # boundaries; 5 distance buckets,
                                     # x 2 car directions = 10 states
ACTION_WAIT = 0
ACTION_GO   = 1

# Q-learning hyperparameters
Q_ALPHA   = 0.3   # learning rate   -- how fast Q-values update
Q_GAMMA   = 0.9   # discount factor -- how much future reward matters
Q_EPSILON = 0.3   # initial exploration rate (decays over runs)
Q_EPSILON_DECAY = 0.85  # multiply epsilon by this each run
Q_EPSILON_MIN   = 0.05  # never go fully greedy

def load_q_table():
    n_states = (len(GAP_BUCKETS) + 1) * 2  # gap buckets x car direction
    if os.path.exists(Q_FILE):
        try:
            with open(Q_FILE) as f:
                data = json.load(f)
                if len(data["q_table"]) == n_states:
                    return (data["q_table"], data["epsilon"],
                            data["run_count"], data["total_crossings"])
                # A table saved before the direction bit existed (or
                # after any other state-layout change) can't be reused
                # -- its indices mean different situations now.
                print("Q-table on disk has an old state layout -- starting fresh.")
        except (json.JSONDecodeError, KeyError):
            pass
    q_table = [[0.0, 0.0] for _ in range(n_states)]  # [Q(s,WAIT), Q(s,GO)]
    return q_table, Q_EPSILON, 0, 0

def save_q_table(q_table, epsilon, run_count, total_crossings):
    with open(Q_FILE, "w") as f:
        json.dump({
            "q_table": q_table,
            "epsilon": round(epsilon, 4),
            "run_count": run_count,
            "total_crossings": total_crossings
        }, f, indent=2)

def gap_to_bucket(gap_metres):
    """Discretise a continuous gap distance into one of 5 buckets."""
    for i, boundary in enumerate(GAP_BUCKETS):
        if gap_metres < boundary:
            return i
    return len(GAP_BUCKETS)  # largest bucket

def q_state(gap_metres, approaching):
    """Combine gap bucket and car direction into a single Q-state
    index: bucket*2, plus 1 if the car is approaching the crossing."""
    return gap_to_bucket(gap_metres) * 2 + (1 if approaching else 0)

def crossing_gap_and_direction():
    """For the nearest car (by edge-to-crossing distance) on the
    Q-agent's road (the service road, y=8): return that gap and
    whether the car is moving TOWARD the crossing point. Using edge
    distance means the robot waits until the car has fully cleared
    the crossing, not just when its centre has passed."""
    road_cars = [car for car in cars if abs(car["y"] - Q_CROSSING_Y) < 0.5]
    best_gap, best_approaching = float("inf"), False
    for car in road_cars:
        gap = max(0.0, abs(car["x"] - Q_CROSSING_X) - CAR_HALF_WIDTH)
        if gap < best_gap:
            best_gap = gap
            best_approaching = ((car["speed"] > 0 and car["x"] < Q_CROSSING_X) or
                                (car["speed"] < 0 and car["x"] > Q_CROSSING_X))
    return best_gap, best_approaching

CAR_HALF_WIDTH = 0.8    # half the car's x-extent (from create_car halfExtents)
ROBOT_HALF_WIDTH = 0.5  # conservative estimate of robot's half-width
NEAR_MISS_DISTANCE = 2.0  # robot-to-car centre distance (m) below which a
                          # committed GO crossing counts as a near miss

def nearest_crossing_car_distance(px, py):
    """Actual centre-to-centre distance from the robot to the nearest
    car on the Q-agent's road. Unlike crossing_gap_and_direction()
    (car edge to the fixed crossing point), this measures the robot itself -- it's
    used to judge, AFTER a committed GO crossing, whether the crossing
    was genuinely safe, rather than rewarding the decision based only
    on the gap at the moment it was made."""
    road_cars = [car for car in cars if abs(car["y"] - Q_CROSSING_Y) < 0.5]
    if not road_cars:
        return float("inf")
    return min(math.hypot(car["x"] - px, car["y"] - py)
               for car in road_cars)

def car_time_to_crossing(road_y, crossing_x=14.0):
    """Soonest time (seconds) until any car on this road reaches the
    crossing point, from its position, speed AND direction -- not just
    its current distance. A car driving away from the crossing is only
    a threat after it bounces at its road end and comes back, so its
    time is measured along that full path. This is the
    predict-where-the-car-WILL-be upgrade over the old distance-only
    gap: a car 6m past the crossing and receding blocks nothing, while
    a car 6m out and closing fast blocks the crossing. A pure distance
    threshold also deadlocked geometrically: on the x=4-24 apron roads
    the largest possible gap is 9.2m, so any threshold above that
    could simply never be met."""
    road_cars = [car for car in cars if abs(car["y"] - road_y) < 0.5]
    soonest = float("inf")
    for car in road_cars:
        # A car whose body currently covers (or hangs over) the
        # crossing blocks it NOW, regardless of direction. Without
        # this, a car that has just passed the crossing point but
        # whose tail still overlaps it reads as "receding -- safe for
        # seconds", and the robot commits and sideswipes its rear
        # corner as it pulls away.
        if abs(car["x"] - crossing_x) - CAR_HALF_WIDTH <= ROBOT_HALF_WIDTH + 0.2:
            return 0.0
        if car["speed"] == 0:
            continue
        if car["speed"] > 0:
            if car["x"] <= crossing_x:
                dist = crossing_x - car["x"]  # closing directly
            else:  # receding -- bounces at xmax, then comes back
                dist = (car["xmax"] - car["x"]) + (car["xmax"] - crossing_x)
        else:
            if car["x"] >= crossing_x:
                dist = car["x"] - crossing_x  # closing directly
            else:  # receding -- bounces at xmin, then comes back
                dist = (car["x"] - car["xmin"]) + (crossing_x - car["xmin"])
        # Measure to the car's leading edge, not its centre.
        dist = max(0.0, dist - CAR_HALF_WIDTH)
        soonest = min(soonest, dist / abs(car["speed"]))
    return soonest

# Horizontal roads the robot crosses at x=14 under RULE-BASED control
# (the predicted-time threshold below). The service road (y=8) is NOT
# in this list: that crossing belongs to the Q-learning agent, which
# makes the WAIT/GO decision there itself.
CROSSING_ROADS = [
    {"y": 12.0, "label": "apron front"},
    {"y": 15.0, "label": "apron midline"},
    {"y": 18.0, "label": "apron back"},
]
MIN_CROSSING_TIME = 2.5  # seconds: hold unless every car on the road is
                         # at least this far (in time) from the crossing.
                         # The robot needs ~1.8s to clear the car lane
                         # from its hold point at ~1.7 m/s; 2.5 adds margin.

# While REJOINING (driving a collision detour) the Q-agent is never
# consulted, so the service road falls back to the rule-based guard
# for the duration of the detour.
REJOIN_CROSSING_ROADS = [{"y": 8.0, "label": "service road"}] + CROSSING_ROADS
CROSSING_STOP_MARGIN = 2.0    # start holding this far before each road
CROSSING_COMMIT_MARGIN = 1.0  # within this distance of the road centre the
                              # robot is entering the car's lane: COMMITTED.
                              # Never command a stop here -- accelerating
                              # through is always safer than freezing in
                              # the car's path, which is what the old
                              # window (extending 0.5m PAST the road) did.
CROSSING_X = 14.0

def choose_action(state, epsilon):
    """ε-greedy action selection."""
    if random.random() < epsilon:
        return random.choice([ACTION_WAIT, ACTION_GO])  # explore
    # Exploit: pick action with highest Q-value; default to GO on tie
    return ACTION_GO if q_table[state][ACTION_GO] >= q_table[state][ACTION_WAIT] else ACTION_WAIT

def q_update(state, action, reward, next_state):
    """Standard Q-learning update."""
    best_next = max(q_table[next_state])
    q_table[state][action] += Q_ALPHA * (
        reward + Q_GAMMA * best_next - q_table[state][action])

q_table, q_epsilon, q_run_count, q_total_crossings = load_q_table()
q_run_count += 1
print(f"Q-learning: run #{q_run_count}, ε={q_epsilon:.2f}, "
      f"{q_total_crossings} crossings learned so far")
bucket_labels = ["0-3m", "3-5m", "5-7m", "7-9m", "9m+"]

def state_label(s):
    """Human-readable name for a Q-state: gap bucket + car direction."""
    return f"{bucket_labels[s // 2]} {'appr' if s % 2 else 'rcdg'}"

print(f"Q-table (WAIT | GO) per state (gap bucket x car direction):")
for s in range(len(q_table)):
    print(f"  {state_label(s):>10}: WAIT={q_table[s][ACTION_WAIT]:+.2f}  "
          f"GO={q_table[s][ACTION_GO]:+.2f}")

# ---------------------------------------------------------
# 3. A* PATH PLANNING
# ---------------------------------------------------------
def heuristic(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def neighbors(cell):
    gx, gy = cell
    # Straight (90 degree) moves first, diagonal moves after.
    # Diagonal moves get an extra cost penalty below (in a_star),
    # so the planner favors a road-network look of straight segments
    # and only cuts diagonally when it saves real distance.
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:
        nx, ny = gx + dx, gy + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H and (nx, ny) not in blocked:
            yield (nx, ny)

DIAGONAL_PENALTY = 1.3  # >1.0 makes diagonal moves less attractive than straight ones

def a_star(start, goal):
    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        for nxt in neighbors(current):
            is_diagonal = (nxt[0] != current[0]) and (nxt[1] != current[1])
            step_cost = math.hypot(nxt[0]-current[0], nxt[1]-current[1])
            if is_diagonal:
                step_cost *= DIAGONAL_PENALTY
            tentative_g = g_score[current] + step_cost
            if nxt not in g_score or tentative_g < g_score[nxt]:
                g_score[nxt] = tentative_g
                came_from[nxt] = current
                f_score = tentative_g + heuristic(nxt, goal)
                heapq.heappush(open_set, (f_score, nxt))

    return None  # no path found

start_grid = world_to_grid(*start_pos)
goal_grid = world_to_grid(*target_pos)

print("Planning route with A*...")
grid_path = a_star(start_grid, goal_grid)

if grid_path is None:
    print("No path found! Check obstacle layout.")
    p.disconnect()
    exit()

# Convert grid path back to world waypoints
waypoints = [grid_to_world(gx, gy) for (gx, gy) in grid_path]

# ---------------------------------------------------------
# 3b. SMOOTH THE PATH (line-of-sight shortcutting)
# ---------------------------------------------------------
def _segments_adjacent(seg_a, seg_b, tol=0.05):
    """Two road segments are 'adjacent' if they share an endpoint
    (within a small tolerance), i.e. one's end is the other's start --
    meaning the road actually continues smoothly from one to the
    other, like the corner where the road turns."""
    (a1, a2) = seg_a
    (b1, b2) = seg_b
    for pa in [a1, a2]:
        for pb in [b1, b2]:
            if math.hypot(pa[0] - pb[0], pa[1] - pb[1]) <= tol:
                return True
    return False

def has_clear_line(p1, p2, blocked_cells, samples=20, max_segments_touched=2):
    """Checks whether a straight line between two world points is a
    legitimate shortcut.

    Every sampled point must be clear of blocked cells and sit near
    SOME road segment (within ROAD_HALF_WIDTH) -- otherwise the line
    has wandered off any road entirely.

    On top of that: across ALL sampled points, the line may pass near
    at most `max_segments_touched` distinct road segments overall, and
    those segments must all be mutually adjacent (i.e. form one
    continuous stretch of road, like a single straight section or one
    corner where two segments meet) rather than a scattered handful of
    unrelated segments.

    Without this second check, a long diagonal line cutting straight
    from near the start to near the goal can "graze" past several
    different, unrelated road segments in sequence (each one only
    slightly closer than the last as the line crosses the field),
    individually staying within tolerance of each at different points
    along its length -- which made the smoothing step wrongly approve
    cutting straight across open field and ignoring the road's actual
    shape entirely. Limiting it to 2 mutually-adjacent segments still
    allows the genuinely useful shortcuts: skipping ahead within one
    straight stretch, or cutting the inside of a single corner where
    the road bends from one segment into the next.
    """
    def nearest_segment(px, py):
        best_idx, best_dist = None, float("inf")
        for i, (s, e) in enumerate(ROAD_SEGMENTS):
            d = distance_to_segment(px, py, s[0], s[1], e[0], e[1])
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_dist > ROAD_HALF_WIDTH:
            return None  # not near any road at all
        return best_idx

    touched_segments = []
    for i in range(samples + 1):
        t = i / samples
        x = p1[0] + (p2[0] - p1[0]) * t
        y = p1[1] + (p2[1] - p1[1]) * t

        if world_to_grid(x, y) in blocked_cells:
            return False

        seg_idx = nearest_segment(x, y)
        if seg_idx is None:
            return False  # this point isn't near any road at all

        if seg_idx not in touched_segments:
            touched_segments.append(seg_idx)

    if len(touched_segments) > max_segments_touched:
        return False

    # All touched segments must be mutually adjacent -- forming one
    # continuous stretch/corner, not a scattered set.
    for i in range(len(touched_segments) - 1):
        seg_a = ROAD_SEGMENTS[touched_segments[i]]
        seg_b = ROAD_SEGMENTS[touched_segments[i + 1]]
        if not _segments_adjacent(seg_a, seg_b):
            return False

    return True

def smooth_path(points, blocked_cells):
    """Greedy shortcutting: from each point, jump as far ahead as
    possible while the straight line to that point stays clear of
    obstacles. Turns the jagged grid-staircase path into a smooth,
    natural-looking route with far fewer turns."""
    if len(points) <= 2:
        return points

    smoothed = [points[0]]
    current_index = 0

    while current_index < len(points) - 1:
        next_index = current_index + 1
        for candidate in range(len(points) - 1, current_index, -1):
            if has_clear_line(points[current_index], points[candidate], blocked_cells):
                next_index = candidate
                break
        smoothed.append(points[next_index])
        current_index = next_index

    return smoothed

print(f"Path found with {len(waypoints)} grid steps (used only as a "
      f"sanity check that the road network is connected).")

# ---------------------------------------------------------
# 3c. THE IDEAL ROUTE: literal road centerline, not a grid path
# ---------------------------------------------------------
# The robot's NORMAL driving target is no longer A*'s grid-snapped
# path -- it's the literal centerline defined by ROAD_SEGMENTS itself
# (each segment's start point, plus the final segment's end point).
# Since ROAD_SEGMENTS is already a connected chain (each segment's end
# is the next one's start), this is simply the segment boundaries in
# order. This guarantees the yellow line AND the robot's actual drive
# target are EXACTLY the road's true shape -- straight where the road
# is straight, diagonal where it's diagonal -- with no 0.5m grid
# quantization or "shortcut" approximation involved at all.
#
# A* and smooth_path (above) are NOT thrown away -- they're still used
# later, but only reactively: if the robot collides with an obstacle
# sitting on this line, a short grid-based detour gets planned around
# just that obstacle, and the robot steers back onto this exact same
# ideal_route afterward. Normal, obstacle-free driving never touches
# A* at all.
# Explicit path: depot -> along main taxiway -> up connector B ->
# along service road -> up Gate B connector -> into apron -> Gate B
ideal_route = [
    (0,  0),    # depot
    (14, 0),    # taxiway at connector B base
    (14, 8),    # top of connector B / service road
    (14, 12),   # Gate B connector / apron entrance
    (14, 15),   # apron midline
    (14, 20),   # gate bay entrance
    (14, 23),   # inside gate bay (target)
]

print(f"Ideal route: {len(ideal_route)} waypoints tracing the literal road centerline.")

# Draw the ideal route in bold yellow, overlaid directly on top of the
# road's dashed centerline (z=0.013, just above the dashes at z=0.012
# so it doesn't z-fight/flicker with them). Since ideal_route is now
# the literal centerline itself, no snapping is needed for drawing --
# it already matches the dashes exactly. This is drawn once, here, and
# never changes for the rest of the run -- even if the robot later has
# to divert around a collision, this yellow line stays as a reference
# for the lane it was meant to follow.
drawn_route = ideal_route

IDEAL_ROUTE_COLOR = [1, 0.85, 0]  # bold gold-yellow, solid (vs. the
                                   # road's dashed centerline) so the
                                   # two are still visually distinct
                                   # even though they overlap
for i in range(len(drawn_route) - 1):
    p.addUserDebugLine(
        [drawn_route[i][0], drawn_route[i][1], 0.013],
        [drawn_route[i+1][0], drawn_route[i+1][1], 0.013],
        lineColorRGB=IDEAL_ROUTE_COLOR,
        lineWidth=5
    )

# Capture mode: debug lines don't exist in offscreen software renders,
# so draw the same ideal route again as thin gold box strips, and set
# up the fixed capture camera.
if SIM_CAPTURE:
    for i in range(len(drawn_route) - 1):
        x1, y1 = drawn_route[i]
        x2, y2 = drawn_route[i + 1]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if seg_len == 0:
            continue
        strip = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[seg_len / 2, 0.07, 0.012],
            rgbaColor=[1, 0.85, 0, 1])
        p.createMultiBody(0, -1, strip,
                          basePosition=[(x1 + x2) / 2, (y1 + y2) / 2, 0.014],
                          baseOrientation=p.getQuaternionFromEuler(
                              [0, 0, math.atan2(y2 - y1, x2 - x1)]))

    import numpy as np  # needed to snapshot frames (see capture loop)

    # Neutral ground cover: hides the default checkerboard plane in
    # captures. Sits just above the plane (top at z=0.001) and below
    # every road/route element (roads reach z=0.015, dashes 0.012+).
    ground_vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[150, 150, 0.0005],
        rgbaColor=[0.80, 0.82, 0.79, 1])
    p.createMultiBody(0, -1, ground_vis, basePosition=[14, 12, 0.0005])

    # (The robot marker in captures is the GO/STOP signal light defined
    # below -- it follows the robot and shows its drive command.)

    # Second panel for the demo GIF: the same navigation map the live
    # window shows, rendered offscreen through nav_map's own drawing
    # code so the two can never drift apart. Sized to match the 3D
    # panel's height so they composite side by side.
    MAP_PANEL_PX = CAPTURE_SIZE[1]
    capture_map = None
    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import nav_map as navmap

        map_fig, map_ax, map_art = navmap.build_figure(
            {
                "roads": [[list(s), list(e)] for (s, e) in ROAD_SEGMENTS],
                "route": [list(pt) for pt in ideal_route],
                "bay": [BAY_X - BAY_W, BAY_Y0, BAY_X + BAY_W, BAY_Y1],
                "target": list(target_pos),
                "car_colors": [cfg["color"][:3] for cfg in car_configs],
            },
            figsize=(MAP_PANEL_PX / 100.0, MAP_PANEL_PX / 100.0),
            controls=False)
        map_fig.set_dpi(100)
        capture_map = dict(fig=map_fig, ax=map_ax, art=map_art,
                           trail_x=[], trail_y=[])
    except Exception as exc:
        print(f"Map panel unavailable for capture ({exc}) -- "
              f"capturing the 3D view only.")

    def render_map_panel(rx, ry, hd, signal, remaining, speed, state_name):
        """Draw one navigation-map frame and return it as an RGB array."""
        cap = capture_map
        art = cap["art"]
        art["robot"].set_xy(navmap.robot_marker(rx, ry, hd))
        color = navmap.GO_COLOR if signal == "GO" else navmap.STOP_COLOR
        art["robot"].set_facecolor(color)
        art["banner"].set_text(signal)
        art["banner"].get_bbox_patch().set_facecolor(color)

        tx, ty = cap["trail_x"], cap["trail_y"]
        if not tx or math.hypot(rx - tx[-1], ry - ty[-1]) > 0.12:
            tx.append(rx)
            ty.append(ry)
            art["trail"].set_data(tx, ty)

        for patch, car in zip(art["cars"], cars):
            patch.set_xy((car["x"] - 0.8, car["y"] - 0.35))

        eta = f"{remaining / speed:.0f}s" if speed > 0.25 else "--"
        art["readout"].set_text(
            f"{remaining:5.1f} m to gate     ETA {eta}     "
            f"{speed:4.1f} m/s     {state_name}")

        navmap.apply_view(cap["ax"], rx, ry)
        cap["fig"].canvas.draw()
        return np.asarray(cap["fig"].canvas.buffer_rgba())[:, :, :3]

    capture_frames = []
    next_capture_time = 0.0
    last_capture_marker = start_pos
    CAPTURE_VIEW = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[10, 10.5, 0], distance=23,
        yaw=55, pitch=-52, roll=0, upAxisIndex=2)
    CAPTURE_PROJ = p.computeProjectionMatrixFOV(
        fov=60, aspect=CAPTURE_SIZE[0] / CAPTURE_SIZE[1],
        nearVal=0.1, farVal=100)

# ---------------------------------------------------------
# GO/STOP SIGNAL LIGHT (mounted above the robot)
# ---------------------------------------------------------
# A status light hovers above the robot showing the wheel command
# currently being sent: GREEN while it is commanded to move, RED while
# it is commanded to hold (Q-agent wait, rule-based crossing hold, or
# a collision stop). Being part of the scene, it is visible from any
# third-person camera angle in the GUI and also renders in offscreen
# GIF captures -- where it doubles as the robot marker (the husky
# itself is only ~10px at capture zoom).
drive_signal = "GO"         # recomputed every frame by the control logic
light_shown_signal = None
SIGNAL_LIGHT_HEIGHT = 0.55  # metres above the robot's base
SIGNAL_GO_COLOR = [0.1, 0.8, 0.2, 1]
SIGNAL_STOP_COLOR = [0.9, 0.1, 0.1, 1]

_light_vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.25,
                                 rgbaColor=SIGNAL_GO_COLOR)
signal_light_id = p.createMultiBody(
    0, -1, _light_vis,
    basePosition=[start_pos[0], start_pos[1], SIGNAL_LIGHT_HEIGHT])

# ---------------------------------------------------------
# NAVIGATION MAP WINDOW (separate process -- see nav_map.py)
# ---------------------------------------------------------
# A 2D turn-by-turn style map opens alongside the 3D view. It runs as
# its own process because a matplotlib redraw costs ~60ms; inline that
# would stall this 480Hz loop and wreck real-time pacing. Here the sim
# only writes a small JSON line per update, and the pipe write is
# non-blocking, so a slow or closed map window can never hold up the
# simulation.
map_proc = None

def start_nav_map():
    """Launch the map process and send it the static layout."""
    global map_proc
    import subprocess
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "nav_map.py")
    try:
        # stdout is discarded so the map process can never hold open a
        # pipeline reading the simulation's output (its stderr is kept
        # so genuine errors still surface).
        map_proc = subprocess.Popen(
            [sys.executable, script], stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL, text=True)
        os.set_blocking(map_proc.stdin.fileno(), False)
        send_to_map({
            "type": "init",
            "roads": [[list(s), list(e)] for (s, e) in ROAD_SEGMENTS],
            "route": [list(pt) for pt in ideal_route],
            "bay": [BAY_X - BAY_W, BAY_Y0, BAY_X + BAY_W, BAY_Y1],
            "target": list(target_pos),
            "car_colors": [cfg["color"][:3] for cfg in car_configs],
        })
        print("Navigation map window opened (set SIM_MAP=0 to disable).")
    except Exception as exc:
        print(f"Navigation map unavailable ({exc}) -- continuing without it.")
        map_proc = None

def send_to_map(payload):
    """Non-blocking write; drops the update rather than stalling the
    sim if the pipe is full, and disables the map if it has closed."""
    global map_proc
    if map_proc is None or map_proc.stdin is None:
        return
    try:
        map_proc.stdin.write(json.dumps(payload) + "\n")
        map_proc.stdin.flush()
    except (BlockingIOError, InterruptedError):
        pass          # map is behind -- skip this frame's update
    except (BrokenPipeError, ValueError, OSError):
        map_proc = None   # window closed; carry on without it

def update_signal_light(signal, robot_pos):
    """Keep the light riding above the robot; recolor on signal change."""
    global light_shown_signal
    p.resetBasePositionAndOrientation(
        signal_light_id,
        [robot_pos[0], robot_pos[1], robot_pos[2] + SIGNAL_LIGHT_HEIGHT],
        p.getQuaternionFromEuler([0, 0, 0]))
    if signal != light_shown_signal:
        p.changeVisualShape(signal_light_id, -1,
                            rgbaColor=(SIGNAL_GO_COLOR if signal == "GO"
                                       else SIGNAL_STOP_COLOR))
        light_shown_signal = signal

# ---------------------------------------------------------
# 4. DRIVE THE ROBOT ALONG THE WAYPOINTS
# ---------------------------------------------------------
WHEEL_JOINTS = [2, 3, 4, 5]  # husky wheel joint indices
ARRIVAL_RADIUS = 0.3  # slightly widened from 0.25 to suit the faster
                       # top speed below -- still hugs the path closely
LOOKAHEAD_DISTANCE = 1.0  # meters ahead along ideal_route to aim for
                           # while driving normally -- this is the core
                           # of how the robot tracks the literal road
                           # centerline: each frame, find the nearest
                           # point on the line to the robot's current
                           # position, then steer toward a point this
                           # far further along it. Smaller values hug
                           # the line more tightly but can wobble;
                           # larger values are smoother but cut corners
                           # more.

# Driving speed/turning tuning, shared by both STATE_DRIVING and
# STATE_REJOINING so the two behave identically.
MAX_SPEED = 10.0
TURN_SPEED_FLOOR = 0.15
TURN_GAIN = 8
TURN_SEVERITY_ANGLE = math.pi / 3.2
WHEEL_FORCE = 35

# Collisions detected this run get added here, then saved to disk
# at the end (or immediately, so a crash mid-run still keeps them)
newly_learned = []
COLLISION_COOLDOWN = 2.0  # seconds before the same cone can be "learned" again this run
last_collision_time = {}

# Cones already diverted around this run -- once a cone has triggered a
# divert, its blocked-cell radius is added to a *runtime* copy of the
# blocked set so any future planning avoids it too, without touching the
# original `blocked` set (which stays as the static map used to draw
# the fixed yellow ideal route reference).
runtime_blocked = set(blocked)
diverted_cone_ids = set()

def block_cells_around(wx, wy, radius=LEARNED_RADIUS):
    """Add grid cells within `radius` of a world point to runtime_blocked,
    same logic used for learned_obstacles above, so replanning routes
    around a freshly-hit cone."""
    cgx, cgy = world_to_grid(wx, wy)
    cell_radius = int(radius / GRID_SIZE) + 1
    for dx in range(-cell_radius, cell_radius + 1):
        for dy in range(-cell_radius, cell_radius + 1):
            gx, gy = cgx + dx, cgy + dy
            gwx, gwy = grid_to_world(gx, gy)
            if math.hypot(gwx - wx, gwy - wy) <= radius:
                runtime_blocked.add((gx, gy))

def find_path_with_runtime_blocked(start_world, goal_world):
    """Run A* between two world points using runtime_blocked instead of
    the static `blocked` set, returning a smoothed waypoint list (or
    None if no path exists)."""
    start_g = world_to_grid(*start_world)
    goal_g = world_to_grid(*goal_world)

    global blocked
    saved_blocked = blocked
    blocked = runtime_blocked
    try:
        grid_path = a_star(start_g, goal_g)
    finally:
        blocked = saved_blocked

    if grid_path is None:
        return None
    new_waypoints = [grid_to_world(gx, gy) for (gx, gy) in grid_path]
    return smooth_path(new_waypoints, runtime_blocked)

def point_at_distance_along_route(route, start_idx, distance_ahead):
    """Walk forward along `route` (a list of waypoints) starting from
    segment index `start_idx`, advancing `distance_ahead` meters along
    the route's own straight segments, and return the resulting point
    plus which segment index it ended up on. Clamps to the route's
    final point if `distance_ahead` overshoots the end."""
    remaining = distance_ahead
    idx = start_idx
    while idx < len(route) - 1:
        x1, y1 = route[idx]
        x2, y2 = route[idx + 1]
        seg_len = math.hypot(x2 - x1, y2 - y1)
        if remaining <= seg_len:
            t = remaining / seg_len if seg_len > 0 else 0
            return (x1 + t * (x2 - x1), y1 + t * (y2 - y1)), idx
        remaining -= seg_len
        idx += 1
    return route[-1], len(route) - 1

def lookahead_point_on_route(route, current_world, lookahead_distance):
    """Project `current_world` onto the nearest point of `route`, then
    return a target point `lookahead_distance` further along the route
    from that exact projected point (clamped to the route's final
    point), plus the remaining distance from the projection to the
    route's very end. This is the main steering target while driving
    normally: it keeps the robot continuously correcting toward the
    literal road centerline itself, rather than chasing sparse
    discrete waypoints, and it makes "where do I resume driving" a
    non-issue since the nearest point on the route is always
    well-defined no matter where the robot currently is (e.g. just
    after finishing a detour)."""
    best_idx, best_t, best_dist = 0, 0.0, float("inf")
    for i in range(len(route) - 1):
        x1, y1 = route[i]
        x2, y2 = route[i + 1]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        seg_len_sq = seg_dx ** 2 + seg_dy ** 2
        if seg_len_sq == 0:
            continue
        t = max(0, min(1, ((current_world[0]-x1)*seg_dx + (current_world[1]-y1)*seg_dy) / seg_len_sq))
        proj_x, proj_y = x1 + t * seg_dx, y1 + t * seg_dy
        d = math.hypot(current_world[0] - proj_x, current_world[1] - proj_y)
        if d < best_dist:
            best_dist, best_idx, best_t = d, i, t

    x1, y1 = route[best_idx]
    x2, y2 = route[best_idx + 1]
    seg_len = math.hypot(x2 - x1, y2 - y1)
    remaining_in_current_seg = seg_len * (1 - best_t)

    # Total remaining distance to the route's end, from the projected
    # point -- used to detect "arrived" even though we're aiming at a
    # lookahead point further ahead, not the final point directly.
    remaining = remaining_in_current_seg
    for i in range(best_idx + 1, len(route) - 1):
        rx1, ry1 = route[i]
        rx2, ry2 = route[i + 1]
        remaining += math.hypot(rx2 - rx1, ry2 - ry1)

    # Walk forward lookahead_distance from the projected point itself.
    if lookahead_distance <= remaining_in_current_seg:
        t2 = best_t + lookahead_distance / seg_len if seg_len > 0 else best_t
        target = (x1 + t2 * (x2 - x1), y1 + t2 * (y2 - y1))
    else:
        target, _ = point_at_distance_along_route(
            route, best_idx + 1, lookahead_distance - remaining_in_current_seg)

    return target[0], target[1], remaining

def find_rejoin_target(current_world, search_step=0.3, max_search_distance=15.0):
    """Starting from the point on `ideal_route` nearest the robot's
    current (post-reverse) position, walk forward in small steps along
    the route's own literal geometry, and return the first point that's
    clear of every cell in `runtime_blocked` -- this is what lets the
    robot rejoin the EXACT same yellow line almost immediately once
    it's past the obstacle, rather than snapping to a distant sparse
    waypoint. Falls back to the final goal if nothing closer qualifies
    within max_search_distance."""
    # Find which route segment the robot is nearest to right now, and
    # its exact distance-along-route at that projected point (NOT just
    # the segment index -- we need the precise offset within the
    # segment too, otherwise the forward search below would start from
    # the segment's start corner instead of from where the robot
    # actually is).
    best_idx, best_t, best_dist = 0, 0.0, float("inf")
    for i in range(len(ideal_route) - 1):
        x1, y1 = ideal_route[i]
        x2, y2 = ideal_route[i + 1]
        seg_dx, seg_dy = x2 - x1, y2 - y1
        seg_len_sq = seg_dx ** 2 + seg_dy ** 2
        if seg_len_sq == 0:
            continue
        t = max(0, min(1, ((current_world[0] - x1) * seg_dx + (current_world[1] - y1) * seg_dy) / seg_len_sq))
        proj_x, proj_y = x1 + t * seg_dx, y1 + t * seg_dy
        d = math.hypot(current_world[0] - proj_x, current_world[1] - proj_y)
        if d < best_dist:
            best_dist, best_idx, best_t = d, i, t

    # Distance from the start of the route to the robot's projected
    # point: full length of every segment before best_idx, plus the
    # partial distance into segment best_idx itself.
    start_offset = 0.0
    for i in range(best_idx):
        x1, y1 = ideal_route[i]
        x2, y2 = ideal_route[i + 1]
        start_offset += math.hypot(x2 - x1, y2 - y1)
    x1, y1 = ideal_route[best_idx]
    x2, y2 = ideal_route[best_idx + 1]
    seg_len = math.hypot(x2 - x1, y2 - y1)
    start_offset += seg_len * best_t

    CLEARANCE_MARGIN = 1.0
    traveled = 0.0
    found_blocked = False
    while traveled <= max_search_distance:
        point, _ = point_at_distance_along_route(ideal_route, 0, start_offset + traveled)
        pgx, pgy = world_to_grid(point[0], point[1])
        if (pgx, pgy) in runtime_blocked:
            found_blocked = True
            break
        traveled += search_step

    if not found_blocked:
        point, _ = point_at_distance_along_route(ideal_route, 0, start_offset)
        return point

    while traveled <= max_search_distance:
        point, _ = point_at_distance_along_route(ideal_route, 0, start_offset + traveled)
        pgx, pgy = world_to_grid(point[0], point[1])
        if (pgx, pgy) not in runtime_blocked:
            rejoin_point, _ = point_at_distance_along_route(
                ideal_route, 0, start_offset + traveled + CLEARANCE_MARGIN)
            return rejoin_point
        traveled += search_step

    return target_pos

# ---- Live red breadcrumb trail ----
# Drawn incrementally, one short segment per frame of movement, so it
# grows as the robot actually drives -- this is the *actual* driven
# path, separate from (and may differ from) the fixed yellow ideal route.
DRIVEN_TRAIL_COLOR = [1, 0, 0]
last_trail_pos = start_pos  # (x, y) where the trail last left off

# ---- Collision-recovery state machine ----
# DRIVING   : normal driving, tracking a lookahead point on ideal_route
# REVERSING : backing straight away from a just-hit obstacle by a fixed
#             distance, before any replanning is attempted -- this is
#             what prevents the robot from replanning while still
#             touching (or right next to) the obstacle, which is what
#             caused the circling/looping behavior
# REJOINING : driving a short detour around the obstacle back onto the
#             original yellow-line route, rather than a full replan to
#             the goal
STATE_DRIVING = "DRIVING"
STATE_REVERSING = "REVERSING"
STATE_REJOINING = "REJOINING"
STATE_WAITING = "WAITING"   # stopped at the Q-crossing, waiting for a safe gap
state = STATE_DRIVING

REVERSE_DISTANCE = 1.2
REVERSE_SPEED = -8         # faster reverse for the larger layout
reverse_start_pos = None
detour_waypoints = []
detour_index = 0
active_divert_cone_id = None

# Q-learning crossing location: where the robot's route crosses the
# SERVICE ROAD (climbing connector B at x=14, crossing y=8). Unlike
# the main taxiway -- which the robot drives ALONG, so a car there
# shares its lane -- the robot crosses this road perpendicular to its
# traffic, which is what makes it a genuine gap-acceptance decision.
Q_CROSSING_X = 14.0
Q_CROSSING_Y = 8.0
Q_CROSSING_APPROACH = 2.0  # start deciding this far before the road
Q_CROSSING_CLEAR = 1.0     # decision resolves once this far past it
Q_CROSSING_X_TOL = 1.5     # robot must be near the connector's x line

# Q-learning episode tracking
wait_start_time = None
q_current_state = None    # state at the moment the GO decision was made
q_current_action = None   # action chosen (GO or WAIT)
q_wait_cost = 0.0         # accumulated waiting penalty for this crossing
q_go_active = False       # a GO was committed and the robot is mid-crossing;
                          # its reward is resolved when the robot clears the zone
q_go_min_dist = float("inf")  # closest any taxiway car came during that crossing
q_crossing_done = False   # a decision has already been resolved for this
                          # approach -- don't decide again until the robot
                          # leaves the crossing zone (prevents the agent
                          # re-deciding every frame while inside the zone)

# Per-episode stats, reported in a machine-readable RESULT line at the
# end of the run (consumed by evaluate.py).
target_reached = False
episode_collisions = 0
episode_near_misses = 0
episode_go_decisions = 0
episode_wait_decisions = 0

if SIM_MAP:
    start_nav_map()

print("Driving along planned path...")
sim_time = 0.0
map_frame_count = 0

while True:
    if not p.isConnected():
        print("Simulation window was closed -- exiting.")
        break

    # The isConnected() check above can still race a window close that
    # lands mid-frame, so the physics calls are guarded: closing the
    # GUI should end the run tidily (and still write the summary and
    # Q-table below) rather than raising out of the loop.
    try:
        p.stepSimulation()
        if not SIM_HEADLESS:
            update_camera_from_keys()
            time.sleep(1. / 480.)
        sim_time += 1. / 480.

        if SIM_TIME_LIMIT > 0 and sim_time > SIM_TIME_LIMIT:
            print(f"Episode timed out after {SIM_TIME_LIMIT:.0f} "
                  f"simulated seconds.")
            break

        # Advance all traffic cars along the taxiway each frame
        update_cars(1. / 480.)

        pos, orn = p.getBasePositionAndOrientation(robot)
        euler = p.getEulerFromQuaternion(orn)
        heading = euler[2]
    except p.error:
        print("Simulation window was closed -- exiting.")
        break

    # Keep the GO/STOP light riding above the robot. It shows the
    # signal computed by the previous frame's control logic (a 1-frame
    # lag at 480 Hz -- imperceptible).
    update_signal_light(drive_signal, pos)

    # Feed the 2D navigation map window.
    if map_proc is not None:
        map_frame_count += 1
        if map_frame_count % MAP_UPDATE_EVERY == 0:
            _, _, map_remaining = lookahead_point_on_route(
                ideal_route, (pos[0], pos[1]), LOOKAHEAD_DISTANCE)
            vel, _ = p.getBaseVelocity(robot)
            send_to_map({
                "type": "state",
                "t": round(sim_time, 2),
                "robot": [round(pos[0], 3), round(pos[1], 3),
                          round(heading, 3)],
                "signal": drive_signal,
                "state": state,
                "cars": [[round(c["x"], 2), round(c["y"], 2)] for c in cars],
                "remaining": round(map_remaining, 2),
                "speed": round(math.hypot(vel[0], vel[1]), 2),
            })

    # The control logic below recomputes drive_signal for THIS frame,
    # so the settled value from the previous frame is kept for the
    # visual overlays (signal light, map panel, GIF). Without this the
    # capture below would read the freshly-reset "GO" every time and
    # never record a STOP.
    shown_signal = drive_signal

    # Default drive signal for this frame; any branch below that
    # commands the wheels to hold overrides it to STOP.
    drive_signal = "GO"

    # ---- Offscreen GIF capture (placed before any `continue` so
    # frames keep flowing while the robot holds at a crossing) ----
    if SIM_CAPTURE:
        # Driven-path markers stand in for the red debug-line trail.
        if math.hypot(pos[0] - last_capture_marker[0],
                      pos[1] - last_capture_marker[1]) > 0.4:
            disc = p.createVisualShape(p.GEOM_CYLINDER, radius=0.09,
                                       length=0.02, rgbaColor=[1, 0, 0, 1])
            p.createMultiBody(0, -1, disc,
                              basePosition=[pos[0], pos[1], 0.03])
            last_capture_marker = (pos[0], pos[1])
        if sim_time >= next_capture_time:
            next_capture_time += 1.0 / CAPTURE_FPS
            _, _, rgb, _, _ = p.getCameraImage(
                CAPTURE_SIZE[0], CAPTURE_SIZE[1],
                CAPTURE_VIEW, CAPTURE_PROJ,
                renderer=p.ER_TINY_RENDERER)
            # Defensive copy: PyBullet doesn't guarantee ownership of
            # the returned pixel buffer across getCameraImage calls.
            panel_3d = np.reshape(np.array(rgb, dtype=np.uint8, copy=True),
                                  (CAPTURE_SIZE[1], CAPTURE_SIZE[0], 4))[:, :, :3]

            if capture_map is not None:
                _, _, cap_remaining = lookahead_point_on_route(
                    ideal_route, (pos[0], pos[1]), LOOKAHEAD_DISTANCE)
                cap_vel, _ = p.getBaseVelocity(robot)
                panel_map = render_map_panel(
                    pos[0], pos[1], heading, shown_signal, cap_remaining,
                    math.hypot(cap_vel[0], cap_vel[1]), state)
                divider = np.full((CAPTURE_SIZE[1], 3, 3), 210, dtype=np.uint8)
                capture_frames.append(
                    np.hstack([panel_3d, divider, panel_map]))
            else:
                capture_frames.append(panel_3d)

    # ---- Q-learning: gap acceptance at the service-road crossing ----
    # When the robot is climbing connector B and nears the service
    # road (x=14, y=8), it must decide: WAIT or GO? The Q-learning
    # agent makes this decision based on the current gap to the
    # nearest service-road car, using its learned Q-values.
    at_crossing = (state in (STATE_DRIVING, STATE_WAITING) and
                   abs(pos[0] - Q_CROSSING_X) < Q_CROSSING_X_TOL and
                   Q_CROSSING_Y - Q_CROSSING_APPROACH
                   < pos[1] <
                   Q_CROSSING_Y + Q_CROSSING_CLEAR)

    if not at_crossing:
        if q_go_active:
            # The robot has cleared the crossing zone on a committed GO
            # -- resolve the reward now, based on what actually happened
            # while crossing (closest approach of any taxiway car), not
            # on the gap at the moment the decision was made.
            safe = q_go_min_dist >= NEAR_MISS_DISTANCE
            if not safe:
                episode_near_misses += 1
            reward = 10.0 if safe else -10.0
            next_state = q_state(*crossing_gap_and_direction())
            q_update(q_current_state, ACTION_GO, reward, next_state)
            q_total_crossings += 1
            save_q_table(q_table, q_epsilon, q_run_count, q_total_crossings)
            print(f"Q-agent: crossing complete -- closest car "
                  f"{q_go_min_dist:.1f}m -> {'safe' if safe else 'NEAR MISS'}  "
                  f"reward={reward:+.0f}  "
                  f"Q[{state_label(q_current_state)}, GO]="
                  f"{q_table[q_current_state][ACTION_GO]:+.2f}")
            q_go_active = False
        # Out of the zone -- the next approach gets a fresh decision.
        q_crossing_done = False

    if at_crossing:
        if q_go_active:
            # Mid-crossing on a committed GO: keep tracking how close
            # the traffic actually gets, but don't make a new decision.
            q_go_min_dist = min(q_go_min_dist,
                                nearest_crossing_car_distance(pos[0], pos[1]))
        elif q_crossing_done:
            pass  # decision already resolved for this approach -- just drive
        else:
            gap, car_approaching = crossing_gap_and_direction()

            # Only engage the Q-agent when a car is actually close enough
            # to matter. If road is clear beyond the largest gap bucket
            # (9m+), just drive through with no decision needed.
            if gap >= GAP_BUCKETS[-1]:
                if state == STATE_WAITING:
                    state = STATE_DRIVING
                # fall through to normal driving below
            elif state == STATE_DRIVING:
                # Arriving at the crossing -- make ONE GO/WAIT decision
                # for this approach. The q_go_active / q_crossing_done
                # latches stop it being re-made every frame while the
                # robot is still inside the zone.
                current_state = q_state(gap, car_approaching)
                action = choose_action(current_state, q_epsilon)
                q_current_state = current_state
                q_current_action = action
                q_wait_cost = 0.0

                if action == ACTION_WAIT:
                    episode_wait_decisions += 1
                    state = STATE_WAITING
                    wait_start_time = sim_time
                    drive_signal = "STOP"
                    print(f"Q-agent: GAP={gap:.1f}m "
                          f"{'closing' if car_approaching else 'receding'} "
                          f"({state_label(current_state)}) "
                          f"-> WAIT  [ε={q_epsilon:.2f}]")
                    for joint in WHEEL_JOINTS:
                        p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL,
                                                 targetVelocity=0, force=15)
                    continue
                else:
                    # GO -- commit to the crossing. The reward is NOT
                    # applied yet: it resolves when the robot clears the
                    # zone, based on the closest a car actually came.
                    episode_go_decisions += 1
                    q_go_active = True
                    q_go_min_dist = nearest_crossing_car_distance(pos[0], pos[1])
                    print(f"Q-agent: GAP={gap:.1f}m "
                          f"{'closing' if car_approaching else 'receding'} "
                          f"({state_label(current_state)}) "
                          f"-> GO  [ε={q_epsilon:.2f}]")

            elif state == STATE_WAITING:
                # Accumulate waiting cost (small penalty per second)
                q_wait_cost += 1.0 * (1. / 480.)

                # Release the wait using the same predicted-time check
                # as the rule-based crossings: a raw distance threshold
                # is direction-blind, and stepping out on a "7m gap"
                # in front of a fast APPROACHING car is exactly the
                # near-miss the wait was supposed to prevent. (This
                # also releases early behind a car that just passed
                # and is receding -- gap tiny, but perfectly safe.)
                eta = car_time_to_crossing(Q_CROSSING_Y, Q_CROSSING_X)
                if eta >= MIN_CROSSING_TIME:
                    reward = 10.0 - q_wait_cost
                    next_state = q_state(*crossing_gap_and_direction())
                    q_update(q_current_state, ACTION_WAIT, reward, next_state)
                    q_total_crossings += 1
                    save_q_table(q_table, q_epsilon, q_run_count, q_total_crossings)
                    print(f"Q-agent: road clear (next car {eta:.1f}s out) "
                          f"after {sim_time - wait_start_time:.1f}s wait -- "
                          f"crossing. reward={reward:+.1f}  "
                          f"Q[WAIT]={q_table[q_current_state][ACTION_WAIT]:+.2f}")
                    state = STATE_DRIVING
                    # The wait decision is resolved -- cross now without
                    # re-deciding on every remaining frame in the zone.
                    q_crossing_done = True
                else:
                    drive_signal = "STOP"
                    for joint in WHEEL_JOINTS:
                        p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL,
                                                 targetVelocity=0, force=15)
                    continue

    # ---- Rule-based crossing safety at road crossings ----
    # Only check the single next upcoming road -- whichever road's
    # y-coordinate is immediately above the robot's current y.
    # Each road has a tight hold window that ends BEFORE the car's
    # lane: [road.y - STOP_MARGIN, road.y - COMMIT_MARGIN]. Once the
    # robot is closer than COMMIT_MARGIN it is committed to the
    # crossing and never told to stop in the lane. Windows don't
    # overlap, so the robot can never be blocked by two roads at once.
    # This guard also runs while REJOINING: a collision detour crosses
    # the same roads as normal driving, and skipping the check there
    # let cars hit the robot mid-detour. In that state the service
    # road (normally the Q-agent's crossing) is guarded too.
    if state in (STATE_DRIVING, STATE_REJOINING):
        blocked_by_car = False
        roads_to_check = (CROSSING_ROADS if state == STATE_DRIVING
                          else REJOIN_CROSSING_ROADS)
        for road in roads_to_check:
            in_zone = (road["y"] - CROSSING_STOP_MARGIN
                       < pos[1]
                       < road["y"] - CROSSING_COMMIT_MARGIN)
            if in_zone:
                eta = car_time_to_crossing(road["y"], CROSSING_X)
                if eta < MIN_CROSSING_TIME:
                    blocked_by_car = True
                    drive_signal = "STOP"
                    for joint in WHEEL_JOINTS:
                        p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL,
                                                 targetVelocity=0, force=15)
                break  # only ever check the first road we're in range of
        if blocked_by_car:
            continue


    if math.hypot(pos[0] - last_trail_pos[0], pos[1] - last_trail_pos[1]) > 0.03:
        p.addUserDebugLine(
            [last_trail_pos[0], last_trail_pos[1], 0.07],
            [pos[0], pos[1], 0.07],
            lineColorRGB=DRIVEN_TRAIL_COLOR,
            lineWidth=3
        )
        last_trail_pos = (pos[0], pos[1])

    # ---- Live collision detection ----
    # Acts on a fresh collision while DRIVING (normal driving) or
    # REJOINING (actively driving a detour) -- if the detour itself
    # ends up clipping the cone it was supposed to avoid, that's a
    # real problem worth reacting to, not something to silently keep
    # driving through. During REVERSING we ignore further contacts
    # with the same cone (brief re-contact while backing away is
    # expected) so we don't immediately re-enter collision-handling
    # mid-maneuver.
    collided_cone_pos = None
    collided_cone_id = None
    for cone_id in cone_body_ids:
        contacts = p.getContactPoints(bodyA=robot, bodyB=cone_id)
        if contacts:
            last_hit = last_collision_time.get(cone_id, -999)
            if sim_time - last_hit > COLLISION_COOLDOWN:
                cone_pos, _ = p.getBasePositionAndOrientation(cone_id)
                hit_point = (round(cone_pos[0], 2), round(cone_pos[1], 2))

                if hit_point not in learned_obstacles and hit_point not in newly_learned:
                    newly_learned.append(hit_point)
                    print(f"Collision detected at {hit_point} -- "
                          f"remembering this location for future runs.")
                    save_learned_obstacles(learned_obstacles + newly_learned)
                last_collision_time[cone_id] = sim_time

                can_react = state == STATE_DRIVING or state == STATE_REJOINING
                if can_react and cone_id not in diverted_cone_ids:
                    diverted_cone_ids.add(cone_id)
                    block_cells_around(cone_pos[0], cone_pos[1])
                    collided_cone_pos = (cone_pos[0], cone_pos[1])
                    collided_cone_id = cone_id

    # ---- Transient obstacle collisions (cars, bay walls) ----
    # Same reverse-and-rejoin reaction as a cone hit, but nothing is
    # learned or added to the blocked map (see transient_obstacle_ids
    # above). After reversing, find_rejoin_target/A* simply steer the
    # robot back onto the ideal route.
    transient_hit = False
    if collided_cone_pos is None and state in (STATE_DRIVING, STATE_REJOINING):
        for body_id in transient_obstacle_ids:
            contacts = p.getContactPoints(bodyA=robot, bodyB=body_id)
            if contacts:
                last_hit = last_collision_time.get(body_id, -999)
                if sim_time - last_hit > COLLISION_COOLDOWN:
                    last_collision_time[body_id] = sim_time
                    transient_hit = True
                    break

    if collided_cone_pos is not None or transient_hit:
        episode_collisions += 1
        drive_signal = "STOP"
        # Stop immediately, then switch to REVERSING -- back straight
        # away from the obstacle before any replanning is attempted.
        for joint in WHEEL_JOINTS:
            p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=15)
        if transient_hit:
            print(f"Collision with a vehicle or wall at x={pos[0]:.1f}, "
                  f"y={pos[1]:.1f}! Stopping and reversing away...")
        else:
            print("Collision! Stopping and reversing away from obstacle...")
        # If this collision interrupted an in-progress detour around a
        # DIFFERENT cone, clear that previous cone's diverted flag now
        # -- otherwise it would stay stuck in diverted_cone_ids forever
        # since its own detour never got to finish normally.
        if active_divert_cone_id is not None and active_divert_cone_id != collided_cone_id:
            diverted_cone_ids.discard(active_divert_cone_id)
        state = STATE_REVERSING
        reverse_start_pos = (pos[0], pos[1])
        active_divert_cone_id = collided_cone_id
        continue

    if state == STATE_REVERSING:
        traveled = math.hypot(pos[0] - reverse_start_pos[0], pos[1] - reverse_start_pos[1])
        if traveled < REVERSE_DISTANCE:
            for joint in WHEEL_JOINTS:
                p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL,
                                         targetVelocity=REVERSE_SPEED, force=15)
            continue
        else:
            # Far enough away now -- stop, then plan a short detour
            # around the obstacle that rejoins the original yellow
            # route, rather than a full replan straight to the goal.
            for joint in WHEEL_JOINTS:
                p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=15)
            print("Clear of obstacle -- planning detour back onto original route...")

            rejoin_point = find_rejoin_target((pos[0], pos[1]))
            new_detour = find_path_with_runtime_blocked((pos[0], pos[1]), rejoin_point)

            if new_detour is None:
                print("No detour found around this obstacle! Stopping.")
                break

            detour_waypoints = new_detour
            detour_index = 0
            state = STATE_REJOINING
            continue

    if state == STATE_REJOINING:
        if detour_index >= len(detour_waypoints):
            # Detour complete -- resume normal driving. No index to
            # restore: lookahead_point_on_route will automatically
            # find wherever the robot now is on ideal_route and
            # continue from there.
            #
            # Clear this cone's "already diverted" flag now that the
            # maneuver has actually finished -- if the robot ends up
            # touching it again later, that's a genuine new collision
            # (e.g. the detour grazed it, or the robot drifted back
            # into it), not a stale leftover from this same maneuver,
            # so it should be free to trigger another reverse-and-detour.
            if active_divert_cone_id is not None:
                diverted_cone_ids.discard(active_divert_cone_id)
                active_divert_cone_id = None
            state = STATE_DRIVING
            continue

        wx, wy = detour_waypoints[detour_index]
        dx = wx - pos[0]
        dy = wy - pos[1]
        distance = math.hypot(dx, dy)

        if distance < ARRIVAL_RADIUS:
            detour_index += 1
            continue

        desired_heading = math.atan2(dy, dx)
        heading_error = desired_heading - heading
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

        turn_severity = min(abs(heading_error) / TURN_SEVERITY_ANGLE, 1.0)
        base_speed = MAX_SPEED * (1.0 - (1.0 - TURN_SPEED_FLOOR) * turn_severity)
        turn_gain = TURN_GAIN

        left_speed = base_speed - turn_gain * heading_error
        right_speed = base_speed + turn_gain * heading_error

        p.setJointMotorControl2(robot, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=WHEEL_FORCE)
        p.setJointMotorControl2(robot, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=WHEEL_FORCE)
        p.setJointMotorControl2(robot, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=WHEEL_FORCE)
        p.setJointMotorControl2(robot, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=WHEEL_FORCE)
        continue

    # ---- STATE_DRIVING: follow a lookahead point on the ideal_route ----
    # Rather than driving toward discrete waypoints one at a time, find
    # the point on ideal_route nearest the robot's current position,
    # then aim for a point a fixed LOOKAHEAD_DISTANCE further along that
    # same line. This is what lets the robot track the literal road
    # centerline closely (since it's always correcting toward the
    # actual line, not just toward sparse corners), and it also makes
    # "resume normal driving after a detour" trivial -- there's no
    # index to resume from, the robot just naturally finds wherever it
    # currently is on the line and continues from there.
    target_x, target_y, distance_remaining = lookahead_point_on_route(
        ideal_route, (pos[0], pos[1]), LOOKAHEAD_DISTANCE)

    if distance_remaining < ARRIVAL_RADIUS:
        target_reached = True
        print("Target reached!")
        break

    wx, wy = target_x, target_y
    dx = wx - pos[0]
    dy = wy - pos[1]

    desired_heading = math.atan2(dy, dx)
    heading_error = desired_heading - heading
    # normalize to [-pi, pi]
    heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))

    # Slow down when a sharp turn is needed, so the robot turns
    # precisely instead of swinging wide and clipping obstacles.
    # heading_error of 0 = full speed, heading_error near +-pi/2 or
    # more = mostly just rotating in place.
    turn_severity = min(abs(heading_error) / TURN_SEVERITY_ANGLE, 1.0)
    base_speed = MAX_SPEED * (1.0 - (1.0 - TURN_SPEED_FLOOR) * turn_severity)
    turn_gain = TURN_GAIN

    left_speed = base_speed - turn_gain * heading_error
    right_speed = base_speed + turn_gain * heading_error

    p.setJointMotorControl2(robot, 2, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=WHEEL_FORCE)
    p.setJointMotorControl2(robot, 4, p.VELOCITY_CONTROL, targetVelocity=left_speed, force=WHEEL_FORCE)
    p.setJointMotorControl2(robot, 3, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=WHEEL_FORCE)
    p.setJointMotorControl2(robot, 5, p.VELOCITY_CONTROL, targetVelocity=right_speed, force=WHEEL_FORCE)

if newly_learned:
    print(f"Learned {len(newly_learned)} new obstacle location(s) this run. "
          f"They will be avoided automatically next time you run this script.")
else:
    print("No new collisions this run.")

# Decay epsilon for next run -- the agent explores less as it gains experience
new_epsilon = max(Q_EPSILON_MIN, q_epsilon * Q_EPSILON_DECAY)
save_q_table(q_table, new_epsilon, q_run_count, q_total_crossings)
print(f"\nQ-learning summary for run #{q_run_count}:")
print(f"  Epsilon: {q_epsilon:.2f} -> {new_epsilon:.2f} (less exploration next run)")
print(f"  Final Q-table (WAIT | GO) per state (gap bucket x car direction):")
for s in range(len(q_table)):
    best = "GO" if q_table[s][ACTION_GO] >= q_table[s][ACTION_WAIT] else "WAIT"
    print(f"    {state_label(s):>10}: WAIT={q_table[s][ACTION_WAIT]:+.2f}  "
          f"GO={q_table[s][ACTION_GO]:+.2f}  -> prefers {best}")

if SIM_CAPTURE and capture_frames:
    import numpy as np
    from PIL import Image
    print(f"Writing {len(capture_frames)} frames to {SIM_CAPTURE} ...")
    # Downscale, then quantise every frame to ONE shared palette. The
    # side-by-side composite is ~1.7MB at full size, too heavy for the
    # top of a README. The shared palette matters: quantising frames
    # independently gives each its own colour table and defeats the
    # GIF's inter-frame compression (measured 5.2MB that way, vs a few
    # hundred KB here).
    def _prep(frame):
        img = Image.fromarray(frame)
        return img.resize((int(img.width * CAPTURE_SCALE),
                           int(img.height * CAPTURE_SCALE)), Image.LANCZOS)

    # Derive the shared palette from frames sampled across the whole
    # run, not a single one: a colour absent from the sample frame
    # (a car parked off-screen at that instant) gets approximated to
    # the nearest available one for the entire GIF -- which is what
    # turned the blue car teal.
    step = max(1, len(capture_frames) // 12)
    palette = Image.fromarray(
        np.vstack(capture_frames[::step])).quantize(colors=CAPTURE_COLORS)
    pil_frames = [_prep(frame).quantize(palette=palette, dither=Image.NONE)
                  for frame in capture_frames]
    out_dir = os.path.dirname(os.path.abspath(SIM_CAPTURE))
    os.makedirs(out_dir, exist_ok=True)
    pil_frames[0].save(SIM_CAPTURE, save_all=True,
                       append_images=pil_frames[1:],
                       duration=int(1000 / (CAPTURE_FPS * CAPTURE_SPEEDUP)),
                       loop=0, optimize=True)
    print(f"GIF saved: {SIM_CAPTURE}")

# Machine-readable episode summary, parsed by evaluate.py. epsilon is
# the exploration rate that was USED this run (before decay).
print(f"RESULT success={int(target_reached)} sim_time={sim_time:.2f} "
      f"collisions={episode_collisions} near_misses={episode_near_misses} "
      f"go={episode_go_decisions} wait={episode_wait_decisions} "
      f"epsilon={q_epsilon:.4f} run={q_run_count}")

# Closing the map's stdin tells it the run is over; it closes its
# window and exits on its own.
if map_proc is not None:
    try:
        map_proc.stdin.close()
    except OSError:
        pass

if p.isConnected():
    p.disconnect()
