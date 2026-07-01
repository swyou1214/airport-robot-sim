import pybullet as p
import pybullet_data
import time
import math
import heapq
import json
import os

# ---------------------------------------------------------
# 1. SET UP THE WORLD
# ---------------------------------------------------------
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
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

# Road network: airport taxiway layout
# - Main taxiway: full-width straight road at y=0 (depot to far end)
# - Service road:  parallel road at y=8 (connects cross-connectors)
# - Three cross-connectors at x=6, x=14, x=22 linking the two roads
# - Apron: enclosed box area above the service road (y=12 to y=18)
# - Gate connectors: short spurs from apron bottom to service road
#   at Gate A (x=8), Gate B (x=14), Gate C (x=20)
ROAD_SEGMENTS = [
    # Main taxiway (full width, bottom)
    ((0, 0),  (28, 0)),
    # Service road (full width, above main taxiway)
    ((0, 8),  (28, 8)),
    # Cross-connectors (link taxiway to service road)
    ((6,  0), (6,  8)),    # connector A (left)
    ((14, 0), (14, 8)),    # connector B (centre)
    ((22, 0), (22, 8)),    # connector C (right)
    # Apron perimeter (top, bottom, left, right walls)
    ((4,  12), (24, 12)),  # apron front
    ((4,  18), (24, 18)),  # apron back
    ((4,  12), (4,  18)),  # apron left wall
    ((24, 12), (24, 18)),  # apron right wall
    # Gate connectors (apron bottom -> service road)
    ((8,  8),  (8,  12)),  # Gate A connector
    ((14, 8),  (14, 12)),  # Gate B connector
    ((20, 8),  (20, 12)),  # Gate C connector
    # Apron interior taxilines (allow robot to reach gate positions inside)
    ((8,  12), (8,  18)),   # Gate A internal
    ((14, 12), (14, 18)),   # Gate B internal (spine)
    ((20, 12), (20, 18)),   # Gate C internal
    ((4,  15), (24, 15)),   # apron midline linking all gates
]

for (start, end) in ROAD_SEGMENTS:
    draw_road_segment(start[0], start[1], end[0], end[1])

# Free-roam camera state (controlled with arrow keys)
camera_yaw = 50
camera_pitch = -40
camera_distance = 20
camera_target = [14, 8, 0]

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
for _ in range(120):  # 0.5s at 240Hz -- plenty for a rigid body to settle
    p.stepSimulation()

# Obstacle positions for the expanded airport layout.
# Mix of DELIBERATE (force interesting detours) and RANDOM (scattered):
#
# Deliberate -- block the most direct path so the robot has to make
# smart routing decisions:
#   - Two cones on the main taxiway near connector B (forces use of
#     connector A or C instead, or detour around them)
#   - One cone inside connector B (blocks the direct route to Gate B)
#   - One cone on the service road approaching Gate A
#
# Random -- scattered to keep the robot alert anywhere on the network:
#   - A few cones at various points on the taxiway and service road
obstacle_positions = [
    # --- Deliberate blockers ---
    (12, 0.5),    # taxiway, just before connector B (left side)
    (14, -0.5),   # taxiway, at connector B (right side)
    (14, 4.5),    # inside connector B (halfway up)
    (9, 8.0),     # service road, between connector A and Gate A
    # --- Random scatter ---
    (4, 0.6),     # taxiway, early stretch
    (20, -0.5),   # taxiway, near connector C
    (6, 5.5),     # inside connector A
    (22, 4.0),    # inside connector C
    (16, 8.5),    # service road, between Gate B and connector C
    (8, 13.5),    # Gate A connector, halfway up
]

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

start_pos = (0, 0)       # depot at the far left of the main taxiway
target_pos = (14, 15)    # Gate B, centre of the apron

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
# GATE MARKER: a simple post-and-sign, like an airport gate
# sign (e.g. "GATE B12"), instead of a plain floating sphere
# ---------------------------------------------------------
gate_post_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.05, 0.05, 0.9],
    rgbaColor=[0.2, 0.2, 0.2, 1]
)
p.createMultiBody(0, -1, gate_post_visual,
                   basePosition=[target_pos[0], target_pos[1], 0.9])

gate_sign_visual = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.5, 0.05, 0.3],
    rgbaColor=[0.1, 0.55, 0.25, 1]
)
p.createMultiBody(0, -1, gate_sign_visual,
                   basePosition=[target_pos[0], target_pos[1], 1.6])

# Small ground marker so the arrival point is still clearly visible
gate_floor_marker = p.createVisualShape(
    p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.01],
    rgbaColor=[0.1, 0.55, 0.25, 0.6]
)
p.createMultiBody(0, -1, gate_floor_marker,
                   basePosition=[target_pos[0], target_pos[1], 0.015])

# ---------------------------------------------------------
# 2. BUILD A GRID MAP FOR A*
# ---------------------------------------------------------
GRID_SIZE = 0.5          # each grid cell is 0.5m x 0.5m
GRID_W, GRID_H = 66, 46  # grid covers the expanded airport layout
ORIGIN_X, ORIGIN_Y = -2, -3  # world coords of grid cell (0,0)

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
# The ideal route is an explicit path through the network --
# since ROAD_SEGMENTS is now a full network (not a single chain),
# we can't auto-derive the route as just the list of segment endpoints.
# This traces: depot -> taxiway -> connector B -> service road ->
# Gate B connector -> apron -> Gate B
ideal_route = [
    (0,  0),    # depot (start)
    (14, 0),    # along taxiway to connector B base
    (14, 8),    # up connector B to service road
    (14, 12),   # up Gate B connector to apron front
    (14, 15),   # Gate B (target, centre of apron)
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
MAX_SPEED = 18.0         # much faster for the larger airport layout
TURN_SPEED_FLOOR = 0.15  # minimum speed fraction during sharp turns
TURN_GAIN = 8            # slightly stronger steering to keep up with speed
TURN_SEVERITY_ANGLE = math.pi / 3.2
WHEEL_FORCE = 40          # higher torque to actually reach the new speed

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

    # Two-phase forward search along the route from the robot's
    # current position:
    #   Phase 1: walk forward until we either (a) hit a blocked
    #            sample, meaning there's an obstacle somewhere ahead
    #            that needs to be gotten past, or (b) reach
    #            max_search_distance with nothing blocked at all, in
    #            which case the robot's current position is already
    #            fine as-is.
    #   Phase 2: once a blocked sample is found, keep walking forward
    #            until the route becomes clear again, then continue an
    #            additional CLEARANCE_MARGIN past that point -- this
    #            guarantees the returned rejoin point is genuinely past
    #            the ENTIRE obstacle, not just the first technically-
    #            unblocked point (which, right after reversing, is
    #            often still close enough to the obstacle's edge that
    #            resuming normal driving would walk straight back into
    #            it -- this was causing the robot to hit the same cone
    #            repeatedly before finally clearing it).
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
        # Nothing blocking anywhere in the scanned range -- the
        # robot's current position is already fine.
        point, _ = point_at_distance_along_route(ideal_route, 0, start_offset)
        return point

    # Phase 2: continue forward from where blocking was first found,
    # until clear again, then add the clearance margin.
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
state = STATE_DRIVING

REVERSE_DISTANCE = 1.2     # meters to back away before diverting
REVERSE_SPEED = -8         # negative = reverse
reverse_start_pos = None
detour_waypoints = []
detour_index = 0
active_divert_cone_id = None  # which cone the current REVERSING/REJOINING
                               # maneuver is for, so it can be cleared
                               # from diverted_cone_ids once the detour
                               # actually succeeds -- allowing a genuine
                               # second collision with the SAME cone
                               # (meaning the first detour didn't really
                               # clear it) to trigger a fresh response,
                               # instead of being silently ignored.

print("Driving along planned path...")
sim_time = 0.0

while True:
    if not p.isConnected():
        print("Simulation window was closed -- exiting.")
        break

    p.stepSimulation()
    update_camera_from_keys()
    time.sleep(1. / 480.)
    sim_time += 1. / 240.

    pos, orn = p.getBasePositionAndOrientation(robot)
    euler = p.getEulerFromQuaternion(orn)
    heading = euler[2]

    # Extend the red breadcrumb trail to the robot's current position.
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

    if collided_cone_pos is not None:
        # Stop immediately, then switch to REVERSING -- back straight
        # away from the obstacle before any replanning is attempted.
        for joint in WHEEL_JOINTS:
            p.setJointMotorControl2(robot, joint, p.VELOCITY_CONTROL, targetVelocity=0, force=15)
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

if p.isConnected():
    p.disconnect()
