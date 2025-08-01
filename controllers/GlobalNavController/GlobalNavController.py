from controller import Robot, Camera, InertialUnit
import math
import random

# --- Parameters ---
TIME_STEP = 32
MAP_WIDTH = 20    # number of cells
MAP_HEIGHT = 20
CELL_SIZE = 0.1   # meters per cell
OBSTACLE_THRESHOLD = 1

# Example: Hardcoded occupancy grid (0: free, 1: obstacle)
occupancy_grid = [[0]*MAP_WIDTH for _ in range(MAP_HEIGHT)]
# Example obstacle in the middle
for i in range(8,12):
    for j in range(8,12):
        occupancy_grid[j][i] = 1

# --- RRT* Node Definition ---
class Vertex:
    def __init__(self, x, y, cost, parent):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

def is_obstacle(x, y):
    grid_x = int(x / CELL_SIZE)
    grid_y = int(y / CELL_SIZE)
    if 0 <= grid_x < MAP_WIDTH and 0 <= grid_y < MAP_HEIGHT:
        return occupancy_grid[grid_y][grid_x] >= OBSTACLE_THRESHOLD
    return True  # Out of bounds is obstacle

def cost_between_points(x1, y1, x2, y2):
    steps = int(math.hypot(x1 - x2, y1 - y2) / 0.05)
    for i in range(steps+1):
        xi = x2 + (x1 - x2) * i / steps
        yi = y2 + (y1 - y2) * i / steps
        if is_obstacle(xi, yi):
            return float('inf'), True
    return math.hypot(x1 - x2, y1 - y2), False

def rrt_star(start, goal, max_iter=1000, step_size=0.2, k_neigh=10):
    vertices = [Vertex(start[0], start[1], 0, -1)]
    for _ in range(max_iter):
        rx = random.uniform(0, MAP_WIDTH*CELL_SIZE)
        ry = random.uniform(0, MAP_HEIGHT*CELL_SIZE)
        if is_obstacle(rx, ry):
            continue
        # Find nearest
        dists = [math.hypot(rx-v.x, ry-v.y) for v in vertices]
        nearest_idx = dists.index(min(dists))
        nearest = vertices[nearest_idx]
        # Steer
        dist = math.hypot(rx - nearest.x, ry - nearest.y)
        if dist > step_size:
            t = step_size / dist
            rx = nearest.x + t * (rx - nearest.x)
            ry = nearest.y + t * (ry - nearest.y)
        cost, hit = cost_between_points(rx, ry, nearest.x, nearest.y)
        if hit:
            continue
        new_cost = nearest.cost + cost
        new_vertex = Vertex(rx, ry, new_cost, nearest_idx)
        vertices.append(new_vertex)
        # Rewire
        for i, v in enumerate(vertices[:-1]):
            if math.hypot(rx - v.x, ry - v.y) < step_size:
                cost, hit = cost_between_points(rx, ry, v.x, v.y)
                if not hit and v.cost > new_cost + cost:
                    v.cost = new_cost + cost
                    v.parent = len(vertices)-1
        # Check if goal reached
        if math.hypot(rx - goal[0], ry - goal[1]) < step_size:
            goal_vertex = Vertex(goal[0], goal[1], new_cost + math.hypot(rx - goal[0], ry - goal[1]), len(vertices)-1)
            vertices.append(goal_vertex)
            # Extract path
            path = []
            idx = len(vertices)-1
            while idx != -1:
                v = vertices[idx]
                path.append((v.x, v.y))
                idx = v.parent
            path.reverse()
            return path
    return None

def get_random_goal(occupancy_grid):
    while True:
        x = random.randint(0, MAP_WIDTH-1)
        y = random.randint(0, MAP_HEIGHT-1)
        if occupancy_grid[y][x] == 0:
            return (x * CELL_SIZE, y * CELL_SIZE)

# --- Webots Controller Main Loop ---
robot = Robot()
timestep = int(robot.getBasicTimeStep())

# Devices
camera = robot.getCamera("camera")
camera.enable(timestep)

imu = robot.getDevice("imu")
if imu is not None:
    imu.enable(timestep)
else:
    print("IMU not found!")

# For demonstration, start at (0.2, 0.2)
start = (0.2, 0.2)
goal = get_random_goal(occupancy_grid)
print("Random goal chosen at:", goal)

# Plan path
path = rrt_star(start, goal)
if path:
    print("Planned path:", path)
else:
    print("No path found!")

# --- Main Loop ---
while robot.step(timestep) != -1:
    # Read camera image (raw bytes, can be processed with OpenCV if desired)
    img = camera.getImage()
    # Example: print image size
    print("Camera image size:", camera.getWidth(), "x", camera.getHeight())

    # Read IMU (roll, pitch, yaw)
    #orientation = imu.getRollPitchYaw()
    #print("IMU orientation (roll, pitch, yaw):", orientation)
    if cv2.waitKey(1)==ord('q'):
            break
   