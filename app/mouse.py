from collections import deque
import math

# Global dictionary to store game states
game_states = {}

class MouseState:
    def __init__(self, game_uuid):
        self.game_uuid = game_uuid
        self.x = 8.0  # cm from origin
        self.y = 8.0
        self.orientation = 0.0  # degrees, 0 for North
        self.momentum = 0
        self.map = [[{'n': None, 's': None, 'e': None, 'w': None} for _ in range(16)] for _ in range(16)]
        self.visited = set()
        self.goal_found = False
        self.status = 'exploring'
        self.best_path = []
        self.current_index = 0
        
        # Initialize outer walls
        for i in range(16):
            for j in range(16):
                if i == 0:
                    self.map[i][j]['w'] = True
                if i == 15:
                    self.map[i][j]['e'] = True
                if j == 0:
                    self.map[i][j]['s'] = True
                if j == 15:
                    self.map[i][j]['n'] = True
        
        self.visited.add((0,0))

def is_goal_cell(i, j):
    return (7 <= i <= 8) and (7 <= j <= 8)

def get_neighbor(i, j, direction):
    if direction == 'n':
        return (i, j+1)
    elif direction == 's':
        return (i, j-1)
    elif direction == 'e':
        return (i+1, j)
    elif direction == 'w':
        return (i-1, j)
    return (i, j)

def get_direction(from_cell, to_cell):
    i1, j1 = from_cell
    i2, j2 = to_cell
    if i2 == i1 + 1:
        return 'e'
    elif i2 == i1 - 1:
        return 'w'
    elif j2 == j1 + 1:
        return 'n'
    elif j2 == j1 - 1:
        return 's'
    return None

def get_orientation_from_direction(direction):
    if direction == 'n':
        return 0
    elif direction == 'e':
        return 90
    elif direction == 's':
        return 180
    elif direction == 'w':
        return 270
    return 0

def get_opposite_direction(orientation):
    if orientation == 0:
        return 's'
    elif orientation == 90:
        return 'w'
    elif orientation == 180:
        return 'n'
    elif orientation == 270:
        return 'e'
    return 's'

def turn_to(target_orientation, current_orientation):
    diff = (target_orientation - current_orientation) % 360
    if diff > 180:
        diff -= 360
    if diff > 0:
        return ['R'] * int(diff / 45)
    else:
        return ['L'] * int(-diff / 45)

def move_forward(x, y, orientation, distance):
    rad = math.radians(orientation)
    x_new = x + distance * math.sin(rad)
    y_new = y + distance * math.cos(rad)
    return x_new, y_new

def BFS(map, start, goal):
    parent = {}
    queue = deque([start])
    parent[start] = None
    while queue:
        cell = queue.popleft()
        if cell == goal:
            break
        i, j = cell
        for direction in ['n','e','s','w']:
            if map[i][j][direction] is False:
                ni, nj = get_neighbor(i, j, direction)
                if 0 <= ni < 16 and 0 <= nj < 16 and (ni, nj) not in parent:
                    parent[(ni, nj)] = (i, j)
                    queue.append((ni, nj))
    if goal not in parent:
        return None
    path = []
    cell = goal
    while cell is not None:
        path.append(cell)
        cell = parent[cell]
    path.reverse()
    return path

def plan_best_run(state):
    i = int(state.x // 16)
    j = int(state.y // 16)
    current_cell = (i, j)
    goal_cells = [(7,7), (7,8), (8,7), (8,8)]
    best_path = None
    for gc in goal_cells:
        path = BFS(state.map, current_cell, gc)
        if path is not None and (best_path is None or len(path) < len(best_path)):
            best_path = path
    if best_path is None:
        return []
    return best_path

def update_map_from_sensors(state, sensor_data):
    i = int(state.x // 16)
    j = int(state.y // 16)
    if state.orientation == 0:  # North
        if sensor_data[2] < 12:
            state.map[i][j]['n'] = True
        else:
            state.map[i][j]['n'] = False
        if sensor_data[0] < 12:
            state.map[i][j]['w'] = True
        else:
            state.map[i][j]['w'] = False
        if sensor_data[4] < 12:
            state.map[i][j]['e'] = True
        else:
            state.map[i][j]['e'] = False
    elif state.orientation == 90:  # East
        if sensor_data[2] < 12:
            state.map[i][j]['e'] = True
        else:
            state.map[i][j]['e'] = False
        if sensor_data[0] < 12:
            state.map[i][j]['s'] = True
        else:
            state.map[i][j]['s'] = False
        if sensor_data[4] < 12:
            state.map[i][j]['n'] = True
        else:
            state.map[i][j]['n'] = False
    elif state.orientation == 180:  # South
        if sensor_data[2] < 12:
            state.map[i][j]['s'] = True
        else:
            state.map[i][j]['s'] = False
        if sensor_data[0] < 12:
            state.map[i][j]['e'] = True
        else:
            state.map[i][j]['e'] = False
        if sensor_data[4] < 12:
            state.map[i][j]['w'] = True
        else:
            state.map[i][j]['w'] = False
    elif state.orientation == 270:  # West
        if sensor_data[2] < 12:
            state.map[i][j]['w'] = True
        else:
            state.map[i][j]['w'] = False
        if sensor_data[0] < 12:
            state.map[i][j]['n'] = True
        else:
            state.map[i][j]['n'] = False
        if sensor_data[4] < 12:
            state.map[i][j]['s'] = True
        else:
            state.map[i][j]['s'] = False

def choose_direction(state, i, j):
    for direction in ['n','e','s','w']:
        if state.map[i][j][direction] is False:
            ni, nj = get_neighbor(i, j, direction)
            if (ni, nj) not in state.visited:
                return direction
    return None

def process_micromouse_request(data):
    """
    Process a micromouse request and return instructions
    """
    game_uuid = data['game_uuid']
    sensor_data = data['sensor_data']
    total_time_ms = data['total_time_ms']
    goal_reached = data['goal_reached']
    best_time_ms = data['best_time_ms']
    run_time_ms = data['run_time_ms']
    run = data['run']
    momentum = data['momentum']
    
    if game_uuid not in game_states:
        game_states[game_uuid] = MouseState(game_uuid)
    state = game_states[game_uuid]
    
    # Update state from request data for verification
    state.momentum = momentum
    # Note: we trust our own position simulation
    
    # Check if we are at cell center (approximately)
    i = int(state.x // 16)
    j = int(state.y // 16)
    at_center = abs(state.x - (i*16+8)) < 0.1 and abs(state.y - (j*16+8)) < 0.1 and state.momentum == 0
    if not at_center:
        # In case of discrepancy, adjust state to cell center? But we assume always at center after instructions.
        state.x = i*16 + 8
        state.y = j*16 + 8
        state.momentum = 0
    
    if state.status == 'exploring':
        update_map_from_sensors(state, sensor_data)
        i = int(state.x // 16)
        j = int(state.y // 16)
        state.visited.add((i,j))
        
        if is_goal_cell(i, j):
            state.goal_found = True
            
        opposite_dir = get_opposite_direction(state.orientation)
        if state.map[i][j][opposite_dir] is None:
            target_orientation = get_orientation_from_direction(opposite_dir)
            instructions = turn_to(target_orientation, state.orientation)
            for inst in instructions:
                if inst == 'R':
                    state.orientation = (state.orientation + 45) % 360
                elif inst == 'L':
                    state.orientation = (state.orientation - 45) % 360
            return {"instructions": instructions, "end": False}
        else:
            direction = choose_direction(state, i, j)
            if direction is None:
                if state.goal_found:
                    state.best_path = plan_best_run(state)
                    if not state.best_path:
                        return {"instructions": [], "end": True}
                    state.status = 'best_run'
                    state.current_index = 0
                    return {"instructions": [], "end": False}  # Will move in next request
                else:
                    return {"instructions": [], "end": True}
            else:
                target_orientation = get_orientation_from_direction(direction)
                if state.orientation != target_orientation:
                    instructions = turn_to(target_orientation, state.orientation)
                    for inst in instructions:
                        if inst == 'R':
                            state.orientation = (state.orientation + 45) % 360
                        elif inst == 'L':
                            state.orientation = (state.orientation - 45) % 360
                    return {"instructions": instructions, "end": False}
                else:
                    instructions = ['F2', 'F0']
                    # Simulate move
                    state.x, state.y = move_forward(state.x, state.y, state.orientation, 8.0)
                    state.momentum = 1
                    state.x, state.y = move_forward(state.x, state.y, state.orientation, 8.0)
                    state.momentum = 0
                    return {"instructions": instructions, "end": False}
                    
    elif state.status == 'best_run':
        if state.current_index >= len(state.best_path) - 1:
            return {"instructions": [], "end": True}
        current_cell = state.best_path[state.current_index]
        next_cell = state.best_path[state.current_index+1]
        direction = get_direction(current_cell, next_cell)
        target_orientation = get_orientation_from_direction(direction)
        if state.orientation != target_orientation:
            instructions = turn_to(target_orientation, state.orientation)
            for inst in instructions:
                if inst == 'R':
                    state.orientation = (state.orientation + 45) % 360
                elif inst == 'L':
                    state.orientation = (state.orientation - 45) % 360
            return {"instructions": instructions, "end": False}
        else:
            instructions = ['F2', 'F0']
            state.x, state.y = move_forward(state.x, state.y, state.orientation, 8.0)
            state.momentum = 1
            state.x, state.y = move_forward(state.x, state.y, state.orientation, 8.0)
            state.momentum = 0
            state.current_index += 1
            if state.current_index >= len(state.best_path) - 1:
                return {"instructions": instructions, "end": True}
            else:
                return {"instructions": instructions, "end": False}
    
    return {"instructions": [], "end": True}
