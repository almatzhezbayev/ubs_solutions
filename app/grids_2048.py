import random

def process_2048_move(grid, direction):
    """
    Process a 2048 move with advanced tile types
    """
    if not grid or not isinstance(grid, list):
        return grid, None
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Convert all cells into clean types
    next_grid = [[convert_value(cell) for cell in row] for row in grid]
    moved = False
    
    if direction == "LEFT":
        moved = move_left_advanced(next_grid, rows, cols)
    elif direction == "RIGHT":
        moved = move_right_advanced(next_grid, rows, cols)
    elif direction == "UP":
        moved = move_up_advanced(next_grid, rows, cols)
    elif direction == "DOWN":
        moved = move_down_advanced(next_grid, rows, cols)
    
    # Add new tile if move happened
    if moved and has_empty_cell(next_grid, rows, cols):
        next_grid = add_random_tile(next_grid, rows, cols)
    

    end_game = check_game_status_advanced(next_grid, rows, cols)
    
    return next_grid, end_game


# ---------- Helpers ----------

def convert_value(value):
    if value is None:
        return None
    elif isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    elif isinstance(value, str):
        if value in ["0", "*2", "1"]:
            return value
        try:
            num = int(value)
            return num if num > 0 else None
        except ValueError:
            return None
    return None


def compress_row(row, direction="LEFT"):
    """Compress a row/col with special 2048 rules"""
    if direction in ("RIGHT", "DOWN"):
        row = row[::-1]

    result = []
    segment = []

    # Split row into segments blocked by '0'
    for cell in row:
        if cell == "0":
            result.extend(process_segment(segment))
            result.append("0")
            segment = []
        else:
            segment.append(cell)
    result.extend(process_segment(segment))

    # Pad with None
    while len(result) < len(row):
        result.append(None)

    if direction in ("RIGHT", "DOWN"):
        result = result[::-1]
    return result


def process_segment(segment):
    """Process one segment without '0' walls"""
    seg = [c for c in segment if c is not None]
    output = []
    i = 0
    while i < len(seg):
        curr = seg[i]
        if i + 1 < len(seg):
            nxt = seg[i + 1]

            # int + int
            if isinstance(curr, int) and isinstance(nxt, int) and curr == nxt and curr != 1:
                output.append(curr * 2)
                i += 2
                continue

            # int + *2
            if isinstance(curr, int) and nxt == "*2":
                output.append(curr * 2)
                i += 2
                continue

            # 1 + *2
            if curr == "1" and nxt == "*2":
                output.append(2)
                i += 2
                continue

        output.append(curr)
        i += 1
    return output


# ---------- Move functions ----------

def move_left_advanced(grid, rows, cols):
    moved = False
    for i in range(rows):
        new_row = compress_row(grid[i], "LEFT")
        if new_row != grid[i]:
            moved = True
            grid[i] = new_row
    return moved


def move_right_advanced(grid, rows, cols):
    moved = False
    for i in range(rows):
        new_row = compress_row(grid[i], "RIGHT")
        if new_row != grid[i]:
            moved = True
            grid[i] = new_row
    return moved


def move_up_advanced(grid, rows, cols):
    moved = False
    for j in range(cols):
        col = [grid[i][j] for i in range(rows)]
        new_col = compress_row(col, "UP")
        if new_col != col:
            moved = True
            for i in range(rows):
                grid[i][j] = new_col[i]
    return moved


def move_down_advanced(grid, rows, cols):
    moved = False
    for j in range(cols):
        col = [grid[i][j] for i in range(rows)]
        new_col = compress_row(col, "DOWN")
        if new_col != col:
            moved = True
            for i in range(rows):
                grid[i][j] = new_col[i]
    return moved


# ---------- Misc ----------

def has_empty_cell(grid, rows, cols):
    return any(grid[i][j] is None for i in range(rows) for j in range(cols))


def add_random_tile(grid, rows, cols):
    empty_cells = [(i, j) for i in range(rows) for j in range(cols) if grid[i][j] is None]
    if empty_cells:
        i, j = random.choice(empty_cells)
        grid[i][j] = 2 if random.random() < 0.9 else 4
    return grid


def check_game_status_advanced(grid, rows, cols):
    # Win?
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2048:
                return "win"
    # Empty?
    if has_empty_cell(grid, rows, cols):
        return None
    # Possible merges?
    for i in range(rows):
        for j in range(cols):
            curr = grid[i][j]
            if curr is None:
                continue
            if j < cols - 1 and can_merge(curr, grid[i][j + 1]):
                return None
            if i < rows - 1 and can_merge(curr, grid[i + 1][j]):
                return None
    return "lose"


def can_merge(a, b):
    if a is None or b is None:
        return False
    if isinstance(a, int) and isinstance(b, int) and a == b and a != 1:
        return True
    if isinstance(a, int) and b == "*2":
        return True
    if a == "*2" and isinstance(b, int):
        return True
    if (a == "1" and b == "*2") or (a == "*2" and b == "1"):
        return True
    return False