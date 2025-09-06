import random

def process_2048_move(grid, direction):
    """
    Process a 2048 move with advanced tile types
    """
    if not grid or not isinstance(grid, list):
        return grid, None
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    # Create a working copy with proper type handling
    next_grid = [[convert_value(cell) for cell in row] for row in grid]
    moved = False
    
    # Process move based on direction
    if direction == "LEFT":
        moved = move_left_advanced(next_grid, rows, cols)
    elif direction == "RIGHT":
        moved = move_right_advanced(next_grid, rows, cols)
    elif direction == "UP":
        moved = move_up_advanced(next_grid, rows, cols)
    elif direction == "DOWN":
        moved = move_down_advanced(next_grid, rows, cols)
    
    # Only add new tile if grid changed AND there's space
    if moved and has_empty_cell(next_grid, rows, cols):
        next_grid = add_random_tile(next_grid, rows, cols)
    
    if not moved:
        end_game = 'lose'
    else:
        end_game = check_game_status_advanced(next_grid, rows, cols)
    return next_grid, end_game

def convert_value(value):
    """Convert values to proper types (numbers as int, special tiles as strings)"""
    if value is None:
        return None
    elif isinstance(value, (int, float)):
        return int(value) if value > 0 else None
    elif isinstance(value, str):
        if value in ['0', '*2', '1']:
            return value
        try:
            num = int(value)
            return num if num > 0 else None
        except ValueError:
            return None
    return None

def move_left_advanced(grid, rows, cols):
    """Advanced left movement with special tiles"""
    moved = False
    for i in range(rows):
        original_row = grid[i][:]
        new_row = []
        j = 0
        
        while j < cols:
            current = grid[i][j]
            if current is None:
                j += 1
                continue
            
            if current == '0':
                # '0' tile stays in place and blocks movement
                new_row.append('0')
                j += 1
                continue
            
            # Look for next non-None tile
            k = j + 1
            while k < cols and grid[i][k] is None:
                k += 1
            
            if k >= cols:
                # No more tiles to process
                if current != '*2' or (new_row and new_row[-1] != '*2'):
                    new_row.append(current)
                j = cols
                continue
            
            next_tile = grid[i][k]
            
            if current == '*2':
                # '*2' merges with any number in front
                if isinstance(next_tile, int):
                    new_row.append(next_tile * 2)
                    moved = True
                    j = k + 1
                else:
                    new_row.append('*2')
                    j += 1
            elif isinstance(current, int):
                if next_tile == '*2':
                    # Number followed by '*2' gets multiplied
                    new_row.append(current * 2)
                    moved = True
                    j = k + 1
                elif next_tile == current and next_tile != '1':
                    # Merge same numbers (except '1')
                    new_row.append(current * 2)
                    moved = True
                    j = k + 1
                else:
                    new_row.append(current)
                    j += 1
            elif current == '1':
                if next_tile == '*2':
                    # '1' followed by '*2' becomes 2
                    new_row.append(2)
                    moved = True
                    j = k + 1
                else:
                    new_row.append('1')
                    j += 1
        
        # Pad with None values and handle remaining '*2' tiles
        while len(new_row) < cols:
            new_row.append(None)
        
        # Update the row
        if original_row != new_row:
            moved = True
            grid[i] = new_row[:cols]
    
    return moved

def move_right_advanced(grid, rows, cols):
    """Advanced right movement with special tiles"""
    moved = False
    for i in range(rows):
        original_row = grid[i][:]
        new_row = []
        j = cols - 1
        
        while j >= 0:
            current = grid[i][j]
            if current is None:
                j -= 1
                continue
            
            if current == '0':
                # '0' tile stays in place
                new_row.insert(0, '0')
                j -= 1
                continue
            
            # Look for previous non-None tile
            k = j - 1
            while k >= 0 and grid[i][k] is None:
                k -= 1
            
            if k < 0:
                # No more tiles to process
                if current != '*2' or (new_row and new_row[0] != '*2'):
                    new_row.insert(0, current)
                j = -1
                continue
            
            prev_tile = grid[i][k]
            
            if current == '*2':
                # '*2' merges with any number behind it
                if isinstance(prev_tile, int):
                    new_row.insert(0, prev_tile * 2)
                    moved = True
                    j = k - 1
                else:
                    new_row.insert(0, '*2')
                    j -= 1
            elif isinstance(current, int):
                if prev_tile == '*2':
                    # Number preceded by '*2' gets multiplied
                    new_row.insert(0, current * 2)
                    moved = True
                    j = k - 1
                elif prev_tile == current and prev_tile != '1':
                    # Merge same numbers
                    new_row.insert(0, current * 2)
                    moved = True
                    j = k - 1
                else:
                    new_row.insert(0, current)
                    j -= 1
            elif current == '1':
                if prev_tile == '*2':
                    # '1' preceded by '*2' becomes 2
                    new_row.insert(0, 2)
                    moved = True
                    j = k - 1
                else:
                    new_row.insert(0, '1')
                    j -= 1
        
        # Pad with None values
        while len(new_row) < cols:
            new_row.insert(0, None)
        
        if original_row != new_row:
            moved = True
            grid[i] = new_row[:cols]
    
    return moved

def move_up_advanced(grid, rows, cols):
    """Advanced upward movement with special tiles"""
    moved = False
    for j in range(cols):
        original_col = [grid[i][j] for i in range(rows)]
        new_col = []
        i = 0
        
        while i < rows:
            current = grid[i][j]
            if current is None:
                i += 1
                continue
            
            if current == '0':
                new_col.append('0')
                i += 1
                continue
            
            k = i + 1
            while k < rows and grid[k][j] is None:
                k += 1
            
            if k >= rows:
                if current != '*2' or (new_col and new_col[-1] != '*2'):
                    new_col.append(current)
                i = rows
                continue
            
            next_tile = grid[k][j]
            
            if current == '*2':
                if isinstance(next_tile, int):
                    new_col.append(next_tile * 2)
                    moved = True
                    i = k + 1
                else:
                    new_col.append('*2')
                    i += 1
            elif isinstance(current, int):
                if next_tile == '*2':
                    new_col.append(current * 2)
                    moved = True
                    i = k + 1
                elif next_tile == current and next_tile != '1':
                    new_col.append(current * 2)
                    moved = True
                    i = k + 1
                else:
                    new_col.append(current)
                    i += 1
            elif current == '1':
                if next_tile == '*2':
                    new_col.append(2)
                    moved = True
                    i = k + 1
                else:
                    new_col.append('1')
                    i += 1
        
        while len(new_col) < rows:
            new_col.append(None)
        
        # Update the column
        for idx in range(rows):
            if grid[idx][j] != new_col[idx]:
                moved = True
                grid[idx][j] = new_col[idx]
    
    return moved

def move_down_advanced(grid, rows, cols):
    """Advanced downward movement with special tiles"""
    moved = False
    for j in range(cols):
        original_col = [grid[i][j] for i in range(rows)]
        new_col = []
        i = rows - 1
        
        while i >= 0:
            current = grid[i][j]
            if current is None:
                i -= 1
                continue
            
            if current == '0':
                new_col.insert(0, '0')
                i -= 1
                continue
            
            k = i - 1
            while k >= 0 and grid[k][j] is None:
                k -= 1
            
            if k < 0:
                if current != '*2' or (new_col and new_col[0] != '*2'):
                    new_col.insert(0, current)
                i = -1
                continue
            
            prev_tile = grid[k][j]
            
            if current == '*2':
                if isinstance(prev_tile, int):
                    new_col.insert(0, prev_tile * 2)
                    moved = True
                    i = k - 1
                else:
                    new_col.insert(0, '*2')
                    i -= 1
            elif isinstance(current, int):
                if prev_tile == '*2':
                    new_col.insert(0, current * 2)
                    moved = True
                    i = k - 1
                elif prev_tile == current and prev_tile != '1':
                    new_col.insert(0, current * 2)
                    moved = True
                    i = k - 1
                else:
                    new_col.insert(0, current)
                    i -= 1
            elif current == '1':
                if prev_tile == '*2':
                    new_col.insert(0, 2)
                    moved = True
                    i = k - 1
                else:
                    new_col.insert(0, '1')
                    i -= 1
        
        while len(new_col) < rows:
            new_col.insert(0, None)
        
        # Update the column
        for idx in range(rows):
            if grid[idx][j] != new_col[idx]:
                moved = True
                grid[idx][j] = new_col[idx]
    
    return moved

def has_empty_cell(grid, rows, cols):
    """Check for empty cells"""
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] is None:
                return True
    return False

def add_random_tile(grid, rows, cols):
    """Add random tile (only 2 or 4 for now)"""
    empty_cells = []
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] is None:
                empty_cells.append((i, j))
    
    if empty_cells:
        i, j = random.choice(empty_cells)
        grid[i][j] = 2 if random.random() < 0.9 else 4
    
    return grid

def check_game_status_advanced(grid, rows, cols):
    """Check game status considering special tiles"""
    # Check for win
    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 2048:
                return 'win'
    
    # Check for empty cells
    if has_empty_cell(grid, rows, cols):
        return None
    
    # Check for possible moves considering special tiles
    for i in range(rows):
        for j in range(cols):
            current = grid[i][j]
            if current is None:
                continue
            
            # Check adjacent tiles for possible moves
            if j < cols - 1:
                next_tile = grid[i][j + 1]
                if can_merge(current, next_tile):
                    return None
            if i < rows - 1:
                below_tile = grid[i + 1][j]
                if can_merge(current, below_tile):
                    return None
    
    return 'lose'

def can_merge(tile1, tile2):
    """Check if two tiles can merge considering special rules"""
    if tile1 is None or tile2 is None:
        return False
    
    # Regular number merging
    if isinstance(tile1, int) and isinstance(tile2, int) and tile1 == tile2:
        return True
    
    # '*2' can merge with any number
    if tile1 == '*2' and isinstance(tile2, int):
        return True
    if isinstance(tile1, int) and tile2 == '*2':
        return True
    
    # '*2' can transform '1' into 2
    if tile1 == '*2' and tile2 == '1':
        return True
    if tile1 == '1' and tile2 == '*2':
        return True
    
    return False