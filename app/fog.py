import logging
from typing import Dict, List, Tuple, Set, Optional
import json

logger = logging.getLogger(__name__)

class FogOfWallSolver:
    def __init__(self):
        # Game state tracking
        self.game_states: Dict[str, 'GameState'] = {}
    
    def process_request(self, data: dict) -> dict:
        """Main entry point for processing fog-of-wall requests"""
        challenger_id = data['challenger_id']
        game_id = data['game_id']
        
        # Create game state key
        state_key = f"{challenger_id}_{game_id}"
        
        # Check if this is an initial request
        if 'test_case' in data:
            # Initialize new game state
            self.game_states[state_key] = GameState(data['test_case'])
            logger.info(f"Starting new game: {state_key}")
        
        # Get current game state
        game_state = self.game_states.get(state_key)
        if not game_state:
            raise ValueError(f"Game state not found for {state_key}")
        
        # Process previous action if present
        if 'previous_action' in data:
            game_state.process_previous_action(data['previous_action'])
        
        # Decide next action
        next_action = game_state.get_next_action()
        
        # Build response
        response = {
            "challenger_id": challenger_id,
            "game_id": game_id
        }
        response.update(next_action)
        
        return response


class GameState:
    def __init__(self, test_case: dict):
        self.grid_size = test_case['length_of_grid']
        self.num_walls = test_case['num_of_walls']
        self.crows = {crow['id']: Crow(crow['id'], crow['x'], crow['y']) for crow in test_case['crows']}
        
        # Grid state: 0=unknown, 1=empty, 2=wall, 3=out_of_bounds
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.discovered_walls: Set[Tuple[int, int]] = set()
        self.scanned_positions: Set[Tuple[int, int]] = set()
        self.move_count = 0
        self.max_moves = self.grid_size * self.grid_size
        
        # Strategy variables
        self.current_crow_id = None
        self.exploration_phase = True
        self.scan_positions = self._generate_optimal_scan_positions()
        self.current_scan_target = 0
        
        logger.info(f"Initialized game: grid_size={self.grid_size}, num_walls={self.num_walls}, crows={len(self.crows)}")
    
    def _generate_optimal_scan_positions(self) -> List[Tuple[int, int]]:
        """Generate optimal scanning positions to minimize overlap"""
        positions = []
        # Use a grid pattern with 5-cell spacing to minimize scan overlap
        # Since scan is 5x5, we want scans centered every 5 cells to minimize overlap
        step = 4  # Slight overlap to ensure no gaps
        
        for y in range(2, self.grid_size - 2, step):
            for x in range(2, self.grid_size - 2, step):
                positions.append((x, y))
        
        # Add edge positions to ensure complete coverage
        edges = []
        # Top and bottom edges
        for x in range(2, self.grid_size - 2, step):
            if (x, 2) not in positions:
                edges.append((x, 2))
            if (x, self.grid_size - 3) not in positions:
                edges.append((x, self.grid_size - 3))
        
        # Left and right edges
        for y in range(2, self.grid_size - 2, step):
            if (2, y) not in positions:
                edges.append((2, y))
            if (self.grid_size - 3, y) not in positions:
                edges.append((self.grid_size - 3, y))
        
        positions.extend(edges)
        return positions
    
    def process_previous_action(self, action: dict):
        """Process the result of the previous action"""
        self.move_count += 1
        crow_id = action['crow_id']
        crow = self.crows[crow_id]
        
        if action['your_action'] == 'move':
            # Update crow position
            new_x, new_y = action['move_result']
            crow.x, crow.y = new_x, new_y
            logger.info(f"Crow {crow_id} moved to ({new_x}, {new_y})")
            
        elif action['your_action'] == 'scan':
            # Process scan result
            scan_result = action['scan_result']
            self._process_scan_result(crow, scan_result)
            self.scanned_positions.add((crow.x, crow.y))
            logger.info(f"Crow {crow_id} scanned at ({crow.x}, {crow.y}), walls found: {len(self.discovered_walls)}")
    
    def _process_scan_result(self, crow: 'Crow', scan_result: List[List[str]]):
        """Process the 5x5 scan result"""
        # Scan result is 5x5 with crow at center [2][2]
        for dy in range(5):
            for dx in range(5):
                # Convert scan coordinates to world coordinates
                world_x = crow.x + dx - 2
                world_y = crow.y + dy - 2
                cell = scan_result[dy][dx]
                
                # Check bounds
                if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                    if cell == 'W':
                        self.grid[world_y][world_x] = 2  # Wall
                        self.discovered_walls.add((world_x, world_y))
                    elif cell == '_' or cell == 'C':
                        self.grid[world_y][world_x] = 1  # Empty
                # Note: 'X' represents out of bounds areas, no action needed
    
    def get_next_action(self) -> dict:
        """Determine the next action to take"""
        # Check if we should submit
        if self._should_submit():
            return self._create_submission()
        
        # Choose which crow to use
        if self.exploration_phase:
            return self._explore_systematically()
        else:
            return self._fill_gaps()
    
    def _should_submit(self) -> bool:
        """Determine if we should submit our findings"""
        # Submit if we've found all walls or are running out of moves
        if len(self.discovered_walls) >= self.num_walls:
            return True
        if self.move_count >= self.max_moves - 5:  # Leave some buffer
            return True
        # Submit if we've completed systematic exploration and gap filling
        if self.current_scan_target >= len(self.scan_positions) and not self._has_unexplored_areas():
            return True
        return False
    
    def _has_unexplored_areas(self) -> bool:
        """Check if there are significant unexplored areas"""
        unexplored_count = 0
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if self.grid[y][x] == 0:  # Unknown
                    unexplored_count += 1
        
        return unexplored_count > self.grid_size  # Threshold for remaining exploration
    
    def _explore_systematically(self) -> dict:
        """Systematic exploration using optimal scan positions"""
        if self.current_scan_target >= len(self.scan_positions):
            self.exploration_phase = False
            return self._fill_gaps()
        
        target_x, target_y = self.scan_positions[self.current_scan_target]
        
        # Find the best crow for this target
        best_crow = self._find_closest_crow(target_x, target_y)
        
        # Check if we're already at the target
        if best_crow.x == target_x and best_crow.y == target_y:
            # Scan if we haven't scanned here yet
            if (target_x, target_y) not in self.scanned_positions:
                return {"crow_id": best_crow.id, "action_type": "scan"}
            else:
                # Move to next target
                self.current_scan_target += 1
                return self._explore_systematically()
        
        # Move towards target
        direction = self._get_direction_to_target(best_crow, target_x, target_y)
        return {"crow_id": best_crow.id, "action_type": "move", "direction": direction}
    
    def _fill_gaps(self) -> dict:
        """Fill in remaining gaps in exploration"""
        # Find the largest unexplored area
        unexplored_areas = self._find_unexplored_areas()
        
        if not unexplored_areas:
            return self._create_submission()
        
        # Choose the best area to explore
        target_x, target_y = self._choose_best_exploration_target(unexplored_areas)
        
        # Find best crow for this target
        best_crow = self._find_closest_crow(target_x, target_y)
        
        # Check if we're at the target
        if best_crow.x == target_x and best_crow.y == target_y:
            return {"crow_id": best_crow.id, "action_type": "scan"}
        
        # Move towards target
        direction = self._get_direction_to_target(best_crow, target_x, target_y)
        return {"crow_id": best_crow.id, "action_type": "move", "direction": direction}
    
    def _find_unexplored_areas(self) -> List[Tuple[int, int]]:
        """Find areas that haven't been properly explored"""
        unexplored = []
        for y in range(2, self.grid_size - 2):  # Leave border for scan range
            for x in range(2, self.grid_size - 2):
                if (x, y) not in self.scanned_positions:
                    # Check if this position would reveal new information
                    if self._would_scan_reveal_new_info(x, y):
                        unexplored.append((x, y))
        return unexplored
    
    def _would_scan_reveal_new_info(self, x: int, y: int) -> bool:
        """Check if scanning at this position would reveal new information"""
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size:
                    if self.grid[check_y][check_x] == 0:  # Unknown cell
                        return True
        return False
    
    def _choose_best_exploration_target(self, targets: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Choose the best exploration target from available options"""
        if not targets:
            return (self.grid_size // 2, self.grid_size // 2)  # Default center
        
        # Score targets based on potential information gain and distance
        best_score = -1
        best_target = targets[0]
        
        for target in targets:
            score = self._score_exploration_target(target)
            if score > best_score:
                best_score = score
                best_target = target
        
        return best_target
    
    def _score_exploration_target(self, target: Tuple[int, int]) -> float:
        """Score an exploration target based on information gain potential"""
        x, y = target
        unknown_count = 0
        
        # Count unknown cells in scan range
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < self.grid_size and 0 <= check_y < self.grid_size:
                    if self.grid[check_y][check_x] == 0:
                        unknown_count += 1
        
        # Find distance to closest crow
        min_distance = min(abs(crow.x - x) + abs(crow.y - y) for crow in self.crows.values())
        
        # Score: prioritize information gain, penalize distance
        score = unknown_count * 10 - min_distance
        return score
    
    def _find_closest_crow(self, target_x: int, target_y: int) -> 'Crow':
        """Find the crow closest to the target position"""
        min_distance = float('inf')
        best_crow = list(self.crows.values())[0]
        
        for crow in self.crows.values():
            distance = abs(crow.x - target_x) + abs(crow.y - target_y)
            if distance < min_distance:
                min_distance = distance
                best_crow = crow
        
        return best_crow
    
    def _get_direction_to_target(self, crow: 'Crow', target_x: int, target_y: int) -> str:
        """Get the direction for the crow to move towards target"""
        dx = target_x - crow.x
        dy = target_y - crow.y
        
        # Choose the direction that reduces the largest distance component
        if abs(dx) > abs(dy):
            return "E" if dx > 0 else "W"
        else:
            return "S" if dy > 0 else "N"
    
    def _create_submission(self) -> dict:
        """Create the final submission with discovered walls"""
        submission = [f"{x}-{y}" for x, y in self.discovered_walls]
        logger.info(f"Submitting {len(submission)} walls out of {self.num_walls} expected")
        return {
            "action_type": "submit",
            "submission": submission
        }


class Crow:
    def __init__(self, crow_id: str, x: int, y: int):
        self.id = crow_id
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Crow({self.id}, {self.x}, {self.y})"


# Global solver instance
fog_solver = FogOfWallSolver()


def solve_fog_of_wall(data: dict) -> dict:
    """Main function to solve fog-of-wall requests"""
    return fog_solver.process_request(data)