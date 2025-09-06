from collections import defaultdict
import heapq
from typing import List
from flask import Blueprint, jsonify, request
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
import math
import re

main_bp = Blueprint('main', __name__)

# Simple test route
@main_bp.route('/')
def home():
    return jsonify({
        'message': 'Welcome to Flask Backend!',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })
#####################################---TRIVIA---###############################################
@main_bp.route('/trivia', methods=['GET'])
def trivia():
    """
    GET endpoint that returns answers to the multiple choice trivia questions
    """
    answers = [
        2,  
        1,  
        2,  
        2,  
        3,  
        1,  
        3,  
        5,  
        4   
    ]
    
    return jsonify({
        "answers": answers
    })

#########################################---TICKETING---###########################################
def compute_distance(loc1, loc2):
    x1, y1 = loc1
    x2, y2 = loc2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

@main_bp.route('/ticketing-agent', methods=['POST'])
def ticketing_agent():
    data = request.get_json()
    customers = data['customers']
    concerts = data['concerts']
    priority = data['priority']

    result = {}
    for customer in customers:
        best_concert = None
        best_score = -1
        for concert in concerts:
            score = 0
            # VIP factor
            if customer['vip_status']:
                score += 100
            # Credit card factor
            cc = customer['credit_card']
            if cc in priority and priority[cc] == concert['name']:
                score += 50
            # Latency factor
            dist = compute_distance(customer['location'], concert['booking_center_location'])
            latency_pts = max(0, 30 - int(dist))
            score += latency_pts

            if score > best_score:
                best_score = score
                best_concert = concert['name']
        result[customer['name']] = best_concert

    return jsonify(result)

#######################################---BLANKETY---#############################################
def simple_impute(series):
    """Simple but robust imputation using pandas interpolation"""
    # Convert to pandas Series
    ts = pd.Series(series)
    
    # Use linear interpolation with limit to handle edge cases
    imputed = ts.interpolate(method='linear', limit_direction='both')
    
    # Fill any remaining NaNs with forward/backward fill or median
    if imputed.isna().any():
        imputed = imputed.fillna(method='ffill').fillna(method='bfill')
        if imputed.isna().any():
            imputed = imputed.fillna(imputed.median())
    
    return imputed.tolist()

@main_bp.route('/blankety', methods=['POST'])
def blankety():
    data = request.get_json()
    series_list = data['series']
    result = []
    
    for series in series_list:
        imputed_series = simple_impute(series)
        result.append(imputed_series)
    
    return jsonify({'answer': result})

#########################################---PRINCESS---###########################################
def dijkstra(graph, start, n_stations):
    dist = [float('inf')] * n_stations
    dist[start] = 0
    pq = [(0, start)]
    
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        
        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(pq, (dist[v], v))
    
    return dist

def solve_subway_scheduling(edges, tasks, s0):
    if not tasks:
        return 0, 0, []
    
    # Build adjacency list for the graph
    stations = set([s0])
    for s1, s2, c in edges:
        stations.add(s1)
        stations.add(s2)
    
    for task in tasks:
        stations.add(task[2])  # Add task stations
    
    station_list = sorted(list(stations))
    station_to_idx = {station: i for i, station in enumerate(station_list)}
    n_stations = len(station_list)
    
    # Build graph
    graph = defaultdict(list)
    for s1, s2, c in edges:
        idx1, idx2 = station_to_idx[s1], station_to_idx[s2]
        graph[idx1].append((idx2, c))
        graph[idx2].append((idx1, c))
    
    # Only compute distances from task stations and starting station
    distance_cache = {}
    
    def get_distance(from_station, to_station):
        key = (from_station, to_station)
        if key not in distance_cache:
            from_idx = station_to_idx[from_station]
            distances = dijkstra(graph, from_idx, n_stations)
            to_idx = station_to_idx[to_station]
            distance_cache[key] = distances[to_idx]
        return distance_cache[key]
    
    # Sort tasks by end time
    indexed_tasks = [(tasks[i][0], tasks[i][1], tasks[i][2], tasks[i][3], i) 
                     for i in range(len(tasks))]
    indexed_tasks.sort(key=lambda x: x[1])
    
    n = len(indexed_tasks)
    
    # dp[i] = (max_reward, min_fee, prev_task_index)
    dp = [None] * n
    
    for i in range(n):
        start_time, end_time, station, reward, orig_idx = indexed_tasks[i]
        
        # Option 1: Take this task as the first task (from starting station)
        transport_cost = get_distance(s0, station)
        if transport_cost != float('inf'):
            dp[i] = (reward, transport_cost, -1)
        
        # Option 2: Take this task after some previous compatible task
        # Use binary search to find the latest compatible task
        left, right = 0, i - 1
        latest_compatible = -1
        
        while left <= right:
            mid = (left + right) // 2
            if indexed_tasks[mid][1] <= start_time:  # prev_end <= start_time
                latest_compatible = mid
                left = mid + 1
            else:
                right = mid - 1
        
        # Check all compatible tasks starting from the latest one
        for j in range(latest_compatible, -1, -1):
            if dp[j] is None:
                continue
                
            prev_start, prev_end, prev_station, prev_reward, prev_orig_idx = indexed_tasks[j]
            prev_total_reward, prev_total_fee, _ = dp[j]
            
            # Early termination: if current best is already better, no need to check earlier tasks
            if dp[i] is not None and prev_total_reward + reward < dp[i][0]:
                break
                
            transport_cost = get_distance(prev_station, station)
            
            if transport_cost != float('inf'):
                new_reward = prev_total_reward + reward
                new_fee = prev_total_fee + transport_cost
                
                # Check if this is better than current best for task i
                if (dp[i] is None or new_reward > dp[i][0] or 
                    (new_reward == dp[i][0] and new_fee < dp[i][1])):
                    dp[i] = (new_reward, new_fee, j)
    
    # Find the task with maximum reward (and minimum fee if tied)
    max_reward = 0
    min_fee = float('inf')
    best_task = -1
    
    for i in range(n):
        if dp[i] is None:
            continue
            
        reward, fee, _ = dp[i]
        last_station = indexed_tasks[i][2]
        
        # Add return cost to starting station
        return_cost = get_distance(last_station, s0)
        total_fee = fee + return_cost
        
        if (reward > max_reward or 
            (reward == max_reward and total_fee < min_fee)):
            max_reward = reward
            min_fee = total_fee
            best_task = i
    
    # Reconstruct solution by backtracking
    selected_tasks = []
    current = best_task
    
    while current != -1:
        if dp[current] is None:
            break
        _, _, prev_task = dp[current]
        selected_tasks.append(indexed_tasks[current][4])  # original index
        current = prev_task
    
    selected_tasks.reverse()
    
    return max_reward, min_fee, selected_tasks

@main_bp.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    data = request.get_json()
    
    tasks_data = data['tasks']
    subway_data = data['subway']
    starting_station = data['starting_station']

    tasks = []
    task_names = []
    for task in tasks_data:
        tasks.append([
            task['start'],
            task['end'], 
            task['station'],
            task['score']
        ])
        task_names.append(task['name'])
    
    edges = []
    for connection in subway_data:
        [s1, s2] = connection['connection']
        fee = connection['fee']
        edges.append([s1, s2, fee])
    
    max_reward, min_fee, selected_tasks = solve_subway_scheduling(edges, tasks, starting_station)
    
    selected_with_start_time = [(i, tasks[i][0]) for i in selected_tasks]  # (index, start_time)
    selected_with_start_time.sort(key=lambda x: x[1])  # Sort by start time
    schedule = [task_names[i] for i, _ in selected_with_start_time]
    
    response = {
        "max_score": max_reward,
        "min_fee": min_fee,
        "schedule": schedule
    }
    
    return jsonify(response)

#########################################---TRADING---###########################################
class LaTeXFormulaEvaluator:
    def __init__(self):
        # Precompiled regex patterns for better performance
        self.patterns = {
            'dollar_signs': re.compile(r'\$\$'),
            'text_commands': re.compile(r'\\text\{([^}]+)\}'),
            'frac': re.compile(r'\\frac\{([^}]+)\}\{([^}]+)\}'),
            'max_min': re.compile(r'\\(max|min)\s*\(([^)]+)\)'),
            'cdot': re.compile(r'\\cdot'),
            'times': re.compile(r'\\times'),
            'sum': re.compile(r'\\sum'),
            'log': re.compile(r'\\log\s*\(([^)]+)\)'),
            'exp': re.compile(r'e\^\{([^}]+)\}'),
            'exp_simple': re.compile(r'e\^([a-zA-Z_]\w*)'),
            'subscript': re.compile(r'([a-zA-Z_]\w*)_\{([^}]+)\}'),
            'subscript_simple': re.compile(r'([a-zA-Z_]\w*)_([a-zA-Z_]\w*)'),
            'variable': re.compile(r'[a-zA-Z_]\w*'),
            'whitespace': re.compile(r'\s+'),
            'assignment': re.compile(r'^([^=]+)\s*=\s*(.+)$'),
            'bracket_var': re.compile(r'([a-zA-Z_]\w*)\[([^\]]+)\]')
        }
    
    def preprocess_formula(self, formula):
        """Clean and normalize the LaTeX formula"""
        # Remove $$ markers
        formula = self.patterns['dollar_signs'].sub('', formula)
        
        # Handle assignment equations (extract right side)
        match = self.patterns['assignment'].match(formula.strip())
        if match:
            formula = match.group(2).strip()
        
        # Convert \text{Variable} to Variable
        formula = self.patterns['text_commands'].sub(r'\1', formula)
        
        # Convert fractions: \frac{a}{b} -> (a)/(b)
        def replace_frac(match):
            numerator = match.group(1)
            denominator = match.group(2)
            return f'({numerator})/({denominator})'
        formula = self.patterns['frac'].sub(replace_frac, formula)
        
        # Convert max/min functions
        def replace_max_min(match):
            func = match.group(1)
            args = match.group(2)
            return f'{func}({args})'
        formula = self.patterns['max_min'].sub(replace_max_min, formula)
        
        # Convert multiplication symbols
        formula = self.patterns['cdot'].sub('*', formula)
        formula = self.patterns['times'].sub('*', formula)
        
        # Handle exponentials: e^{x} -> exp(x)
        def replace_exp(match):
            exponent = match.group(1)
            return f'exp({exponent})'
        formula = self.patterns['exp'].sub(replace_exp, formula)
        formula = self.patterns['exp_simple'].sub(r'exp(\1)', formula)
        
        # Handle logarithms: \log(x) -> log(x)
        formula = self.patterns['log'].sub(r'log(\1)', formula)
        
        # Handle subscripts: Variable_subscript -> Variable_subscript
        formula = self.patterns['subscript'].sub(r'\1_\2', formula)
        formula = self.patterns['subscript_simple'].sub(r'\1_\2', formula)
        
        # Handle bracket notation: E[R_m] -> E_R_m
        formula = self.patterns['bracket_var'].sub(r'\1_\2', formula)
        
        # Remove extra whitespace
        formula = self.patterns['whitespace'].sub(' ', formula).strip()
        
        return formula
    
    def substitute_variables(self, formula, variables):
        """Replace variables in formula with their values"""
        # Sort variables by length (descending) to avoid partial replacements
        sorted_vars = sorted(variables.keys(), key=len, reverse=True)
        
        for var in sorted_vars:
            value = variables[var]
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(var) + r'\b'
            formula = re.sub(pattern, str(value), formula)
        
        return formula
    
    def safe_eval(self, expression):
        """Safely evaluate mathematical expressions"""
        # Define allowed functions and constants
        allowed_names = {
            '__builtins__': {},
            'abs': abs,
            'max': max,
            'min': min,
            'sum': sum,
            'exp': math.exp,
            'log': math.log,
            'sqrt': math.sqrt,
            'sin': math.sin,
            'cos': math.cos,
            'tan': math.tan,
            'pi': math.pi,
            'e': math.e,
        }
        
        try:
            result = eval(expression, allowed_names)
            return float(result)
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
    
    def evaluate(self, formula, variables):
        """Main evaluation function"""
        try:
            # Step 1: Preprocess the formula
            processed_formula = self.preprocess_formula(formula)
            
            # Step 2: Substitute variables
            substituted_formula = self.substitute_variables(processed_formula, variables)
            
            # Step 3: Evaluate the expression
            result = self.safe_eval(substituted_formula)
            
            # Step 4: Round to 4 decimal places
            return round(result, 4)
            
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}': {str(e)}")

evaluator = LaTeXFormulaEvaluator()

@main_bp.route('/trading-formula', methods=['POST'])
def trading_formula():
    """
    Endpoint to evaluate LaTeX formulas for financial calculations
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if not isinstance(data, list):
            return jsonify({'error': 'Expected JSON array'}), 400
        
        results = []
        
        # Process each test case
        for i, test_case in enumerate(data):
            try:
                # Validate test case structure
                if not all(key in test_case for key in ['name', 'formula', 'variables', 'type']):
                    return jsonify({'error': f'Missing required fields in test case {i+1}'}), 400
                
                if test_case['type'] != 'compute':
                    return jsonify({'error': f'Unsupported type "{test_case["type"]}" in test case {i+1}'}), 400
                
                # Extract data
                formula = test_case['formula']
                variables = test_case['variables']
                
                # Evaluate the formula
                result = evaluator.evaluate(formula, variables)
                
                # Append result
                results.append({'result': result})
                
            except Exception as e:
                return jsonify({'error': f'Error processing test case {i+1} ({test_case.get("name", "unnamed")}): {str(e)}'}), 400
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

#######################################---FLAG---#############################################
@main_bp.route('/chasetheflag', methods=['POST'])
def chase_the_flag_main():
    """
    POST endpoint that returns flags for the chase the flag challenges
    """
    flags = {
        "challenge1": "2-nOO9QiTIwXgNtWtBJezz8kv3SLc",
        "challenge2": "your_actual_flag_2", 
        "challenge3": "your_actual_flag_3",
        "challenge4": "your_actual_flag_4",
        "challenge5": "your_actual_flag_5"
    }
    
    return jsonify(flags)

#######################################---investigate---#############################################
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}
    
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False  # Already connected, this edge creates a cycle
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True

def find_cycle_edges(network):
    """
    Fast cycle detection using Union-Find.
    Since there's exactly one cycle, we build MST and find the extra edge,
    then reconstruct the cycle efficiently.
    """
    if not network:
        return []
    
    # Use Union-Find to detect the cycle-forming edge
    uf = UnionFind()
    tree_edges = []
    cycle_edge = None
    
    for connection in network:
        spy1, spy2 = connection['spy1'], connection['spy2']
        if not uf.union(spy1, spy2):
            # This edge creates the cycle
            cycle_edge = connection
        else:
            tree_edges.append(connection)
    
    if not cycle_edge:
        return []  # No cycle found
    
    # Now we need to find the path between the two nodes of the cycle edge
    # using only tree edges, then add the cycle edge to complete the cycle
    
    # Build adjacency list from tree edges only
    graph = defaultdict(list)
    for edge in tree_edges:
        spy1, spy2 = edge['spy1'], edge['spy2']
        graph[spy1].append((spy2, edge))
        graph[spy2].append((spy1, edge))
    
    # Find path between cycle edge endpoints using BFS
    start, end = cycle_edge['spy1'], cycle_edge['spy2']
    queue = [(start, [])]
    visited = {start}
    path_edges = []
    
    while queue:
        node, path = queue.pop(0)
        if node == end:
            path_edges = path
            break
        
        for neighbor, edge in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [edge]))
    
    # The cycle consists of the path edges + the cycle edge
    cycle_edges = path_edges + [cycle_edge]
    return cycle_edges

@main_bp.route('/investigate', methods=['POST'])
def investigate():
    """
    POST endpoint to find all extra channels (edges that are part of the single cycle)
    in spy networks to remove cycles.
    """
    data = request.get_json()
    networks = data['networks']
    
    result_networks = []
    
    for network_data in networks:
        network_id = network_data['networkId']
        network = network_data['network']
        
        extra_channels = find_cycle_edges(network)
        
        result_networks.append({
            "networkId": network_id,
            "extraChannels": extra_channels
        })
    
    return jsonify({
        "networks": result_networks
    })

# Challenge 1: Transformation functions and their inverses
def mirror_words(x: str) -> str:
    """Reverse each word in the sentence"""
    return ' '.join([word[::-1] for word in x.split()])

def encode_mirror_alphabet(x: str) -> str:
    """Replace each letter with its mirror in the alphabet"""
    result = []
    for char in x:
        if char.isalpha():
            if char.islower():
                result.append(chr(219 - ord(char)))  # a=97, z=122 -> 219-97=122, 219-122=97
            else:
                result.append(chr(155 - ord(char)))  # A=65, Z=90 -> 155-65=90, 155-90=65
        else:
            result.append(char)
    return ''.join(result)

def toggle_case(x: str) -> str:
    """Switch uppercase to lowercase and vice versa"""
    return x.swapcase()

def swap_pairs(x: str) -> str:
    """Swap characters in pairs within each word"""
    words = x.split()
    result = []
    for word in words:
        swapped = []
        for i in range(0, len(word) - 1, 2):
            swapped.extend([word[i+1], word[i]])
        if len(word) % 2 == 1:
            swapped.append(word[-1])
        result.append(''.join(swapped))
    return ' '.join(result)

def encode_index_parity(x: str) -> str:
    """Rearrange each word: even indices first, then odd indices"""
    words = x.split()
    result = []
    for word in words:
        evens = [word[i] for i in range(0, len(word), 2)]
        odds = [word[i] for i in range(1, len(word), 2)]
        result.append(''.join(evens + odds))
    return ' '.join(result)

def double_consonants(x: str) -> str:
    """Double every consonant"""
    vowels = 'aeiouAEIOU'
    result = []
    for char in x:
        result.append(char)
        if char.isalpha() and char not in vowels:
            result.append(char)
    return ''.join(result)

# Inverse functions for challenge 1
def inverse_mirror_words(x: str) -> str:
    return mirror_words(x)  # Same as forward

def inverse_encode_mirror_alphabet(x: str) -> str:
    return encode_mirror_alphabet(x)  # Same as forward (involution)

def inverse_toggle_case(x: str) -> str:
    return toggle_case(x)  # Same as forward (involution)

def inverse_swap_pairs(x: str) -> str:
    return swap_pairs(x)  # Same as forward (involution)

def inverse_encode_index_parity(x: str) -> str:
    """Inverse of encode_index_parity"""
    words = x.split()
    result = []
    for word in words:
        mid = math.ceil(len(word) / 2)
        evens = word[:mid]
        odds = word[mid:]
        reconstructed = []
        for i in range(len(word)):
            if i % 2 == 0:
                reconstructed.append(evens[i//2] if i//2 < len(evens) else '')
            else:
                reconstructed.append(odds[i//2] if i//2 < len(odds) else '')
        result.append(''.join(reconstructed))
    return ' '.join(result)

def inverse_double_consonants(x: str) -> str:
    """Remove doubled consonants"""
    vowels = 'aeiouAEIOU'
    result = []
    i = 0
    while i < len(x):
        result.append(x[i])
        if (i + 1 < len(x) and x[i] == x[i+1] and 
            x[i].isalpha() and x[i] not in vowels):
            i += 1  # Skip the duplicate
        i += 1
    return ''.join(result)

# Challenge 2: Coordinate pattern analysis
def analyze_coordinates(coordinates: List[List[str]]) -> str:
    """Extract hidden parameter from coordinate pattern"""
    # Convert to floats and filter outliers
    coords = [(float(lat), float(lng)) for lat, lng in coordinates]
    
    # Simple approach: look for pattern in decimal parts
    # This is a placeholder - actual implementation depends on the pattern
    decimal_pattern = []
    for lat, lng in coords:
        lat_dec = abs(lat) - int(abs(lat))
        lng_dec = abs(lng) - int(abs(lng))
        decimal_pattern.extend([lat_dec, lng_dec])
    
    # Convert decimals to digits (simplified)
    hidden_number = ''.join(str(int(d * 10)) for d in decimal_pattern)
    return hidden_number

# Challenge 3: Cipher decryption
def decrypt_railfence(text: str, rails: int = 3) -> str:
    """Decrypt rail fence cipher"""
    length = len(text)
    fence = [['\n' for _ in range(length)] for _ in range(rails)]
    rail = 0
    direction = 1
    
    # Mark positions with '*'
    for i in range(length):
        fence[rail][i] = '*'
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction = -direction
    
    # Fill the fence with cipher text
    index = 0
    for i in range(rails):
        for j in range(length):
            if fence[i][j] == '*' and index < length:
                fence[i][j] = text[index]
                index += 1
    
    # Read the plain text
    result = []
    rail = 0
    direction = 1
    for i in range(length):
        result.append(fence[rail][i])
        rail += direction
        if rail == 0 or rail == rails - 1:
            direction = -direction
    
    return ''.join(result)

def decrypt_keyword(text: str, keyword: str = "SHADOW") -> str:
    """Decrypt keyword substitution cipher"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Remove duplicates from keyword and create cipher alphabet
    key = ''.join(dict.fromkeys(keyword.upper()))
    cipher_alphabet = key + ''.join([c for c in alphabet if c not in key])
    
    # Create mapping from cipher to normal alphabet
    mapping = {cipher: normal for cipher, normal in zip(cipher_alphabet, alphabet)}
    
    result = []
    for char in text.upper():
        if char in mapping:
            result.append(mapping[char])
        else:
            result.append(char)
    
    return ''.join(result)

def decrypt_polybius(text: str) -> str:
    """Decrypt Polybius square cipher"""
    polybius_square = {
        'A': '11', 'B': '12', 'C': '13', 'D': '14', 'E': '15',
        'F': '21', 'G': '22', 'H': '23', 'I': '24', 'J': '24', 'K': '25',
        'L': '31', 'M': '32', 'N': '33', 'O': '34', 'P': '35',
        'Q': '41', 'R': '42', 'S': '43', 'T': '44', 'U': '45',
        'V': '51', 'W': '52', 'X': '53', 'Y': '54', 'Z': '55'
    }
    
    # Reverse mapping
    reverse_mapping = {v: k for k, v in polybius_square.items()}
    
    result = []
    # Split text into pairs of digits
    for i in range(0, len(text), 2):
        if i + 1 < len(text):
            pair = text[i:i+2]
            if pair in reverse_mapping:
                result.append(reverse_mapping[pair])
            else:
                result.append('?')
        else:
            result.append(text[i])
    
    return ''.join(result)

def parse_and_decrypt_log(log_entry: str) -> str:
    """Parse log entry and decrypt based on cipher type"""
    # Extract cipher type and encrypted payload
    cipher_match = re.search(r'CIPHER_TYPE: (\w+)', log_entry)
    payload_match = re.search(r'ENCRYPTED_PAYLOAD: (\w+)', log_entry)
    
    if not cipher_match or not payload_match:
        return "ERROR: Could not parse log entry"
    
    cipher_type = cipher_match.group(1)
    encrypted_payload = payload_match.group(1)
    
    # Decrypt based on cipher type
    if cipher_type == "RAILFENCE":
        return decrypt_railfence(encrypted_payload)
    elif cipher_type == "KEYWORD":
        return decrypt_keyword(encrypted_payload)
    elif cipher_type == "POLYBIUS":
        return decrypt_polybius(encrypted_payload)
    else:
        return f"ERROR: Unknown cipher type {cipher_type}"

# Challenge 4: Final decryption (placeholder - depends on previous results)
def decrypt_final_message(challenge1: str, challenge2: str, challenge3: str) -> str:
    """Combine all components for final decryption"""
    # This would be specific to the actual encryption scheme
    # For now, just combine them as a placeholder
    return f"{challenge1}_{challenge2}_{challenge3}"

@main_bp.route('/operation-safeguard', methods=['POST'])
def operation_safeguard():
    """
    POST endpoint for Operation Safeguard challenge
    """
    try:
        data = request.get_json()
        
        # Challenge 1: Reverse transformations
        transformations_str = data['challenge_one']['transformations']
        transformed_word = data['challenge_one']['transformed_encrypted_word']
        
        # Parse transformations list
        transformations = re.findall(r'(\w+)\(x\)', transformations_str)
        transformations.reverse()  # Apply in reverse order
        
        # Apply inverse transformations
        current_word = transformed_word
        for transform in transformations:
            if transform == 'mirror_words':
                current_word = inverse_mirror_words(current_word)
            elif transform == 'encode_mirror_alphabet':
                current_word = inverse_encode_mirror_alphabet(current_word)
            elif transform == 'toggle_case':
                current_word = inverse_toggle_case(current_word)
            elif transform == 'swap_pairs':
                current_word = inverse_swap_pairs(current_word)
            elif transform == 'encode_index_parity':
                current_word = inverse_encode_index_parity(current_word)
            elif transform == 'double_consonants':
                current_word = inverse_double_consonants(current_word)
        
        challenge1_result = current_word
        
        # Challenge 2: Coordinate pattern analysis
        coordinates = data['challenge_two']
        challenge2_result = analyze_coordinates(coordinates)
        
        # Challenge 3: Log decryption
        log_entry = data['challenge_three']
        challenge3_result = parse_and_decrypt_log(log_entry)
        
        # Challenge 4: Final decryption
        challenge4_result = decrypt_final_message(
            challenge1_result, challenge2_result, challenge3_result
        )
        
        response = {
            "challenge_one": challenge1_result,
            "challenge_two": challenge2_result,
            "challenge_three": challenge3_result,
            "challenge_four": challenge4_result
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 400

