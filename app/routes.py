from collections import defaultdict
import heapq
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
def find_cycle_edges(network):
    """
    Find all edges that are part of the single cycle in the network.
    Since there's only one cycle (tree + one edge), we find it using DFS.
    """
    if not network:
        return []
    
    # Build adjacency list and spy mapping
    spy_to_id = {}
    spy_id = 0
    graph = defaultdict(list)
    edge_map = {}  # Maps (id1, id2) to original edge
    
    for connection in network:
        spy1, spy2 = connection['spy1'], connection['spy2']
        
        if spy1 not in spy_to_id:
            spy_to_id[spy1] = spy_id
            spy_id += 1
        if spy2 not in spy_to_id:
            spy_to_id[spy2] = spy_id
            spy_id += 1
        
        id1, id2 = spy_to_id[spy1], spy_to_id[spy2]
        graph[id1].append(id2)
        graph[id2].append(id1)
        
        # Store edge mapping (ensure consistent ordering)
        key = (min(id1, id2), max(id1, id2))
        edge_map[key] = connection
    
    # DFS to find cycle
    visited = [False] * spy_id
    parent = [-1] * spy_id
    cycle_edges = []
    
    def dfs(node, par):
        visited[node] = True
        parent[node] = par
        
        for neighbor in graph[node]:
            if neighbor == par:  # Skip the edge we came from
                continue
            
            if visited[neighbor]:
                # Found back edge - this creates the cycle
                # Reconstruct cycle path
                cycle_path = []
                current = node
                while current != neighbor:
                    cycle_path.append(current)
                    current = parent[current]
                cycle_path.append(neighbor)
                cycle_path.append(node)  # Complete the cycle
                
                # Convert cycle path to edges
                for i in range(len(cycle_path)):
                    id1 = cycle_path[i]
                    id2 = cycle_path[(i + 1) % len(cycle_path)]
                    key = (min(id1, id2), max(id1, id2))
                    if key in edge_map:
                        cycle_edges.append(edge_map[key])
                
                return True
            else:
                if dfs(neighbor, node):
                    return True
        
        return False
    
    # Start DFS from any node
    for start_node in range(spy_id):
        if not visited[start_node]:
            if dfs(start_node, -1):
                break
    
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