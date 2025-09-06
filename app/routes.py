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

@main_bp.route('/trivia', methods=['GET'])
def trivia():
    """
    GET endpoint that returns answers to the multiple choice trivia questions
    """
    answers = [
        1,  
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

# @main_bp.route('/blankety', methods=['POST'])
# def blankety():
#     data = request.get_json()
#     series_list = data['series']
#     result = []
    
#     for series in series_list:
#         n = len(series)
#         indices = np.arange(n)
#         # Extract known points
#         known_indices = []
#         known_values = []
#         for i, val in enumerate(series):
#             if val is not None:
#                 known_indices.append(i)
#                 known_values.append(val)
        
#         known_indices = np.array(known_indices)
#         known_values = np.array(known_values)
        
#         # If there are no known points, we cannot impute - but should not happen?
#         if len(known_indices) == 0:
#             # All null? Then fill with zeros? But should not happen.
#             imputed_series = [0.0] * n
#         elif len(known_indices) == 1:
#             # Only one point, fill constant
#             imputed_series = [known_values[0]] * n
#         else:
#             # Check if we have enough points for cubic spline
#             if len(known_indices) < 4:
#                 # Use linear interpolation
#                 # Create a linear spline with s=0?
#                 spline = UnivariateSpline(known_indices, known_values, k=1, s=0)
#             else:
#                 # Use cubic spline with smoothing
#                 # Choose s: let's use 0.5 * number of points
#                 s_val = 0.5 * len(known_indices)
#                 spline = UnivariateSpline(known_indices, known_values, k=3, s=s_val)
#             # Predict all indices
#             predicted = spline(indices)
#             # Create the completed series
#             imputed_series = []
#             for i, val in enumerate(series):
#                 if val is None:
#                     imputed_series.append(float(predicted[i]))
#                 else:
#                     imputed_series.append(val)
#         result.append(imputed_series)
    
#     return jsonify({'answer': result})

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
    stations = set()
    for s1, s2, c in edges:
        stations.add(s1)
        stations.add(s2)
    
    for task in tasks:
        stations.add(task[2])  # Add task stations
    
    stations.add(s0)  # Add starting station
    station_list = sorted(list(stations))
    station_to_idx = {station: i for i, station in enumerate(station_list)}
    n_stations = len(station_list)
    
    # Build graph
    graph = defaultdict(list)
    for s1, s2, c in edges:
        idx1, idx2 = station_to_idx[s1], station_to_idx[s2]
        graph[idx1].append((idx2, c))
        graph[idx2].append((idx1, c))
    
    # Precompute shortest distances from all stations
    all_distances = {}
    for station in station_list:
        idx = station_to_idx[station]
        all_distances[idx] = dijkstra(graph, idx, n_stations)
    
    # Sort tasks by end time, keeping track of original indices
    indexed_tasks = [(tasks[i][0], tasks[i][1], tasks[i][2], tasks[i][3], i) 
                     for i in range(len(tasks))]
    indexed_tasks.sort(key=lambda x: x[1])
    
    n = len(indexed_tasks)
    
    # dp[i] = (max_reward, min_fee, last_station, prev_task_index)
    dp = [(0, 0, s0, -1) for _ in range(n)]
    
    for i in range(n):
        start_time, end_time, station, reward, orig_idx = indexed_tasks[i]
        
        # Option 1: Take this task as the first task (from starting station)
        transport_cost = all_distances[station_to_idx[s0]][station_to_idx[station]]
        if transport_cost != float('inf'):
            dp[i] = (reward, transport_cost, station, -1)
        
        # Option 2: Take this task after some previous compatible task
        for j in range(i):
            prev_start, prev_end, prev_station, prev_reward, prev_orig_idx = indexed_tasks[j]
            
            # Check if tasks are compatible (previous task ends before current starts)
            if prev_end <= start_time:
                prev_total_reward, prev_total_fee, _, _ = dp[j]
                transport_cost = all_distances[station_to_idx[prev_station]][station_to_idx[station]]
                
                if transport_cost != float('inf'):
                    new_reward = prev_total_reward + reward
                    new_fee = prev_total_fee + transport_cost
                    
                    # Check if this is better than current best for task i
                    curr_reward, curr_fee, _, _ = dp[i]
                    if (new_reward > curr_reward or 
                        (new_reward == curr_reward and new_fee < curr_fee)):
                        dp[i] = (new_reward, new_fee, station, j)
    
    # Find the task with maximum reward (and minimum fee if tied)
    # Include return cost to starting station
    max_reward = 0
    min_fee = float('inf')
    best_task = -1
    
    for i in range(n):
        reward, fee, last_station, _ = dp[i]
        # Add return cost to starting station
        return_cost = all_distances[station_to_idx[last_station]][station_to_idx[s0]]
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
        _, _, _, prev_task = dp[current]
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



@main_bp.route('/trading-formula', methods=['POST'])
def trading_formula():
    data = request.get_json()
    results = []
    
    for test_case in data:
        formula = test_case['formula']
        variables = test_case['variables']
        
        try:
            # Preprocess LaTeX formula
            expression = preprocess_latex(formula)
            
            # Substitute variable values
            expression = substitute_variables(expression, variables)
            
            # Evaluate safely
            result = safe_eval(expression)
            
            # Round to 4 decimal places
            result = round(result, 4)
            
            results.append({"result": result})
            
        except Exception as e:
            # In case of error, return 0 or handle appropriately
            results.append({"result": 0.0})
    
    return jsonify(results)

def preprocess_latex(latex_formula):
    """Convert LaTeX formula to Python-compatible expression"""
    expression = latex_formula
    
    # Remove $$ wrappers if present
    expression = re.sub(r'^\$\$(.*?)\$\$$', r'\1', expression)
    
    # Remove assignment part if present (e.g., "Fee = ")
    expression = re.sub(r'^[^=]*=', '', expression).strip()
    
    # Replace LaTeX commands
    expression = re.sub(r'\\text{([^}]*)}', r'\1', expression)
    expression = re.sub(r'\\frac{([^}]*)}{([^}]*)}', r'(\1)/(\2)', expression)
    expression = re.sub(r'\\cdot', '*', expression)
    expression = re.sub(r'\\times', '*', expression)
    expression = re.sub(r'\\max\(', 'max(', expression)
    expression = re.sub(r'\\min\(', 'min(', expression)
    expression = re.sub(r'\\log\(', 'math.log(', expression)
    expression = re.sub(r'\\ln\(', 'math.log(', expression)
    expression = re.sub(r'e\^{([^}]*)}', r'math.exp(\1)', expression)
    expression = re.sub(r'\\sum', 'sum', expression)
    
    # Handle special characters and formatting
    expression = re.sub(r'\\', '', expression)
    expression = re.sub(r'\s+', '', expression)
    
    return expression

def substitute_variables(expression, variables):
    """Replace variable names with their values"""
    for var_name, value in variables.items():
        # Handle variables with underscores and special characters
        safe_var_name = re.escape(var_name)
        expression = re.sub(r'\b' + safe_var_name + r'\b', str(value), expression)
    return expression

def safe_eval(expression):
    """Safely evaluate mathematical expression"""
    safe_dict = {
        'math': math,
        'max': max,
        'min': min,
        'sum': sum,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'pi': math.pi,
        'e': math.e
    }
    
    # Remove any potentially dangerous characters
    safe_expression = re.sub(r'[^a-zA-Z0-9_+\-*/()., ]', '', expression)
    
    try:
        return eval(safe_expression, {"__builtins__": None}, safe_dict)
    except:
        try:
            return eval(safe_expression)
        except:
            raise ValueError(f"Could not evaluate expression: {expression}")