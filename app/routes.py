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

#######################################---TRADING---#############################################
def preprocess_latex(latex_formula):
    """Convert LaTeX formula to Python-compatible expression"""
    expression = latex_formula.strip()
    
    # Remove $$ wrappers if present
    if expression.startswith('$$') and expression.endswith('$$'):
        expression = expression[2:-2].strip()
    
    # Remove assignment part if present (e.g., "Fee = ")
    if '=' in expression:
        parts = expression.split('=', 1)
        if len(parts) > 1:
            expression = parts[1].strip()
    
    # Step 1: Handle summations first (most complex)
    expression = handle_summations(expression)
    
    # Step 2: Replace other LaTeX commands
    expression = re.sub(r'\\text{([^}]*)}', r'\1', expression)
    expression = re.sub(r'\\frac{([^}]*)}{([^}]*)}', r'(\1)/(\2)', expression)
    expression = re.sub(r'\\cdot', '*', expression)
    expression = re.sub(r'\\times', '*', expression)
    expression = re.sub(r'\\max\(', 'max(', expression)
    expression = re.sub(r'\\min\(', 'min(', expression)
    expression = re.sub(r'\\log\(', 'math.log(', expression)
    expression = re.sub(r'\\ln\(', 'math.log(', expression)
    
    # Handle exponential notation carefully
    expression = re.sub(r'e\^{([^}]*)}', r'math.exp(\1)', expression)
    expression = re.sub(r'(\d+)e\^{([^}]*)}', r'\1*math.exp(\2)', expression)
    
    # Handle square roots and powers
    expression = re.sub(r'\\sqrt{([^}]*)}', r'math.sqrt(\1)', expression)
    expression = re.sub(r'\\sqrt\[([^\]]*)\]{(.*?)}', r'(\2)**(1/\1)', expression)
    expression = re.sub(r'(\w+)\^\{?([^}]*)\}?', r'\1**\2', expression)
    
    # Remove remaining backslashes and whitespace
    expression = re.sub(r'\\', '', expression)
    expression = re.sub(r'\s+', '', expression)
    
    # Ensure proper operator spacing for evaluation
    expression = re.sub(r'([a-zA-Z_])([0-9\(])', r'\1*\2', expression)  # var*number
    expression = re.sub(r'([0-9\)])([a-zA-Z_])', r'\1*\2', expression)  # number*var
    
    return expression

def handle_summations(expression):
    """Handle summation notation \sum_{i=1}^{n} expression"""
    # Pattern for \sum_{start}^{end} expression
    pattern = r'\\sum_{([^}]*)}\^{([^}]*)}[\s]*([^\s]*)'
    
    def replace_sum(match):
        var = match.group(1)
        end = match.group(2)
        expr = match.group(3)
        
        # Simple case: \sum_{i=1}^{n} i → sum(i for i in range(1, n+1))
        if var.isalpha() and end.isdigit():
            return f'sum({var} for {var} in range(1, {int(end)+1}))'
        elif '=' in var:
            # More complex: \sum_{i=1}^{n} → range from start to end
            parts = var.split('=')
            if len(parts) == 2:
                var_name = parts[0].strip()
                start_val = parts[1].strip()
                return f'sum({expr} for {var_name} in range({start_val}, {int(end)+1}))'
        
        return f'sum({expr})'  # Fallback
    
    return re.sub(pattern, replace_sum, expression)

def substitute_variables(expression, variables):
    """Replace variable names with their values, handling complex names"""
    # Sort variables by length (longest first) to avoid partial matches
    sorted_vars = sorted(variables.keys(), key=len, reverse=True)
    
    for var_name in sorted_vars:
        value = variables[var_name]
        # Create regex pattern that matches the whole variable name
        pattern = r'\b' + re.escape(var_name) + r'\b'
        expression = re.sub(pattern, str(value), expression)
    
    return expression

def safe_eval(expression):
    """Safely evaluate mathematical expression with comprehensive function support"""
    # Enhanced safe dictionary
    safe_dict = {
        'math': math,
        'max': max,
        'min': min,
        'sum': sum,
        'abs': abs,
        'exp': math.exp,
        'log': math.log,
        'log10': math.log10,
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'asin': math.asin,
        'acos': math.acos,
        'atan': math.atan,
        'sinh': math.sinh,
        'cosh': math.cosh,
        'tanh': math.tanh,
        'pi': math.pi,
        'e': math.e,
        'inf': float('inf'),
        'Infinity': float('inf')
    }
    
    # Add basic arithmetic functions
    safe_dict.update({
        '__builtins__': None,
        'True': True,
        'False': False,
        'None': None
    })
    
    try:
        # Use ast.literal_eval for safer evaluation
        import ast
        
        # First try to parse as a literal
        try:
            return ast.literal_eval(expression)
        except (ValueError, SyntaxError):
            # If not a simple literal, use eval with safe context
            # Clean the expression
            clean_expr = re.sub(r'[^a-zA-Z0-9_+\-*/().,=<>!&|^% ]', '', expression)
            
            # Handle special cases for financial formulas
            clean_expr = clean_expr.replace('^', '**')
            
            return eval(clean_expr, safe_dict)
            
    except Exception as e:
        # Fallback: try basic evaluation with error handling
        try:
            # Handle common financial notation
            if 'E[' in expression or 'E(' in expression:
                # Expected value notation - treat as variable
                clean_expr = expression.replace('E[', 'E_').replace('E(', 'E_')
                clean_expr = clean_expr.replace(']', '').replace(')', '')
                return eval(clean_expr, safe_dict)
            return float(expression)
        except:
            raise ValueError(f"Evaluation failed for: {expression}")

# Helper function for complex summation cases
def handle_complex_summation(expr, variables):
    """Handle more complex summation cases that require variable substitution"""
    # This would need to be expanded based on specific test cases
    # For now, provide a basic implementation
    if 'sum' in expr:
        # Simple range-based summation
        match = re.search(r'sum\(([^\)]+)\)', expr)
        if match:
            sum_expr = match.group(1)
            # Try to evaluate the sum expression
            try:
                return safe_eval(sum_expr)
            except:
                return 0
    return expr   

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
            
            # Round to 4 decimal places (handle very small numbers)
            if abs(result) < 1e-10:
                result = 0.0
            result = round(float(result), 4)
            
            results.append({"result": result})
            
        except Exception as e:
            print(f"Error evaluating {formula}: {e}")
            results.append({"result": 0.0})
    
    return jsonify(results)

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