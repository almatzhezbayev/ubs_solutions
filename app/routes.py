from collections import defaultdict
import heapq
from flask import Blueprint, jsonify, request
from datetime import datetime
import numpy as np
from scipy.interpolate import UnivariateSpline
import math

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
        1,  # 1. "Trivia!": How many challenges are there this year, which title ends with an exclamation mark? -> 4
        1,  # 2. "Ticketing Agent": What type of tickets is the ticketing agent handling? -> Museum
        2,  # 3. "Blankety Blanks": How many lists and elements per list are included? -> 100 lists x 1000 elements
        2,  # 4. "Princess Diaries": What's Princess Mia's cat name? -> Fat Louie
        4,  # 5. "MST Calculation": What is the average number of nodes? -> 10
        4,  # 6. "Universal Bureau of Surveillance": Which singer did not have a James Bond theme? -> Amy Winehouse
        3,  # 7. "Operation Safeguard": What is the smallest font size? -> 2px
        4,  # 8. "Capture The Flag": Which of these are anagrams? -> graft cute leapt
        2   # 9. "Filler 1": Where has UBS Global Coding Challenge been held? -> Australia, Hong Kong, Japan, Singapore
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

@main_bp.route('/blankety', methods=['POST'])
def blankety():
    data = request.get_json()
    series_list = data['series']
    result = []
    
    for series in series_list:
        n = len(series)
        indices = np.arange(n)
        # Extract known points
        known_indices = []
        known_values = []
        for i, val in enumerate(series):
            if val is not None:
                known_indices.append(i)
                known_values.append(val)
        
        known_indices = np.array(known_indices)
        known_values = np.array(known_values)
        
        # If there are no known points, we cannot impute - but should not happen?
        if len(known_indices) == 0:
            # All null? Then fill with zeros? But should not happen.
            imputed_series = [0.0] * n
        elif len(known_indices) == 1:
            # Only one point, fill constant
            imputed_series = [known_values[0]] * n
        else:
            # Check if we have enough points for cubic spline
            if len(known_indices) < 4:
                # Use linear interpolation
                # Create a linear spline with s=0?
                spline = UnivariateSpline(known_indices, known_values, k=1, s=0)
            else:
                # Use cubic spline with smoothing
                # Choose s: let's use 0.5 * number of points
                s_val = 0.5 * len(known_indices)
                spline = UnivariateSpline(known_indices, known_values, k=3, s=s_val)
            # Predict all indices
            predicted = spline(indices)
            # Create the completed series
            imputed_series = []
            for i, val in enumerate(series):
                if val is None:
                    imputed_series.append(float(predicted[i]))
                else:
                    imputed_series.append(val)
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

def b_search(tasks, i):
    left, right = 0, i - 1
    result = -1
    
    while left <= right:
        mid = (left + right) // 2
        if tasks[mid][1] <= tasks[i][0]:
            result = mid
            left = mid + 1
        else:
            right = mid - 1
    
    return result

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
    

    indexed_tasks = [(tasks[i][0], tasks[i][1], tasks[i][2], tasks[i][3], i) 
                     for i in range(len(tasks))]
    indexed_tasks.sort(key=lambda x: x[1])
    
    n = len(indexed_tasks)
    
    dp = [(0, 0, [s0])] * (n + 1)
    
    for i in range(1, n + 1):
        _, _, station, reward, _ = indexed_tasks[i-1]
        dp[i] = dp[i-1]
        last = b_search(indexed_tasks, i-1)

        if last == -1:
            prev_reward = 0
            prev_fee = 0
            prev_stations = [s0]
        else:
            prev_reward, prev_fee, prev_stations = dp[last+1]

    best_new_reward = -1
    best_new_fee = float('inf')
        
    for prev_station in prev_stations:
        # Transport cost from prev_station to current task station
        transport_cost = all_distances[station_to_idx[prev_station]][station_to_idx[station]]
        
        if transport_cost != float('inf'):
            new_reward = prev_reward + reward
            new_fee = prev_fee + transport_cost
            
            if (new_reward > best_new_reward or 
                (new_reward == best_new_reward and new_fee < best_new_fee)):
                best_new_reward = new_reward
                best_new_fee = new_fee
    
    curr_reward, curr_fee, curr_stations = dp[i]
    
    if best_new_reward != -1:
        if (best_new_reward > curr_reward or 
            (best_new_reward == curr_reward and best_new_fee < curr_fee)):
            dp[i] = (best_new_reward, best_new_fee, [station])
        elif (best_new_reward == curr_reward and best_new_fee == curr_fee):
            # Same reward and fee, add to end stations list
            if station not in curr_stations:
                new_stations = curr_stations + [station]
                dp[i] = (curr_reward, curr_fee, new_stations)

    max_reward, min_fee, end_stations = dp[n]
    
    # Add the cost of returning
    if end_stations:
        s0_idx = station_to_idx[s0]
        min_return_cost = float('inf')
        
        for end_station in end_stations:
            end_station_idx = station_to_idx[end_station]
            return_cost = all_distances[end_station_idx][s0_idx]
            if return_cost < min_return_cost:
                min_return_cost = return_cost
        
        if min_return_cost != float('inf'):
            min_fee += min_return_cost

    selected_tasks = reconstruct_solution(dp, indexed_tasks, n)
    
    return max_reward, min_fee, selected_tasks

def reconstruct_solution(dp, indexed_tasks, n):
    selected = []
    i = n
    
    while i > 0:
        curr_reward, curr_fee, _ = dp[i]
        prev_reward, prev_fee, _ = dp[i-1]
        
        if curr_reward != prev_reward or curr_fee != prev_fee:
            selected.append(indexed_tasks[i-1][4])
            j = b_search(indexed_tasks, i-1)
            i = j + 1 if j != -1 else 0
        else:
            i -= 1
    
    return sorted(selected)

@main_bp.route('/princess-diaries', methods=['POST'])
def princess_diaries():
    try:
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
            s1, s2 = connection['connection']
            fee = connection['fee']
            edges.append([s1, s2, fee])
        
        max_reward, min_fee, selected_tasks = solve_subway_scheduling(edges, tasks, starting_station)
        
        selected_with_start_time = [(i, tasks[i][0]) for i in selected_tasks]  # (index, start_time)
        selected_with_start_time.sort(key=lambda x: x[1])  # Sort by start time
        schedule = [task_names[i] for i, _ in selected_with_start_time]
        
        return jsonify({
            "max_score": max_reward,
            "min_fee": min_fee,
            "schedule": schedule
        })
        
    except Exception as e:
        return jsonify({
            "error": str(e),
            "max_score": 0,
            "min_fee": 0,
            "schedule": []
        }), 400