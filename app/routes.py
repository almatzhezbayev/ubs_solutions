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