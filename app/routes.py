from flask import Blueprint, jsonify, request
from datetime import datetime

main_bp = Blueprint('main', __name__)

# Simple test route
@main_bp.route('/')
def home():
    return jsonify({
        'message': 'Welcome to Flask Backend!',
        'status': 'success',
        'timestamp': datetime.now().isoformat()
    })

# Example API route
@main_bp.route('/api/users', methods=['GET'])
def get_users():
    users = [
        {'id': 1, 'name': 'John Doe', 'email': 'john@example.com'},
        {'id': 2, 'name': 'Jane Smith', 'email': 'jane@example.com'}
    ]
    return jsonify(users)

# Route with parameter
@main_bp.route('/api/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = {'id': user_id, 'name': 'Example User', 'email': f'user{user_id}@example.com'}
    return jsonify(user)

# POST route example
@main_bp.route('/api/users', methods=['POST'])
def create_user():
    data = request.get_json()
    
    if not data or 'name' not in data or 'email' not in data:
        return jsonify({'error': 'Name and email are required'}), 400
    
    # Simulate creating a user
    new_user = {
        'id': 3,  # In real app, this would come from database
        'name': data['name'],
        'email': data['email']
    }
    
    return jsonify(new_user), 201