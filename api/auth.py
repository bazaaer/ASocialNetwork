from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    jwt_required,
    create_access_token,
    create_refresh_token,
    get_jwt_identity,
    get_jwt
)
import bcrypt
import os
from dotenv import load_dotenv

from db import db

# Load environment variables
load_dotenv()
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "adminpassword")

# Access the users collection
users_collection = db['users']

# Insert admin user if not exists
if not users_collection.find_one({"username": ADMIN_USERNAME}):
    hashed_password = bcrypt.hashpw(ADMIN_PASSWORD.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "username": ADMIN_USERNAME,
        "password": hashed_password,
        "role": "admin"
    })

# Auth Blueprint
auth_blueprint = Blueprint('auth', __name__)

@auth_blueprint.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    user = users_collection.find_one({"username": username})
    if not user or not bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return jsonify({'message': 'Invalid username or password'}), 401

    # Create access and refresh tokens
    additional_claims = {"role": user.get("role", "user")}
    access_token = create_access_token(identity=username, additional_claims=additional_claims)
    refresh_token = create_refresh_token(identity=username)

    return jsonify({
        'access_token': access_token,
        'refresh_token': refresh_token
    }), 200

@auth_blueprint.route('/api/refresh', methods=['POST'])
@jwt_required(refresh=True)
def refresh():
    current_user = get_jwt_identity()
    claims = get_jwt()
    
    # Create new access and refresh tokens
    new_access_token = create_access_token(identity=current_user, additional_claims={"role": claims.get("role", "user")})
    new_refresh_token = create_refresh_token(identity=current_user)
    
    return jsonify({
        'access_token': new_access_token,
        'refresh_token': new_refresh_token
    }), 200

@auth_blueprint.route('/api/create_user', methods=['POST'])
@jwt_required()
def create_user():
    current_user = get_jwt_identity()
    claims = get_jwt()
    if claims.get("role") != "admin":
        return jsonify({'message': 'Unauthorized'}), 403

    data = request.get_json()
    username = data.get('username')
    password = data.get('password')
    role = data.get('role', 'user')

    if not username or not password:
        return jsonify({'message': 'Username and password are required'}), 400

    if users_collection.find_one({"username": username}):
        return jsonify({'message': 'User already exists'}), 400

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    users_collection.insert_one({
        "username": username,
        "password": hashed_password,
        "role": role
    })

    return jsonify({'message': 'User created successfully'}), 201

@auth_blueprint.route('/api/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify({'message': f'Welcome {current_user}, this is a protected route!'}), 200

@auth_blueprint.route('/api/admin', methods=['GET'])
@jwt_required()
def admin_route():
    claims = get_jwt()
    if claims.get('role') != 'admin':
        return jsonify({'message': 'Admins only!'}), 403
    return jsonify({'message': 'Welcome, admin!'}), 200
