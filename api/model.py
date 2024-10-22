from flask import Blueprint

# Create a blueprint
api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/api', methods=['GET'])
def index():
    return 'ASocialNetwork API is running!'

@api_blueprint.route('/api/post', methods=['POST'])
def create_post():
    return 'Create post'