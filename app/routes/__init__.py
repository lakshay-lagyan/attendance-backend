from flask import Blueprint

# Create blueprints
auth_bp = Blueprint('auth', __name__)
enrollment_bp = Blueprint('enrollment', __name__)
attendance_bp = Blueprint('attendance', __name__)
admin_bp = Blueprint('admin', __name__)
face_bp = Blueprint('face', __name__)
user_bp = Blueprint('user', __name__)

# Import routes to register them
from app.routes import auth
from app.routes import enrollment  
from app.routes import attendance
from app.routes import admin
from app.routes import face
from app.routes import user_routes

__all__ = [
    'auth_bp',
    'enrollment_bp', 
    'attendance_bp',
    'admin_bp',
    'face_bp',
    'user_bp'
]



