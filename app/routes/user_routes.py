from flask import request, jsonify
from app import db
from app.routes import user_bp

@user_bp.route('/profile', methods=['GET'])
def get_profile():
    """Get user profile"""
    return jsonify({"message": "User profile endpoint"}), 501


    