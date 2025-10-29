from flask import request, jsonify
from app import db
from app.routes import admin_bp

@admin_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get admin stats"""
    return jsonify({"message": "Admin endpoint - implement logic"}), 501
