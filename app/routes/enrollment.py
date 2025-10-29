from flask import request, jsonify
from app import db, limiter
from app.routes import enrollment_bp

@enrollment_bp.route('/request', methods=['POST'])
@limiter.limit("5 per hour")
def submit_request():
    """Submit enrollment request"""
    return jsonify({"message": "Enrollment endpoint - implement logic"}), 501

