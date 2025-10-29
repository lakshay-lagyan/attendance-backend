from flask import request, jsonify
from app import db, limiter
from app.routes import face_bp

@face_bp.route('/recognize', methods=['POST'])
@limiter.limit("30 per minute")
def recognize():
    """Recognize face"""
    return jsonify({"message": "Face recognition endpoint - implement logic"}), 501

