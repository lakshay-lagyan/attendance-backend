from flask import request, jsonify
from app import db, limiter
from app.routes import face_bp
from app.services.face_recognition import face_recognition_service
from app.services.crowd_recognition import crowd_recognition_service
from app.utils.logger import logger
import numpy as np

@face_bp.route('/recognize', methods=['POST'])
@limiter.limit("30 per minute")
def recognize():
    """Recognize single face"""
    try:
        # Check if image file present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Validate image
        is_valid, error_msg = face_recognition_service.validate_image(image_file)
        if not is_valid:
            return jsonify({"error": error_msg}), 400
        
        # Preprocess image
        image = face_recognition_service.preprocess_image(image_file)
        if image is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Extract embedding
        embedding = face_recognition_service.extract_embedding(image)
        if embedding is None:
            return jsonify({"error": "Failed to extract face embedding"}), 400
        
        # Search for match
        from app.services.faiss_service import faiss_service
        results = faiss_service.search(embedding, threshold=0.6, top_k=1)
        
        if results and results[0][0] is not None:
            name, similarity = results[0]
            return jsonify({
                "recognized": True,
                "name": name,
                "similarity": round(similarity, 3),
                "confidence": "high" if similarity >= 0.7 else "medium"
            }), 200
        else:
            return jsonify({
                "recognized": False,
                "message": "Face not recognized"
            }), 200
        
    except Exception as e:
        logger.error(f"Face recognition error: {e}")
        return jsonify({"error": "Recognition failed"}), 500


@face_bp.route('/recognize/crowd', methods=['POST'])
@limiter.limit("10 per minute")
def recognize_crowd():
    """
    Recognize multiple faces in crowded scene
    
    Body:
        - image: Image file with multiple faces
        - enable_tracking: (optional) Enable face tracking for video
        - cooldown: (optional) Cooldown seconds between recognitions (default: 5)
    """
    try:
        # Check if image file present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files['image']
        
        # Get optional parameters
        enable_tracking = request.form.get('enable_tracking', 'false').lower() == 'true'
        cooldown = int(request.form.get('cooldown', 5))
        
        # Update cooldown if provided
        if cooldown != 5:
            crowd_recognition_service.set_cooldown(cooldown)
        
        # Preprocess image
        image = face_recognition_service.preprocess_image(image_file)
        if image is None:
            return jsonify({"error": "Failed to process image"}), 400
        
        # Process crowd
        result = crowd_recognition_service.process_crowd_image(
            image, 
            enable_tracking=enable_tracking
        )
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Crowd recognition error: {e}")
        return jsonify({"error": "Crowd recognition failed"}), 500


@face_bp.route('/recognize/reset-cache', methods=['POST'])
def reset_recognition_cache():
    """Reset recognition cache (clear duplicate detection)"""
    try:
        crowd_recognition_service.reset_cache()
        return jsonify({"message": "Recognition cache cleared"}), 200
    except Exception as e:
        logger.error(f"Cache reset error: {e}")
        return jsonify({"error": "Failed to reset cache"}), 500

