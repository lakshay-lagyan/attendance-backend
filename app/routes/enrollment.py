import logging
from flask import request, jsonify
from app import db, limiter
from app.routes import enrollment_bp
from app.services.advanced_enrollment import advanced_enrollment_service
from werkzeug.utils import secure_filename
import os

logger = logging.getLogger(__name__)

@enrollment_bp.route('/enroll', methods=['POST'])
@limiter.limit("3 per hour")
def enroll_user():
    
    try:
        # Validate form data
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if not name or not email or not password:
            return jsonify({
                "success": False,
                "message": "Missing required fields: name, email, password"
            }), 400
        
        # Get uploaded files
        files = request.files.getlist('files')
        
        if not files:
            return jsonify({
                "success": False,
                "message": "No images uploaded"
            }), 400
        
        if len(files) < 3:
            return jsonify({
                "success": False,
                "message": f"Minimum 3 images required. Provided: {len(files)}"
            }), 400
        
        logger.info(f"Enrollment request for {name} with {len(files)} images")
        
        # Prepare user data
        user_data = {
            'email': email,
            'password': password,
            'department': request.form.get('department', ''),
            'phone': request.form.get('phone', '')
        }
        
        # Process enrollment with advanced service
        result = advanced_enrollment_service.process_enrollment(
            images=files,
            name=name,
            user_data=user_data
        )
        
        if result['success']:
            return jsonify(result), 200
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"Enrollment endpoint error: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "message": f"Enrollment failed: {str(e)}"
        }), 500


@enrollment_bp.route('/validate-image', methods=['POST'])
@limiter.limit("30 per minute")
def validate_image():
   
    try:
        if 'image' not in request.files:
            return jsonify({
                "valid": False,
                "message": "No image provided"
            }), 400
        
        image_file = request.files['image']
        
        # Preprocess image
        from app.services.face_recognition import face_recognition_service
        import cv2 as cv
        import numpy as np
        
        image = face_recognition_service.preprocess_image(image_file)
        
        if image is None:
            return jsonify({
                "valid": False,
                "quality_score": 0.0,
                "issues": ["Invalid image format"],
                "metrics": {}
            }), 200
        
        # Detect face
        from deepface import DeepFace
        faces = DeepFace.extract_faces(
            img_path=image,
            detector_backend='retinaface',
            enforce_detection=False,
            align=False
        )
        
        issues = []
        metrics = {}
        
        if not faces:
            issues.append("No face detected")
            return jsonify({
                "valid": False,
                "quality_score": 0.0,
                "issues": issues,
                "metrics": metrics
            }), 200
        
        if len(faces) > 1:
            issues.append(f"Multiple faces detected ({len(faces)})")
        
        face_data = faces[0]
        facial_area = face_data.get('facial_area', {})
        w = facial_area.get('w', 0)
        h = facial_area.get('h', 0)
        
        # Extract face region
        x = facial_area.get('x', 0)
        y = facial_area.get('y', 0)
        face_region = image[y:y+h, x:x+w]
        
        # Calculate quality metrics
        gray = cv.cvtColor(face_region, cv.COLOR_BGR2GRAY)
        
        # Sharpness
        laplacian_var = cv.Laplacian(gray, cv.CV_64F).var()
        sharpness = min(laplacian_var / 150.0, 1.0)
        
        # Brightness
        brightness_val = np.mean(gray)
        brightness = 1.0 - abs(brightness_val - 130) / 130.0
        brightness = max(0, brightness)
        
        # Contrast
        contrast_val = np.std(gray)
        contrast = min(contrast_val / 50.0, 1.0)
        
        # Overall quality
        quality_score = (sharpness * 0.4 + brightness * 0.3 + contrast * 0.3)
        
        # Check thresholds
        if w < 80 or h < 80:
            issues.append(f"Face too small ({w}x{h}). Minimum 80x80 pixels")
        if sharpness < 0.4:
            issues.append("Image is blurry. Please ensure good focus")
        if brightness < 0.5:
            issues.append("Poor lighting. Adjust brightness")
        if contrast < 0.3:
            issues.append("Low contrast. Improve lighting conditions")
        
        metrics = {
            "sharpness": round(sharpness, 3),
            "brightness": round(brightness, 3),
            "contrast": round(contrast, 3),
            "face_detected": True,
            "face_size": [w, h],
            "face_confidence": round(face_data.get('confidence', 0), 3)
        }
        
        valid = quality_score >= 0.50 and len(issues) <= 1  # Allow minor issues
        
        return jsonify({
            "valid": valid,
            "quality_score": round(quality_score, 3),
            "issues": issues,
            "metrics": metrics
        }), 200
        
    except Exception as e:
        logger.error(f"Image validation error: {e}")
        return jsonify({
            "valid": False,
            "quality_score": 0.0,
            "issues": [f"Validation error: {str(e)}"],
            "metrics": {}
        }), 500


@enrollment_bp.route('/request', methods=['POST'])
@limiter.limit("5 per hour")
def submit_request():
    
    return jsonify({
        "message": "Please use /api/enrollment/enroll endpoint for direct enrollment",
        "redirect": "/api/enrollment/enroll"
    }), 200

