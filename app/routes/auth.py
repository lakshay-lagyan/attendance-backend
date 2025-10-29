from flask import request, jsonify, session
from werkzeug.security import check_password_hash, generate_password_hash
from app import db, limiter
from app.models import Admin, User  
from app.routes import auth_bp  

@auth_bp.route('/login', methods=['POST'])
@limiter.limit("10 per minute")
def login():
    """User login"""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    # Check admin
    admin = Admin.query.filter_by(email=email).first()
    if admin and check_password_hash(admin.password_hash, password):
        session['user_id'] = admin.id
        session['user_role'] = 'admin'
        return jsonify({
            "success": True,
            "user": admin.to_dict()
        })
    
    # Check regular user
    user = User.query.filter_by(email=email).first()
    if user:
        if user.status == 'blocked':
            return jsonify({"error": "Account blocked"}), 403
        
        if check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['user_role'] = 'user'
            return jsonify({
                "success": True,
                "user": user.to_dict()
            })
    
    return jsonify({"error": "Invalid credentials"}), 401


@auth_bp.route('/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})


@auth_bp.route('/me', methods=['GET'])
def get_current_user():
    """Get current authenticated user"""
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    
    if not user_id or not user_role:
        return jsonify({"error": "Not authenticated"}), 401
    
    if user_role == 'admin':
        user = Admin.query.get(user_id)
    else:
        user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    return jsonify({"user": user.to_dict()})


@auth_bp.route('/change-password', methods=['POST'])
def change_password():
    """Change user password"""
    user_id = session.get('user_id')
    user_role = session.get('user_role')
    
    if not user_id:
        return jsonify({"error": "Not authenticated"}), 401
    
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not current_password or not new_password:
        return jsonify({"error": "Current and new passwords required"}), 400
    
    if len(new_password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    
    # Get user
    if user_role == 'admin':
        user = Admin.query.get(user_id)
    else:
        user = User.query.get(user_id)
    
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Verify current password
    if not check_password_hash(user.password_hash, current_password):
        return jsonify({"error": "Current password incorrect"}), 401
    
    # Update password
    user.password_hash = generate_password_hash(new_password)
    db.session.commit()
    
    return jsonify({"success": True, "message": "Password changed successfully"})