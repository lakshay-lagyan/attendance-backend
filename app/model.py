"""
Database models
"""

from datetime import datetime
from app import db

class Admin(db.Model):
    """Admin user model"""
    __tablename__ = 'admins'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    profile_image = db.Column(db.Text, default='')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'profile_image': self.profile_image,
            'role': 'admin'
        }


class User(db.Model):
    """Regular user model"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    department = db.Column(db.String(100), default='')
    phone = db.Column(db.String(20), default='')
    profile_image = db.Column(db.Text, default='')
    status = db.Column(db.String(20), default='active')  # active, blocked
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'department': self.department,
            'phone': self.phone,
            'profile_image': self.profile_image,
            'status': self.status,
            'role': 'user'
        }


class Person(db.Model):
    """Person with face embeddings"""
    __tablename__ = 'persons'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    embedding = db.Column(db.LargeBinary, nullable=False)
    embedding_dim = db.Column(db.Integer, nullable=False)
    photos_count = db.Column(db.Integer, default=0)
    status = db.Column(db.String(20), default='active')
    enrollment_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'photos_count': self.photos_count,
            'status': self.status,
            'enrollment_date': self.enrollment_date.isoformat() if self.enrollment_date else None
        }


class Profile(db.Model):
    """User profile information"""
    __tablename__ = 'profiles'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), default='')
    email = db.Column(db.String(120), nullable=False, index=True)
    phone = db.Column(db.String(20), default='')
    profile_image = db.Column(db.Text, default='')
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'department': self.department,
            'email': self.email,
            'phone': self.phone,
            'profile_image': self.profile_image,
            'registered_at': self.registered_at.isoformat() if self.registered_at else None
        }


class EnrollmentRequest(db.Model):
    """Enrollment request from users"""
    __tablename__ = 'enrollment_requests'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), nullable=False, index=True)
    phone = db.Column(db.String(20), default='')
    password_hash = db.Column(db.String(255), nullable=False)
    images = db.Column(db.JSON, nullable=False)
    status = db.Column(db.String(20), default='pending')  # pending, approved, rejected
    submitted_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime, nullable=True)
    processed_by = db.Column(db.String(120), nullable=True)
    rejection_reason = db.Column(db.Text, nullable=True)
    
    def to_dict(self, include_images=False):
        data = {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'status': self.status,
            'submitted_at': self.submitted_at.isoformat() if self.submitted_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'processed_by': self.processed_by,
            'rejection_reason': self.rejection_reason
        }
        if include_images:
            data['images'] = self.images
        return data


class Attendance(db.Model):
    """Attendance records"""
    __tablename__ = 'attendance'
    
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, index=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    confidence = db.Column(db.Float, default=0.0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'name': self.name,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'confidence': round(self.confidence, 3)
        }


# Create indexes
db.Index('idx_attendance_name_timestamp', Attendance.name, Attendance.timestamp.desc())
db.Index('idx_persons_status', Person.status)
db.Index('idx_users_status', User.status)
db.Index('idx_enrollment_status', EnrollmentRequest.status)