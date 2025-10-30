import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import Admin, User, Person, Profile, EnrollmentRequest, Attendance
from werkzeug.security import generate_password_hash

def init_database():
    """Initialize database with all tables"""
    
    print("🚀 Initializing database...")
    
    # Create app
    app = create_app('production')
    
    with app.app_context():
        
        # Create all tables
        print("📋 Creating tables...")
        db.create_all()
        
        # Verify tables were created
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        
        print(f"✅ Created {len(tables)} tables:")
        for table in tables:
            print(f"   - {table}")
        
        # Create default admin if none exists
        admin_count = Admin.query.count()
        if admin_count == 0:
            print("\n👤 Creating default admin account...")
            
            default_admin = Admin(
                name="Admin",
                email="admin@attendance.com",
                password_hash=generate_password_hash("admin123")
            )
            
            db.session.add(default_admin)
            db.session.commit()
            
            print("✅ Default admin created:")
            print(f"   Email: admin@attendance.com")
            print(f"   Password: admin123")
            print(f"   ⚠️  CHANGE THIS PASSWORD IMMEDIATELY!")
        else:
            print(f"\n✅ {admin_count} admin(s) already exist")
        
        # Show table counts
        print("\n📊 Database Statistics:")
        print(f"   Admins: {Admin.query.count()}")
        print(f"   Users: {User.query.count()}")
        print(f"   Persons: {Person.query.count()}")
        print(f"   Profiles: {Profile.query.count()}")
        print(f"   Enrollment Requests: {EnrollmentRequest.query.count()}")
        print(f"   Attendance Records: {Attendance.query.count()}")
        
        print("\n✅ Database initialization complete!")
        return True

if __name__ == '__main__':
    try:
        success = init_database()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)