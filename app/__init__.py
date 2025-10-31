import logging
from flask import Flask, jsonify
from flask_compress import Compress
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from flask_jwt_extended import JWTManager
# from flask_session import Session  # Temporarily commented - will add after testing
from werkzeug.middleware.proxy_fix import ProxyFix

from config import config

logger = logging.getLogger(__name__)

# Initialize extensions
db = SQLAlchemy()
migrate = Migrate()
compress = Compress()
jwt = JWTManager()
# sess = Session()  # Temporarily commented - using Flask's built-in session
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per hour"],
    storage_uri="memory://"
)

def create_app(config_name='production'):
    
    # Ensure valid config name
    if config_name not in config:
        logger.warning(f"Invalid config name '{config_name}', using 'production'")
        config_name = 'production'
    
    app = Flask(__name__)
    app.config.from_object(config[config_name])
    
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
    
    # Initialize extensions
    db.init_app(app)
    migrate.init_app(app, db)
    compress.init_app(app)
    jwt.init_app(app)
    # sess.init_app(app)  # Temporarily commented - using Flask's built-in session
    
    if app.config.get('REDIS_URL'):
        limiter.storage_uri = app.config['REDIS_URL']
    limiter.init_app(app)
    
    # Configure CORS for frontend access
    frontend_urls = app.config.get('CORS_ORIGINS', ['https://attendance-frontend-p3xd.onrender.com'])
    CORS(app, 
         resources={r"/api/*": {"origins": frontend_urls}},
         supports_credentials=True,
         allow_headers=["Content-Type", "Authorization"],
         methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         expose_headers=["Content-Type"],
         max_age=3600
    )
    
    # Initialize services
    with app.app_context():
        from app.services.faiss_service import faiss_service
        from app.services.cache_service import cache_service
        
        # Initialize FAISS index
        try:
            faiss_service.initialize()
            logger.info("[OK] FAISS service initialized")
        except Exception as e:
            logger.error(f"[ERROR] FAISS initialization failed: {e}")
        
        # Initialize cache
        try:
            cache_service.initialize(app.config.get('REDIS_URL'))
            logger.info("[OK] Cache service initialized")
        except Exception as e:
            logger.warning(f"[WARNING] Cache service unavailable: {e}")
    
    # Register blueprints (API only)
    register_blueprints(app)
    
    # Register error handlers
    register_error_handlers(app)
    
    # Register health check
    register_health_check(app)
    
    # Register CLI commands
    register_commands(app)
    
    logger.info(f"[OK] API-only application created: {config_name} mode")
    
    return app


def register_blueprints(app):
    """Register API blueprints only"""
    from app.routes import (
        auth_bp, enrollment_bp, attendance_bp, 
        admin_bp, face_bp, user_bp
    )
    
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(enrollment_bp, url_prefix='/api/enrollment')
    app.register_blueprint(attendance_bp, url_prefix='/api/attendance')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    app.register_blueprint(face_bp, url_prefix='/api/face')
    app.register_blueprint(user_bp, url_prefix='/api/user')
    
    logger.info("[OK] API Blueprints registered")


def register_error_handlers(app):
    """Register error handlers"""
    
    @app.errorhandler(400)
    def bad_request(error):
        return jsonify({"error": "Bad request", "message": str(error)}), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        return jsonify({"error": "Unauthorized", "message": "Authentication required"}), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        return jsonify({"error": "Forbidden", "message": "Insufficient permissions"}), 403
    
    @app.errorhandler(404)
    def not_found(error):
        return jsonify({"error": "Not found", "message": "Resource not found"}), 404
    
    @app.errorhandler(429)
    def ratelimit_handler(error):
        return jsonify({"error": "Rate limit exceeded", "message": str(error)}), 429
    
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        logger.error(f"Internal error: {error}")
        return jsonify({"error": "Internal server error"}), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        logger.exception(f"Unhandled exception: {error}")
        return jsonify({"error": "Internal server error"}), 500


def register_health_check(app):
    """Register health check endpoint"""
    
    @app.route('/health')
    def health_check():
        """Comprehensive health check"""
        from app.services.faiss_service import faiss_service
        from app.services.cache_service import cache_service
        from sqlalchemy import text
        
        health = {
            "status": "healthy",
            "version": "2.0.0",
            "mode": "api-only",
            "checks": {}
        }
        
        # Check database
        try:
            with db.engine.connect() as conn:
                conn.execute(text('SELECT 1'))
                conn.commit()
            health["checks"]["database"] = "healthy"
        except Exception as e:
            health["checks"]["database"] = f"unhealthy: {str(e)}"
            health["status"] = "unhealthy"
        
        # Check cache
        try:
            if cache_service.is_available():
                cache_service.ping()
                health["checks"]["cache"] = "healthy"
            else:
                health["checks"]["cache"] = "not_configured"
        except Exception as e:
            health["checks"]["cache"] = f"unhealthy: {str(e)}"
        
        # Check FAISS
        try:
            total = faiss_service.get_total_persons()
            health["checks"]["faiss"] = f"healthy ({total} persons)"
        except Exception as e:
            health["checks"]["faiss"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"
        
        status_code = 200 if health["status"] in ["healthy", "degraded"] else 503
        return jsonify(health), status_code
    
    @app.route('/')
    @app.route('/api')
    def index():
        """API information"""
        return jsonify({
            "service": "Smart Attendance API",
            "version": "2.0.0",
            "mode": "api-only",
            "status": "running",
            "documentation": "/api/docs",
            "health": "/health",
            "init_db": "/init-db",
            "endpoints": {
                "auth": "/api/auth/*",
                "enrollment": "/api/enrollment/*",
                "attendance": "/api/attendance/*",
                "admin": "/api/admin/*",
                "face": "/api/face/*",
                "user": "/api/user/*"
            }
        })
    
    @app.route('/init-db')
    def init_database_endpoint():
        """Initialize database tables (call once)"""
        from sqlalchemy import inspect
        from app.models import Admin
        from werkzeug.security import generate_password_hash
        
        try:
            # Check if already initialized
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            
            result = {
                "status": "success",
                "message": "Database initialization",
                "tables": []
            }
            
            if not existing_tables:
                # Create all tables
                db.create_all()
                
                # Verify creation
                inspector = inspect(db.engine)
                tables = inspector.get_table_names()
                result["tables"] = tables
                result["message"] = f"Created {len(tables)} tables"
                
                # Create default admin
                if Admin.query.count() == 0:
                    admin = Admin(
                        name="Admin",
                        email="admin@admin.com",
                        password_hash=generate_password_hash("password123")
                    )
                    db.session.add(admin)
                    db.session.commit()
                    
                    result["admin"] = {
                        "email": "admin@admin.com",
                        "password": "password123",
                        "warning": "CHANGE THIS PASSWORD IMMEDIATELY!"
                    }
            else:
                result["tables"] = existing_tables
                result["message"] = f"Database already initialized with {len(existing_tables)} tables"
                result["admin_count"] = Admin.query.count()
            
            return jsonify(result), 200
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return jsonify({
                "status": "error",
                "message": str(e)
            }), 500


def register_commands(app):
    """Register CLI commands"""
    
    @app.cli.command()
    def init_db():
        """Initialize database"""
        db.create_all()
        print("[OK] Database initialized")
    
    @app.cli.command()
    def create_admin():
        """Create admin user"""
        from app.models import Admin
        from werkzeug.security import generate_password_hash
        
        email = input("Email: ")
        password = input("Password: ")
        name = input("Name: ")
        
        admin = Admin(
            email=email,
            password_hash=generate_password_hash(password),
            name=name
        )
        db.session.add(admin)
        db.session.commit()
        print(f"[OK] Admin created: {email}")
    
    @app.cli.command()
    def rebuild_faiss():
        """Rebuild FAISS index"""
        from app.services.faiss_service import faiss_service
        faiss_service.rebuild_from_database()
        print("[OK] FAISS index rebuilt")