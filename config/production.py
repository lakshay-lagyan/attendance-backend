
import os
from datetime import timedelta

class ProductionConfig:
    """Production configuration with performance optimizations"""
    
    # Flask
    SECRET_KEY = os.getenv('FLASK_SECRET', os.urandom(32).hex())  # FIXED
    DEBUG = False
    TESTING = False
    
    # Database - Optimized connection pooling
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://localhost/mydb')
    if SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 20,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 30,
        'pool_timeout': 30,
        'echo': False,
        'connect_args': {
            'connect_timeout': 10,
            'options': '-c statement_timeout=30000'
        }
    }
    
    # Query optimization
    SQLALCHEMY_RECORD_QUERIES = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = False
    
    # Redis Cache
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = REDIS_URL
    CACHE_DEFAULT_TIMEOUT = 3600
    CACHE_KEY_PREFIX = 'attendance:'
    
    # Session
    SESSION_TYPE = 'redis'
    SESSION_REDIS = REDIS_URL
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=1)
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=30)
    JWT_ALGORITHM = 'HS256'
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'
    
    # CORS
    CORS_ORIGINS = os.getenv('FRONTEND_URL', 'https://localhost:3000').split(',')
    CORS_SUPPORTS_CREDENTIALS = True
    CORS_MAX_AGE = 3600
    
    # Rate Limiting
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_STRATEGY = 'fixed-window'
    RATELIMIT_HEADERS_ENABLED = True
    RATELIMIT_SWALLOW_ERRORS = True
    
    # Custom rate limits
    RATE_LIMITS = {
        'login': '10 per minute',
        'enrollment': '5 per hour',
        'face_recognition': '30 per minute',
        'attendance': '100 per hour',
        'api_default': '200 per hour'
    }
    
    # Upload Settings
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    UPLOAD_FOLDER = '/tmp/uploads'
    
    # Face Recognition
    FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_THRESHOLD', '0.6'))
    MIN_ENROLLMENT_IMAGES = int(os.getenv('MIN_IMAGES', '5'))
    MAX_ENROLLMENT_IMAGES = int(os.getenv('MAX_IMAGES', '10'))
    EMBEDDING_DIM = 512
    FACE_DETECTION_BACKEND = 'opencv'
    FACE_RECOGNITION_MODEL = 'ArcFace'
    
    # FAISS Configuration
    FAISS_INDEX_TYPE = 'IndexFlatIP'
    FAISS_CACHE_TTL = 3600
    FAISS_REBUILD_THRESHOLD = 100
    
    # Image Processing
    IMAGE_MAX_SIZE = (1024, 1024)
    IMAGE_QUALITY = 85
    IMAGE_COMPRESSION = True
    IMAGE_FORMAT = 'JPEG'
    
    # Background Tasks (Celery)
    CELERY_BROKER_URL = REDIS_URL
    CELERY_RESULT_BACKEND = REDIS_URL
    CELERY_TASK_SERIALIZER = 'json'
    CELERY_RESULT_SERIALIZER = 'json'
    CELERY_ACCEPT_CONTENT = ['json']
    CELERY_TIMEZONE = 'UTC'
    CELERY_ENABLE_UTC = True
    
    # Email Configuration
    MAIL_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('SMTP_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv('SMTP_USERNAME')
    MAIL_PASSWORD = os.getenv('SMTP_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_FROM', 'noreply@attendance.com')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = os.getenv('LOG_FILE', 'app.log')
    LOG_MAX_BYTES = 10 * 1024 * 1024
    LOG_BACKUP_COUNT = 5
    
    # Monitoring
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    ENABLE_METRICS = os.getenv('ENABLE_METRICS', 'true').lower() == 'true'
    METRICS_PORT = int(os.getenv('METRICS_PORT', 9090))
    
    # Security
    SECURITY_PASSWORD_SALT = os.getenv('SECURITY_PASSWORD_SALT', SECRET_KEY)
    SECURITY_PASSWORD_HASH = 'bcrypt'
    BCRYPT_LOG_ROUNDS = 12
    
    # Admin
    CREATE_ADMIN_SECRET = os.getenv('CREATE_ADMIN_SECRET', 'change-this-secret')
    
    # Feature Flags
    ENABLE_EMAIL_NOTIFICATIONS = os.getenv('ENABLE_EMAIL', 'false').lower() == 'true'
    ENABLE_BACKGROUND_TASKS = os.getenv('ENABLE_CELERY', 'false').lower() == 'true'
    ENABLE_FACE_LIVENESS = os.getenv('ENABLE_LIVENESS', 'true').lower() == 'true'
    
    # Performance
    SEND_FILE_MAX_AGE_DEFAULT = 31536000
    COMPRESS_MIMETYPES = [
        'text/html', 'text/css', 'text/xml',
        'application/json', 'application/javascript'
    ]
    COMPRESS_LEVEL = 6
    COMPRESS_MIN_SIZE = 500
    
    # Pagination
    DEFAULT_PAGE_SIZE = 20
    MAX_PAGE_SIZE = 100
    
    # Timeouts
    REQUEST_TIMEOUT = 30
    DATABASE_QUERY_TIMEOUT = 10
    FACE_RECOGNITION_TIMEOUT = 15
    
    # Backup
    BACKUP_ENABLED = os.getenv('BACKUP_ENABLED', 'true').lower() == 'true'
    BACKUP_SCHEDULE = os.getenv('BACKUP_SCHEDULE', '0 2 * * *')
    BACKUP_RETENTION_DAYS = int(os.getenv('BACKUP_RETENTION_DAYS', 30))
    
    @classmethod
    def init_app(cls, app):
        """Initialize app-specific configuration"""
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Setup file logging
        if not app.debug:
            file_handler = RotatingFileHandler(
                cls.LOG_FILE,
                maxBytes=cls.LOG_MAX_BYTES,
                backupCount=cls.LOG_BACKUP_COUNT
            )
            file_handler.setLevel(getattr(logging, cls.LOG_LEVEL))
            file_handler.setFormatter(logging.Formatter(cls.LOG_FORMAT))
            app.logger.addHandler(file_handler)
        
        # Setup Sentry
        if cls.SENTRY_DSN:
            try:
                import sentry_sdk
                from sentry_sdk.integrations.flask import FlaskIntegration
                sentry_sdk.init(
                    dsn=cls.SENTRY_DSN,
                    integrations=[FlaskIntegration()],
                    traces_sample_rate=0.1,
                    environment='production'
                )
            except ImportError:
                app.logger.warning("Sentry SDK not installed")


# Export config dictionary
config = {
    'production': ProductionConfig,
    'default': ProductionConfig
}