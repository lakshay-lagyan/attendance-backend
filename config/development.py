import os
from datetime import timedelta

class DevelopmentConfig:
    
    # Flask
    SECRET_KEY = os.getenv('FLASK_SECRET', 'dev-secret-key-change-in-production')
    DEBUG = True
    TESTING = False
    
    # Database - Use SQLite for local development
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///attendance_dev.db')
    if SQLALCHEMY_DATABASE_URI.startswith('postgres://'):
        SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI.replace('postgres://', 'postgresql://', 1)
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'echo': True  # Log SQL queries in dev mode
    }
    
    # Redis Cache - Use memory cache if Redis unavailable
    REDIS_URL = os.getenv('REDIS_URL', 'memory://')
    CACHE_TYPE = 'simple'  # Use simple cache for development
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_KEY_PREFIX = 'attendance:'
    
    # Session
    SESSION_TYPE = 'filesystem'
    SESSION_COOKIE_SECURE = False
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # JWT
    JWT_SECRET_KEY = os.getenv('JWT_SECRET', SECRET_KEY)
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)  # Longer for dev
    JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=90)
    JWT_ALGORITHM = 'HS256'
    JWT_TOKEN_LOCATION = ['headers']
    JWT_HEADER_NAME = 'Authorization'
    JWT_HEADER_TYPE = 'Bearer'
    
    # CORS - Allow all origins in development
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5173', 'http://127.0.0.1:3000']
    CORS_SUPPORTS_CREDENTIALS = True
    CORS_MAX_AGE = 3600
    
    # Rate Limiting - More lenient in development
    RATELIMIT_STORAGE_URL = 'memory://'
    RATELIMIT_STRATEGY = 'fixed-window'
    RATELIMIT_HEADERS_ENABLED = True
    RATELIMIT_SWALLOW_ERRORS = True
    
    # Custom rate limits - More permissive for dev
    RATE_LIMITS = {
        'login': '100 per minute',
        'enrollment': '50 per hour',
        'face_recognition': '300 per minute',
        'attendance': '1000 per hour',
        'api_default': '2000 per hour'
    }
    
    # Upload Settings
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
    UPLOAD_FOLDER = './uploads'
    
    # Face Recognition
    FACE_RECOGNITION_THRESHOLD = float(os.getenv('FACE_THRESHOLD', '0.6'))
    MIN_ENROLLMENT_IMAGES = int(os.getenv('MIN_IMAGES', '3'))  # Lower for dev
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
    
    # Background Tasks (Celery) - Disabled in dev
    CELERY_BROKER_URL = None
    CELERY_RESULT_BACKEND = None
    
    # Email Configuration - Disabled in dev
    MAIL_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.getenv('SMTP_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.getenv('SMTP_USERNAME')
    MAIL_PASSWORD = os.getenv('SMTP_PASSWORD')
    MAIL_DEFAULT_SENDER = os.getenv('MAIL_FROM', 'noreply@attendance.com')
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'DEBUG')
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_FILE = None  # Console only in dev
    
    # Monitoring - Disabled in dev
    SENTRY_DSN = None
    ENABLE_METRICS = False
    
    # Security
    SECURITY_PASSWORD_SALT = os.getenv('SECURITY_PASSWORD_SALT', SECRET_KEY)
    SECURITY_PASSWORD_HASH = 'bcrypt'
    BCRYPT_LOG_ROUNDS = 4  # Faster in dev
    
    # Admin
    CREATE_ADMIN_SECRET = os.getenv('CREATE_ADMIN_SECRET', 'dev-admin-secret')
    
    # Feature Flags
    ENABLE_EMAIL_NOTIFICATIONS = False
    ENABLE_BACKGROUND_TASKS = False
    ENABLE_FACE_LIVENESS = False  # Disabled for faster dev
    
    # Performance
    SEND_FILE_MAX_AGE_DEFAULT = 0  # No caching in dev
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
    REQUEST_TIMEOUT = 60  # Longer for dev/debugging
    DATABASE_QUERY_TIMEOUT = 30
    FACE_RECOGNITION_TIMEOUT = 30
    
    # Backup - Disabled in dev
    BACKUP_ENABLED = False
