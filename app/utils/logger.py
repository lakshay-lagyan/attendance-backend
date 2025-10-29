import logging
import sys
from logging.handlers import RotatingFileHandler
import os

def setup_logging():
    """Setup application logging"""
    
    # Get log level from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if os.getenv('LOG_FILE'):
        try:
            file_handler = RotatingFileHandler(
                os.getenv('LOG_FILE', 'app.log'),
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, log_level))
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
    
    # Suppress noisy loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.ERROR)
    
    root_logger.info("✅ Logging initialized")

