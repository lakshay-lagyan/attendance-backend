import os
import logging
from app import create_app
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

app = create_app(os.getenv('FLASK_ENV', 'production'))

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    debug = os.getenv('FLASK_ENV') == 'development'
    
    logger.info(f"   Starting Smart Attendance API")
    logger.info(f"   Environment: {os.getenv('FLASK_ENV', 'production')}")
    logger.info(f"   Port: {port}")
    logger.info(f"   Debug: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )