import os
import sys
import logging
from dotenv import load_dotenv

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

# Load environment 
load_dotenv()

from app import create_app
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

flask_env = os.getenv('FLASK_ENV', 'production')
if flask_env not in ['production', 'development', 'testing']:
    logger.warning(f"Invalid FLASK_ENV '{flask_env}', defaulting to 'production'")
    flask_env = 'production'

app = create_app(flask_env)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 10000))
    debug = flask_env == 'development'
    
    logger.info(f"Starting Smart Attendance API")
    logger.info(f"Environment: {flask_env}")
    logger.info(f"Port: {port}")
    logger.info(f"Debug: {debug}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )