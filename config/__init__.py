from .production import ProductionConfig
from .development import DevelopmentConfig

config = {
    'production': ProductionConfig,
    'development': DevelopmentConfig,
    'testing': DevelopmentConfig,
    'default': DevelopmentConfig
}

__all__ = ['config', 'ProductionConfig', 'DevelopmentConfig']

# from app.models import Person