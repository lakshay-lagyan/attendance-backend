# ============================================================
# FIXED FILE: config/__init__.py
# ============================================================

from config.production import ProductionConfig

config = {
    'production': ProductionConfig,
    'default': ProductionConfig
}

__all__ = ['config', 'ProductionConfig']


# Change: from app.models import Person