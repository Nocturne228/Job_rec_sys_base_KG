"""
Configuration module for the job recommendation system.
"""

from .settings import (
    DataConfig,
    ModelConfig,
    Settings,
    SystemConfig,
    get_settings,
    update_settings,
)

__all__ = ["Settings", "ModelConfig", "DataConfig", "SystemConfig"]
