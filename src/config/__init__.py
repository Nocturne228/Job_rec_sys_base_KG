"""
Configuration module for the job recommendation system.
"""

from .settings import Settings, ModelConfig, DataConfig, SystemConfig, get_settings, update_settings

__all__ = ["Settings", "ModelConfig", "DataConfig", "SystemConfig"]