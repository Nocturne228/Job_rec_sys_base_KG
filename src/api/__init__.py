
"""
API layer: FastAPI REST endpoints covering competition requirements FR-4 through FR-7.
Designed for competition deliverable D-4 (demo video) and D-5 (installable package).
"""

from .routes import create_app

__all__ = ["create_app"]
