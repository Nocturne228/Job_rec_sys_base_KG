"""
Data layer for the job recommendation system.
"""

from .models import Skill, User, JobPosting, Application, Interaction, SkillLevel, GraphEntities
from .generator import generate_mock_data
from .loader import DataLoader, GraphLoader

__all__ = [
    "Skill",
    "User",
    "JobPosting",
    "Application",
    "Interaction",
    "SkillLevel",
    "GraphEntities",
    "generate_mock_data",
    "DataLoader",
    "GraphLoader",
]