"""
Data layer for the job recommendation system.
"""

from .generator import compatibility_score, generate_mock_data
from .graph_store import InMemorySkillGraph, Neo4jSkillGraph, SkillGraphStore
from .loader import DataLoader, GraphLoader
from .models import (
    Application,
    GraphEntities,
    Interaction,
    JobPosting,
    Skill,
    SkillLevel,
    SkillRelation,
    User,
)

__all__ = [
    "Skill",
    "User",
    "JobPosting",
    "Application",
    "Interaction",
    "SkillLevel",
    "SkillRelation",
    "GraphEntities",
    "generate_mock_data",
    "compatibility_score",
    "DataLoader",
    "GraphLoader",
    "SkillGraphStore",
    "InMemorySkillGraph",
    "Neo4jSkillGraph",
]
