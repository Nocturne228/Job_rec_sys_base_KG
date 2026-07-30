"""
Data layer for the job recommendation system.
"""

from .external import (
    ExternalDatasetManifest,
    ExternalInteraction,
    ExternalJob,
    prepare_external_snapshot,
)
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
from .private_profile_store import (
    PRIVATE_PROFILE_FIELDS,
    EncryptedProfileStore,
    PrivateProfile,
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
    "ExternalJob",
    "ExternalInteraction",
    "ExternalDatasetManifest",
    "prepare_external_snapshot",
    "PRIVATE_PROFILE_FIELDS",
    "PrivateProfile",
    "EncryptedProfileStore",
]
