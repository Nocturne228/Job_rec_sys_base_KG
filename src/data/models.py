"""
Data models for the job recommendation system.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SkillLevel(str, Enum):
    """Skill proficiency levels."""

    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class Skill(BaseModel):
    """Skill entity."""

    id: str
    name: str
    category: str  # e.g., "programming", "data_science", "soft_skill"
    description: Optional[str] = None


class User(BaseModel):
    """User entity representing a job seeker."""

    id: str
    name: str
    education: Optional[str] = None
    experience_years: float = 0.0
    skills: Dict[str, SkillLevel] = Field(default_factory=dict)  # skill_id -> level
    resume_text: Optional[str] = None


class JobPosting(BaseModel):
    """Job posting entity."""

    id: str
    title: str
    company: str
    description: str
    required_skills: Dict[str, SkillLevel] = Field(
        default_factory=dict
    )  # skill_id -> min_level
    preferred_skills: Dict[str, SkillLevel] = Field(default_factory=dict)
    salary_range: Optional[tuple[float, float]] = None
    posted_at: str = "1970-01-01T00:00:00"


class Application(BaseModel):
    """User's application to a job."""

    user_id: str
    job_id: str
    status: str  # "applied", "interview", "rejected", "accepted"
    date: str


class Interaction(BaseModel):
    """User interaction with a job (click, view, save)."""

    user_id: str
    job_id: str
    interaction_type: str  # "view", "click", "save", "apply"
    timestamp: str


class FeedExposure(BaseModel):
    """一次岗位内容流曝光及可观测结果。"""

    impression_id: str
    user_id: str
    job_id: str
    timestamp: str
    position: int = Field(ge=1)
    clicked: bool = False
    dwell_seconds: float = Field(default=0.0, ge=0.0)
    saved: bool = False
    applied: bool = False

    @property
    def engaged(self) -> bool:
        """Pointwise 排序标签；定义透明，不冒充真实业务目标。"""
        return bool(
            self.clicked or self.saved or self.applied or self.dwell_seconds >= 20.0
        )


class SkillRelation(BaseModel):
    """Typed, provenance-aware edge in the skill graph."""

    source_skill_id: str
    target_skill_id: str
    relation_type: str = "PREREQUISITE_OF"
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = "curated_demo"


class GraphEntities(BaseModel):
    """Container for graph entities."""

    users: List[User]
    jobs: List[JobPosting]
    skills: List[Skill]
    applications: List[Application]
    interactions: List[Interaction]
    exposures: List[FeedExposure] = Field(default_factory=list)
    skill_relations: List[SkillRelation] = Field(default_factory=list)
