"""
Data models for the job recommendation system.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


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
    required_skills: Dict[str, SkillLevel] = Field(default_factory=dict)  # skill_id -> min_level
    preferred_skills: Dict[str, SkillLevel] = Field(default_factory=dict)
    salary_range: Optional[tuple[float, float]] = None


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


class GraphEntities(BaseModel):
    """Container for graph entities."""
    users: List[User]
    jobs: List[JobPosting]
    skills: List[Skill]
    applications: List[Application]
    interactions: List[Interaction]