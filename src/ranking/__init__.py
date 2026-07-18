"""
Ranking layer for job recommendation system.
Includes linear fusion ranking, GAT-based skill weighting, and potential future deep ranking models.
"""

from .gat_weighter import GATSkillWeighter
from .linear_fusion import LinearFusionRanker, RankingFeatures
from .skill_coverage import SkillCoverageCalculator

__all__ = [
    "LinearFusionRanker",
    "RankingFeatures",
    "SkillCoverageCalculator",
    "GATSkillWeighter",
]
