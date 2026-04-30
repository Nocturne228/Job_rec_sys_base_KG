"""
Skill coverage calculation for ranking.
Computes how well a user's skills match job requirements,
with optional GAT-weighted skill importance for production-scale KG.
"""
from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SkillCoverageCalculator:
    """Calculate skill coverage between users and jobs.

    Supports two modes:
    - Uniform weights (demo): all skills equal, suitable for ~20 skill nodes
    - GAT-weighted (production): learned skill importance via GAT,
      meaningful for 1,000+ skill nodes in KG, making "missing Spring Boot"
      penalize coverage much more than "missing Git"
    """

    def __init__(self, skill_level_mapping: Optional[Dict[str, int]] = None,
                 gat_weighter=None):
        """
        Args:
            skill_level_mapping: mapping from skill level strings to numeric values
            gat_weighter: GATSkillWeighter instance (None => falls back to uniform weights)
        """
        if skill_level_mapping is None:
            self.skill_level_mapping = {
                'beginner': 1, 'intermediate': 2,
                'advanced': 3, 'expert': 4
            }
        else:
            self.skill_level_mapping = skill_level_mapping

        self.gat_weighter = gat_weighter

    def calculate_coverage(self,
                          user_skills: Dict[str, str],
                          job_required_skills: Dict[str, str],
                          job_preferred_skills: Optional[Dict[str, str]] = None
                          ) -> Dict[str, Any]:
        """
        Calculate skill coverage, returning both uniform and GAT-weighted scores.

        Returns:
            Dict containing 'coverage_score' (uniform weights) and
            'gat_coverage_score' (GAT weighted, None if GAT not enabled).
        """
        if job_preferred_skills is None:
            job_preferred_skills = {}

        required_metrics = self._calculate_skill_match(
            user_skills, job_required_skills, is_required=True
        )
        preferred_metrics = self._calculate_skill_match(
            user_skills, job_preferred_skills, is_required=False
        )

        total_required = len(job_required_skills)
        total_preferred = len(job_preferred_skills)
        total_skills = total_required + total_preferred

        if total_skills == 0:
            coverage_score = 1.0
        else:
            required_weight = 0.7 if total_required > 0 else 0.0
            preferred_weight = 0.3 if total_preferred > 0 else 0.0
            weight_sum = required_weight + preferred_weight
            if weight_sum > 0:
                required_weight /= weight_sum
                preferred_weight /= weight_sum

            coverage_score = (required_weight * required_metrics['coverage_ratio'] +
                            preferred_weight * preferred_metrics['coverage_ratio'])

        # GAT-weighted coverage: use same required+preferred pool as uniform
        # so uniform vs GAT comparison is on the same skill set
        gat_coverage: Optional[float] = None
        if self.gat_weighter is not None:
            all_required = dict(job_required_skills)
            all_required.update(job_preferred_skills)
            gat_coverage = self._gat_weighted_coverage(user_skills, all_required)

        skill_gap = self._calculate_skill_gap(user_skills, job_required_skills)

        return {
            'coverage_score': coverage_score,
            'gat_coverage_score': gat_coverage,
            'required_metrics': required_metrics,
            'preferred_metrics': preferred_metrics,
            'skill_gap': skill_gap,
            'total_required_skills': total_required,
            'total_preferred_skills': total_preferred,
            'user_skill_count': len(user_skills)
        }

    # ------------------- GAT weighted coverage -------------------

    def _gat_weighted_coverage(self,
                               user_skills: Dict[str, str],
                               job_required_skills: Dict[str, str]) -> float:
        """
        GAT-weighted skill coverage.

        Coverage_GAT = sum(weights of matched skills) /
                       sum(weights of all required skills)

        Uses GAT-learned per-skill importance weights so that missing a core
        professional skill (e.g., Spring Boot) penalizes coverage more than
        missing a generic skill (e.g., Git).
        """
        user_skill_ids = set(user_skills.keys())
        matched_weights = []
        total_weights = []

        for skill_id in job_required_skills:
            weight = self.gat_weighter.get_skill_weight(skill_id)
            total_weights.append(weight)
            if skill_id in user_skill_ids:
                matched_weights.append(weight)

        total = sum(total_weights)
        return sum(matched_weights) / total if total > 0 else 0.0

    # ------------------- core matching logic -------------------

    def _calculate_skill_match(self,
                              user_skills: Dict[str, str],
                              job_skills: Dict[str, str],
                              is_required: bool = True) -> Dict[str, Any]:
        """Calculate match metrics for a set of skills."""
        if not job_skills:
            return {
                'coverage_ratio': 1.0, 'matched_skills': [],
                'missing_skills': [], 'level_deficits': []
            }

        matched_skills = []
        missing_skills = []
        level_deficits = []

        for skill_id, required_level in job_skills.items():
            user_level = user_skills.get(skill_id)
            if user_level is None:
                missing_skills.append(skill_id)
            else:
                user_level_num = self.skill_level_mapping.get(user_level.lower(), 0)
                required_level_num = self.skill_level_mapping.get(required_level.lower(), 0)

                if user_level_num >= required_level_num:
                    matched_skills.append(skill_id)
                else:
                    level_deficits.append({
                        'skill_id': skill_id,
                        'user_level': user_level,
                        'required_level': required_level,
                        'deficit': required_level_num - user_level_num
                    })

        return {
            'coverage_ratio': len(matched_skills) / len(job_skills),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'level_deficits': level_deficits,
            'is_required': is_required
        }

    def _calculate_skill_gap(self,
                            user_skills: Dict[str, str],
                            job_required_skills: Dict[str, str]) -> List[Dict[str, Any]]:
        """Calculate detailed skill gap information."""
        skill_gap = []

        for skill_id, required_level in job_required_skills.items():
            user_level = user_skills.get(skill_id)
            if user_level is None:
                skill_gap.append({
                    'skill_id': skill_id, 'user_level': None,
                    'required_level': required_level,
                    'gap_type': 'missing', 'gap_severity': 'high'
                })
            else:
                user_level_num = self.skill_level_mapping.get(user_level.lower(), 0)
                required_level_num = self.skill_level_mapping.get(required_level.lower(), 0)

                if user_level_num < required_level_num:
                    gap_size = required_level_num - user_level_num
                    skill_gap.append({
                        'skill_id': skill_id,
                        'user_level': user_level,
                        'required_level': required_level,
                        'gap_type': 'level_deficit',
                        'gap_size': gap_size,
                        'gap_severity': 'medium' if gap_size == 1 else 'high'
                    })

        return skill_gap
