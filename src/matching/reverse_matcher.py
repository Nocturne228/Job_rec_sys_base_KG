"""
Enterprise-side reverse matching: given a job requirement, rank candidates.
Uses the same underlying models (SBERT semantic + skill coverage + GAT weighting)
in reverse direction for person-job fit scoring.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CandidateResult:
    user_id: str
    score: float
    matched_skills: List[str] = field(default_factory=list)
    missing_skills: List[str] = field(default_factory=list)
    coverage_ratio: float = 0.0


class ReverseMatcher:
    def __init__(self, sbert_recall=None, skill_calculator=None, gat_weighter=None):
        self.sbert = sbert_recall
        self.skill_calc = skill_calculator
        self.gat = gat_weighter

    def match_candidates(
        self,
        job_id: str,
        job_required_skills: Dict[str, str],
        job_preferred_skills: Dict[str, str],
        candidate_ids: List[str],
        candidate_skills_map: Dict[str, Dict[str, str]],
        top_k: int = 20,
    ) -> List[CandidateResult]:

        results = []
        for uid in candidate_ids:
            user_skills = candidate_skills_map.get(uid, {})

            if self.skill_calc is not None:
                cov = self.skill_calc.calculate_coverage(
                    user_skills, job_required_skills, job_preferred_skills
                )
                coverage = cov.get("gat_coverage_score") or cov["coverage_score"]
                matched = cov["required_metrics"]["matched_skills"]
                missing = cov["required_metrics"]["missing_skills"]
            else:
                matched_required = set(user_skills) & set(job_required_skills)
                coverage = len(matched_required) / max(len(job_required_skills), 1)
                matched = list(matched_required)
                missing = list(set(job_required_skills) - set(user_skills))

            sb_score = 0.5
            if self.sbert is not None:
                sbr = self.sbert.recommend_for_user(uid, k=1)
                if sbr:
                    sb_score = sbr[0][1]

            final_score = 0.6 * coverage + 0.4 * sb_score

            results.append(CandidateResult(
                user_id=uid, score=round(final_score, 4),
                matched_skills=matched, missing_skills=missing,
                coverage_ratio=round(coverage, 4),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
