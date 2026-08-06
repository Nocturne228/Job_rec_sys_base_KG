"""相关性之后的确定性多样性与新岗位探索重排。"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from src.data import JobPosting


class DiversityReranker:
    def __init__(self, diversity_penalty: float = 0.12, freshness_bonus: float = 0.05):
        self.diversity_penalty = diversity_penalty
        self.freshness_bonus = freshness_bonus

    @staticmethod
    def _similarity(left: JobPosting, right: JobPosting) -> float:
        left_skills = set(left.required_skills) | set(left.preferred_skills)
        right_skills = set(right.required_skills) | set(right.preferred_skills)
        union = left_skills | right_skills
        skill_similarity = (
            len(left_skills & right_skills) / len(union) if union else 0.0
        )
        return min(
            1.0,
            skill_similarity
            + 0.15 * (left.company == right.company)
            + 0.10 * (left.title == right.title),
        )

    def rerank(
        self,
        ranked: Sequence[tuple[int, float, dict[str, float]]],
        jobs: Sequence[JobPosting],
        top_k: int,
    ) -> list[tuple[int, float, dict[str, float]]]:
        remaining = list(ranked)
        selected: list[tuple[int, float, dict[str, float]]] = []
        newest = max(datetime.fromisoformat(job.posted_at) for job in jobs)
        while remaining and len(selected) < top_k:
            best_position = 0
            best_adjusted = float("-inf")
            best_adjustment = 0.0
            for position, (index, score, _) in enumerate(remaining):
                job = jobs[index]
                similarity = max(
                    (self._similarity(job, jobs[row[0]]) for row in selected),
                    default=0.0,
                )
                age_days = max(0, (newest - datetime.fromisoformat(job.posted_at)).days)
                freshness = max(0.0, 1.0 - age_days / 30.0)
                adjustment = (
                    self.freshness_bonus * freshness
                    - self.diversity_penalty * similarity
                )
                adjusted = score + adjustment
                if adjusted > best_adjusted:
                    best_position = position
                    best_adjusted = adjusted
                    best_adjustment = adjustment
            index, _, contributions = remaining.pop(best_position)
            selected.append(
                (
                    index,
                    best_adjusted,
                    {**contributions, "rerank_adjustment": best_adjustment},
                )
            )
        return selected
