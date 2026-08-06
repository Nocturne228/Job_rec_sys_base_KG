"""离线训练和在线服务共享的岗位 Feed 排序特征。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
import torch

from src.data import DataLoader, GraphEntities, Interaction, JobPosting
from src.ranking.skill_coverage import SkillCoverageCalculator
from src.recall import TextRecall

FEED_FEATURE_NAMES = (
    "lightgcn",
    "text",
    "skill",
    "popularity",
    "freshness",
    "recent_interest",
)


@dataclass(frozen=True)
class FeedRankingFeatures:
    lightgcn: float = 0.0
    text: float = 0.0
    skill: float = 0.0
    popularity: float = 0.0
    freshness: float = 0.0
    recent_interest: float = 0.0

    def values(self) -> list[float]:
        return [float(getattr(self, name)) for name in FEED_FEATURE_NAMES]


class FeatureBuilder:
    """只使用训练状态构造候选特征，避免离线/在线各写一套逻辑。"""

    def __init__(
        self,
        data: GraphEntities,
        loader: DataLoader,
        text: TextRecall,
        user_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        recent_limit: int = 5,
    ) -> None:
        self.data = data
        self.loader = loader
        self.text = text
        self.user_embeddings = user_embeddings
        self.item_embeddings = item_embeddings
        self.skills = SkillCoverageCalculator()
        self.jobs = {job.id: job for job in data.jobs}
        popularity = np.asarray(loader.train_R.sum(axis=0)).ravel()
        self.popularity = {
            loader.idx_to_job_id[index]: float(value)
            for index, value in enumerate(popularity)
        }
        newest = max(datetime.fromisoformat(job.posted_at) for job in data.jobs)
        self.freshness = {
            job.id: max(
                0.0,
                1.0 - (newest - datetime.fromisoformat(job.posted_at)).days / 60.0,
            )
            for job in data.jobs
        }
        train_seen = {
            user_id: set(items)
            for user_id, items in self._train_items_by_user().items()
        }
        interactions_by_user: dict[str, list[Interaction]] = {}
        for interaction in data.interactions:
            if interaction.job_id in train_seen.get(interaction.user_id, set()):
                interactions_by_user.setdefault(interaction.user_id, []).append(
                    interaction
                )
        self.recent_skills: dict[str, set[str]] = {}
        for user_id, interactions in interactions_by_user.items():
            recent = sorted(interactions, key=lambda row: row.timestamp, reverse=True)[
                :recent_limit
            ]
            skill_ids: set[str] = set()
            for interaction in recent:
                job = self.jobs[interaction.job_id]
                skill_ids.update(job.required_skills)
                skill_ids.update(job.preferred_skills)
            self.recent_skills[user_id] = skill_ids

    def _train_items_by_user(self) -> dict[str, list[str]]:
        return {
            self.loader.idx_to_user_id[user_idx]: [
                self.loader.idx_to_job_id[item_idx]
                for item_idx in self.loader.train_R[user_idx].indices
            ]
            for user_idx in range(self.loader.n_users)
        }

    def build(
        self,
        *,
        user_id: str | None,
        resume_text: str,
        user_skills: Mapping[str, object],
        jobs: Sequence[JobPosting],
        known_user: bool,
    ) -> dict[str, FeedRankingFeatures]:
        text_scores = dict(
            self.text.recommend_for_text(
                resume_text, k=len(jobs), job_ids=[job.id for job in jobs]
            )
        )
        collaborative: dict[str, float] = {}
        if known_user and user_id in self.loader.user_id_to_idx:
            user_index = self.loader.user_id_to_idx[user_id]
            raw = self.user_embeddings[user_index] @ self.item_embeddings.T
            collaborative = {
                job.id: float(raw[self.loader.job_id_to_idx[job.id]])
                for job in jobs
                if job.id in self.loader.job_id_to_idx
            }
        recent = self.recent_skills.get(user_id or "", set())
        rows: dict[str, FeedRankingFeatures] = {}
        for job in jobs:
            job_skills = set(job.required_skills) | set(job.preferred_skills)
            union = recent | job_skills
            recent_overlap = len(recent & job_skills) / len(union) if union else 0.0
            coverage = self.skills.calculate_coverage(
                user_skills, job.required_skills, job.preferred_skills
            )
            rows[job.id] = FeedRankingFeatures(
                lightgcn=collaborative.get(job.id, 0.0),
                text=text_scores.get(job.id, 0.0),
                skill=float(coverage["coverage_score"]),
                popularity=self.popularity.get(job.id, 0.0),
                freshness=self.freshness.get(job.id, 0.0),
                recent_interest=recent_overlap,
            )
        return rows

    @staticmethod
    def recall_routes(
        rows: Mapping[str, FeedRankingFeatures], known_user: bool
    ) -> dict[str, dict[str, float]]:
        routes = {
            "text": {job_id: row.text for job_id, row in rows.items()},
            "skill": {job_id: row.skill for job_id, row in rows.items()},
            "popular_fresh": {
                job_id: 0.7 * row.popularity + 0.3 * row.freshness
                for job_id, row in rows.items()
            },
        }
        if known_user:
            routes["lightgcn"] = {job_id: row.lightgcn for job_id, row in rows.items()}
        return routes
