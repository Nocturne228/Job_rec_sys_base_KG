"""带位置观察偏差的可复现概率行为模型。"""

from __future__ import annotations

import hashlib
from typing import Tuple

from pydantic import BaseModel, Field, model_validator

from .schemas import BehaviorOutcome, PersonaJobJudgment


class BehaviorConfig(BaseModel):
    """行为漏斗参数；默认值仅用于协议演示，不代表真实平台校准。"""

    examination_by_rank: Tuple[float, ...] = (
        1.00,
        0.82,
        0.68,
        0.57,
        0.49,
        0.43,
        0.38,
        0.34,
        0.31,
        0.28,
    )
    minimum_attraction: float = Field(default=0.05, ge=0.0, le=1.0)
    relevance_attraction_weight: float = Field(default=0.85, ge=0.0, le=1.0)
    save_given_click: float = Field(default=0.35, ge=0.0, le=1.0)
    apply_given_save: float = Field(default=0.30, ge=0.0, le=1.0)
    feedback_given_click: float = Field(default=0.45, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_examination_curve(self) -> "BehaviorConfig":
        if not self.examination_by_rank:
            raise ValueError("examination_by_rank cannot be empty")
        if any(not 0.0 <= value <= 1.0 for value in self.examination_by_rank):
            raise ValueError("examination probabilities must be in [0, 1]")
        if any(
            right > left
            for left, right in zip(
                self.examination_by_rank, self.examination_by_rank[1:]
            )
        ):
            raise ValueError("examination probabilities must be non-increasing")
        if self.minimum_attraction + self.relevance_attraction_weight > 1.0:
            raise ValueError("attraction probability can exceed 1")
        return self


def _uniform(seed: int, persona_id: str, job_id: str, action: str) -> float:
    payload = f"{seed}|{persona_id}|{job_id}|{action}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(2**64)


class PositionAwareBehaviorModel:
    """将潜在相关性转为观察、点击、收藏、投递与反馈事件。"""

    def __init__(
        self, seed: int = 20260731, config: BehaviorConfig | None = None
    ) -> None:
        self.seed = seed
        self.config = config or BehaviorConfig()

    def simulate(
        self,
        persona_id: str,
        job_id: str,
        rank: int,
        judgment: PersonaJobJudgment,
    ) -> BehaviorOutcome:
        if rank < 1 or rank > len(self.config.examination_by_rank):
            raise ValueError(
                f"rank must be between 1 and {len(self.config.examination_by_rank)}"
            )
        examination = self.config.examination_by_rank[rank - 1]
        attraction = self.config.minimum_attraction + (
            self.config.relevance_attraction_weight * judgment.latent_relevance
        )
        marginal_click = examination * attraction

        examined = _uniform(self.seed, persona_id, job_id, "examine") < examination
        clicked = examined and (
            _uniform(self.seed, persona_id, job_id, "click") < attraction
        )
        saved = clicked and (
            _uniform(self.seed, persona_id, job_id, "save")
            < self.config.save_given_click
        )
        applied = saved and (
            _uniform(self.seed, persona_id, job_id, "apply")
            < self.config.apply_given_save
        )
        feedback = clicked and (
            _uniform(self.seed, persona_id, job_id, "feedback")
            < self.config.feedback_given_click
        )
        satisfaction_probability = 0.10 + 0.80 * judgment.latent_relevance
        satisfied = feedback and (
            _uniform(self.seed, persona_id, job_id, "satisfied")
            < satisfaction_probability
        )
        return BehaviorOutcome(
            rank=rank,
            examination_probability=examination,
            attraction_probability=attraction,
            marginal_click_probability=marginal_click,
            examined=examined,
            clicked=clicked,
            saved=saved,
            applied=applied,
            feedback_given=feedback,
            satisfied=satisfied,
        )
