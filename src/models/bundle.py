"""Versioned model-bundle contract shared by offline training and online serving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel, Field, model_validator


class ModelBundle(BaseModel):
    schema_version: int = 1
    model_version: str
    created_at: str
    data_seed: int = 42
    lightgcn_checkpoint: str
    user_id_to_idx: Dict[str, int]
    job_id_to_idx: Dict[str, int]
    train_items_by_user: Dict[str, List[str]] = Field(default_factory=dict)
    skill_weights: Dict[str, float] = Field(default_factory=dict)
    ranking_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "lightgcn_score": 0.4,
            "sbert_score": 0.3,
            "skill_coverage": 0.3,
        }
    )

    @model_validator(mode="after")
    def validate_indices(self) -> "ModelBundle":
        if sorted(self.user_id_to_idx.values()) != list(
            range(len(self.user_id_to_idx))
        ):
            raise ValueError("user mapping must be contiguous")
        if sorted(self.job_id_to_idx.values()) != list(range(len(self.job_id_to_idx))):
            raise ValueError("job mapping must be contiguous")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        bundle_path = Path(path)
        data = json.loads(bundle_path.read_text(encoding="utf-8"))
        bundle = cls.model_validate(data)
        checkpoint = Path(bundle.lightgcn_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = bundle_path.parent / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"LightGCN checkpoint not found: {checkpoint}")
        bundle.lightgcn_checkpoint = str(checkpoint.resolve())
        return bundle

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")


class StaticSkillWeighter:
    """Read-only online view of offline-computed GAT skill scores."""

    def __init__(self, weights: Dict[str, float]):
        self.weights = dict(weights)

    def get_skill_weight(self, skill_id: str) -> float:
        return float(self.weights.get(skill_id, 0.0))

    def get_top_k_skills(self, k: int = 10):
        return sorted(self.weights.items(), key=lambda item: item[1], reverse=True)[:k]
