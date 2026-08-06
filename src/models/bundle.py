"""离线训练与在线服务共享的内容可追溯发布契约。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def data_fingerprint(data: BaseModel) -> str:
    payload = json.dumps(
        data.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def serving_fingerprint(
    *,
    data_sha256: str,
    checkpoint_sha256: str,
    ranking_model: dict[str, Any],
    text_config: dict[str, Any],
    reranking_config: dict[str, Any],
) -> str:
    """标识会改变在线排序结果的完整 Serving 配置。"""
    payload = json.dumps(
        {
            "data_sha256": data_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "ranking_model": ranking_model,
            "text_config": text_config,
            "reranking_config": reranking_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ModelBundle(BaseModel):
    schema_version: int = 3
    model_version: str
    created_at: str
    data_seed: int
    data_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint_path: str
    checkpoint_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    serving_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    training_config: dict[str, Any]
    user_id_to_idx: dict[str, int]
    job_id_to_idx: dict[str, int]
    train_items_by_user: dict[str, list[str]]
    ranking_weights: dict[str, float] = Field(
        default_factory=lambda: {"lightgcn": 0.4, "text": 0.3, "skill": 0.3}
    )
    ranking_model: dict[str, Any]
    text_config: dict[str, Any] = Field(
        default_factory=lambda: {"kind": "feature_hashing", "n_features": 512}
    )
    reranking_config: dict[str, Any] = Field(
        default_factory=lambda: {
            "diversity_penalty": 0.12,
            "freshness_bonus": 0.05,
        }
    )

    @model_validator(mode="after")
    def validate_contract(self) -> "ModelBundle":
        if self.schema_version != 3:
            raise ValueError("unsupported bundle schema")
        if sorted(self.user_id_to_idx.values()) != list(
            range(len(self.user_id_to_idx))
        ):
            raise ValueError("user mapping must be contiguous")
        if sorted(self.job_id_to_idx.values()) != list(range(len(self.job_id_to_idx))):
            raise ValueError("job mapping must be contiguous")
        if set(self.ranking_weights) != {"lightgcn", "text", "skill"}:
            raise ValueError("unexpected ranking weight keys")
        known_jobs = set(self.job_id_to_idx)
        if any(
            job_id not in known_jobs
            for jobs in self.train_items_by_user.values()
            for job_id in jobs
        ):
            raise ValueError("seen set contains an unknown job")
        expected_serving_hash = serving_fingerprint(
            data_sha256=self.data_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            ranking_model=self.ranking_model,
            text_config=self.text_config,
            reranking_config=self.reranking_config,
        )
        if expected_serving_hash != self.serving_sha256:
            raise ValueError("serving configuration hash does not match bundle")
        return self

    @classmethod
    def load(cls, path: str | Path) -> "ModelBundle":
        bundle_path = Path(path)
        bundle = cls.model_validate_json(bundle_path.read_text(encoding="utf-8"))
        checkpoint = Path(bundle.checkpoint_path)
        if not checkpoint.is_absolute():
            checkpoint = bundle_path.parent / checkpoint
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
        if sha256_file(checkpoint) != bundle.checkpoint_sha256:
            raise ValueError("checkpoint hash does not match bundle")
        bundle.checkpoint_path = str(checkpoint.resolve())
        return bundle

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")
