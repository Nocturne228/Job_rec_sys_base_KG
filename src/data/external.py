"""外部岗位/交互数据的隔离校验、伪名化与 lineage 清单。

该模块只准备本地 staging snapshot，不会让服务自动切换数据源或训练模型。
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Type, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .models import SkillLevel


class ExternalJob(BaseModel):
    """来源无关的最小岗位契约。"""

    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    company: str = Field(min_length=1)
    description: str = Field(min_length=1)
    required_skills: Dict[str, SkillLevel] = Field(default_factory=dict)
    preferred_skills: Dict[str, SkillLevel] = Field(default_factory=dict)
    source_url: str | None = None


class ExternalInteraction(BaseModel):
    """外部交互的 staging 契约；写出前必须替换原始用户 ID。"""

    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1)
    job_id: str = Field(min_length=1)
    interaction_type: Literal["view", "click", "save", "apply"]
    timestamp: str = Field(min_length=1)

    @field_validator("timestamp")
    @classmethod
    def require_aware_iso_timestamp(cls, value: str) -> str:
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("timestamp must be ISO 8601") from exc
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value


class ExternalDatasetManifest(BaseModel):
    """伴随标准化快照发布的来源、许可和完整性证据。"""

    schema_version: int = 1
    source_name: str
    source_url: str
    license_name: str
    license_url: str
    retrieved_at: str
    prepared_at: str
    jobs: int
    interactions: int
    distinct_users: int
    input_sha256: Dict[str, str]
    limitations: List[str]


ModelT = TypeVar("ModelT", bound=BaseModel)


def _read_jsonl(path: Path, model: Type[ModelT]) -> List[ModelT]:
    rows: List[ModelT] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rows.append(model.model_validate_json(raw))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: invalid record: {exc}") from exc
    if not rows:
        raise ValueError(f"{path}: no records")
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    lines = [row.model_dump_json() for row in rows]
    payload = "\n".join(lines) + ("\n" if lines else "")
    path.write_text(payload, encoding="utf-8")


def _pseudonymize(user_id: str, secret: str) -> str:
    digest = hmac.new(
        secret.encode("utf-8"), user_id.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    return f"user_{digest[:24]}"


def prepare_external_snapshot(
    jobs_path: str | Path,
    output_dir: str | Path,
    *,
    source_name: str,
    source_url: str,
    license_name: str,
    license_url: str,
    retrieved_at: str,
    interactions_path: str | Path | None = None,
    pseudonymization_secret: str | None = None,
) -> ExternalDatasetManifest:
    """校验本地 JSONL，写出标准化数据和可复核 manifest。

    有交互时必须提供仅来自环境变量的伪名化 secret。原始用户 ID 不会进入输出。
    """
    source_jobs = Path(jobs_path)
    jobs = _read_jsonl(source_jobs, ExternalJob)
    job_ids = [job.job_id for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("job_id must be unique")

    interactions: List[ExternalInteraction] = []
    input_hashes = {"jobs": _sha256(source_jobs)}
    source_interactions: Path | None = None
    if interactions_path is not None:
        if not pseudonymization_secret:
            raise ValueError(
                "JOBREC_IMPORT_PSEUDONYM_KEY is required when importing interactions"
            )
        source_interactions = Path(interactions_path)
        raw_interactions = _read_jsonl(source_interactions, ExternalInteraction)
        unknown_jobs = sorted({row.job_id for row in raw_interactions} - set(job_ids))
        if unknown_jobs:
            raise ValueError(
                f"interactions reference unknown jobs: {', '.join(unknown_jobs[:5])}"
            )
        interactions = [
            row.model_copy(
                update={"user_id": _pseudonymize(row.user_id, pseudonymization_secret)}
            )
            for row in raw_interactions
        ]
        input_hashes["interactions"] = _sha256(source_interactions)

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_jsonl(target / "jobs.jsonl", jobs)
    _write_jsonl(target / "interactions.jsonl", interactions)

    limitations = [
        "staging snapshot only; serving and training do not switch automatically",
        "license, consent, retention, deletion and source freshness need independent review",
        "no recommendation-effect or production-scale claim follows from successful import",
    ]
    manifest = ExternalDatasetManifest(
        source_name=source_name,
        source_url=source_url,
        license_name=license_name,
        license_url=license_url,
        retrieved_at=retrieved_at,
        prepared_at=datetime.now(timezone.utc).isoformat(),
        jobs=len(jobs),
        interactions=len(interactions),
        distinct_users=len({row.user_id for row in interactions}),
        input_sha256=input_hashes,
        limitations=limitations,
    )
    (target / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest
