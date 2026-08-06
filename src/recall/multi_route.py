"""多路召回候选协议；小目录使用精确 Top-K，接口可替换为 ANN。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class RecallEvidence:
    source: str
    score: float
    rank: int


@dataclass(frozen=True)
class Candidate:
    job_id: str
    evidence: tuple[RecallEvidence, ...]

    @property
    def sources(self) -> list[str]:
        return [row.source for row in self.evidence]


def merge_recall_routes(
    routes: Mapping[str, Mapping[str, float]],
    allowed_job_ids: Sequence[str],
    per_route_k: int,
) -> list[Candidate]:
    """取每路 Top-K 后合并去重，并保留候选来源、原始分数与路内排名。"""
    if per_route_k <= 0:
        return []
    allowed = set(allowed_job_ids)
    evidence_by_job: dict[str, list[RecallEvidence]] = {}
    for source, score_by_job in routes.items():
        ranked = sorted(
            (
                (job_id, float(score))
                for job_id, score in score_by_job.items()
                if job_id in allowed
            ),
            key=lambda row: (-row[1], row[0]),
        )[:per_route_k]
        for rank, (job_id, score) in enumerate(ranked, start=1):
            evidence_by_job.setdefault(job_id, []).append(
                RecallEvidence(source=source, score=score, rank=rank)
            )
    return [
        Candidate(job_id=job_id, evidence=tuple(rows))
        for job_id, rows in sorted(
            evidence_by_job.items(),
            key=lambda row: (
                min(item.rank for item in row[1]),
                -len(row[1]),
                row[0],
            ),
        )
    ]
