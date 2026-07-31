"""在给定推荐列表上运行合成用户评估，不写入线上事件库。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Sequence

from src.data.models import JobPosting

from .behavior import PositionAwareBehaviorModel
from .judge import PROMPT_VERSION, PersonaJudge
from .schemas import Persona, SimulationRecord, SimulationSummary


@dataclass(frozen=True)
class RecommendationSlate:
    persona: Persona
    jobs: Sequence[JobPosting]


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def run_simulation(
    slates: Sequence[RecommendationSlate],
    judge: PersonaJudge,
    behavior_model: PositionAwareBehaviorModel,
    top_k: int = 10,
) -> SimulationSummary:
    """运行评估并返回结构化结果；同一 persona/job 只能出现一次。"""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    records: List[SimulationRecord] = []
    seen_pairs = set()
    for slate in slates:
        for rank, job in enumerate(slate.jobs[:top_k], start=1):
            pair = (slate.persona.persona_id, job.id)
            if pair in seen_pairs:
                raise ValueError(f"duplicate persona/job pair: {pair}")
            seen_pairs.add(pair)
            judgment = judge.judge(slate.persona, job)
            behavior = behavior_model.simulate(
                slate.persona.persona_id,
                job.id,
                rank,
                judgment.judgment,
            )
            records.append(
                SimulationRecord(
                    persona_id=slate.persona.persona_id,
                    job_id=job.id,
                    rank=rank,
                    judgment=judgment,
                    behavior=behavior,
                )
            )

    total = len(records)
    feedback_records = [row for row in records if row.behavior.feedback_given]
    source_counts = Counter(row.judgment.source for row in records)
    judge_mode = (
        "llm"
        if source_counts and set(source_counts) == {"llm"}
        else (
            "deterministic"
            if source_counts and set(source_counts) == {"deterministic"}
            else "mixed_with_fallback"
        )
    )
    return SimulationSummary(
        protocol="synthetic-persona-judgment-plus-position-aware-behavior",
        prompt_version=PROMPT_VERSION,
        judge_mode=judge_mode,
        top_k=top_k,
        personas=len({slate.persona.persona_id for slate in slates}),
        impressions=total,
        effective_judgments=sum(row.judgment.judgment.effective for row in records),
        proxy_effectiveness_at_k=_rate(
            sum(row.judgment.judgment.effective for row in records), total
        ),
        simulated_ctr_at_k=_rate(sum(row.behavior.clicked for row in records), total),
        simulated_save_rate_at_k=_rate(
            sum(row.behavior.saved for row in records), total
        ),
        simulated_apply_rate_at_k=_rate(
            sum(row.behavior.applied for row in records), total
        ),
        simulated_feedback_count=len(feedback_records),
        simulated_feedback_effectiveness=(
            _rate(
                sum(row.behavior.satisfied for row in feedback_records),
                len(feedback_records),
            )
            if feedback_records
            else None
        ),
        source_counts={str(key): int(value) for key, value in source_counts.items()},
        records=records,
    )
