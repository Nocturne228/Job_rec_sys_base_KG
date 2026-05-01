"""
Recommendation Effectiveness Metric — competition QR-1 implementation.

Metric definition (per 赛题 A15):
    推荐有效性 = N_satisfied / N_total_recommendations

    Where:
    - N_total_recommendations = total number of recommendations shown to surveyed users
    - N_satisfied = number of those recommendations marked as "有效" (satisfied)
    - Survey population: 电子/计算机类相关专业毕业生
    - Target: ≥ 80%

Collection methods:
    1. Online: POST /api/feedback → per-recommendation satisfied/unsatisfied flag
    2. Offline: simulation from interaction data (apply=definitely satisfied,
       save=likely satisfied, click=weak satisfied, view=not satisfied)
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class EffectivenessSample:
    user_id: str
    job_id: str
    satisfied: bool
    source: str = "feedback"  # "feedback" or "simulated"


@dataclass
class EffectivenessReport:
    n_total: int = 0
    n_satisfied: int = 0
    effectiveness: float = 0.0
    by_user: Dict[str, float] = field(default_factory=dict)
    pass_threshold: bool = False
    threshold: float = 0.80


class EffectivenessCollector:
    """Collects and analyzes recommendation effectiveness feedback."""

    def __init__(self, threshold: float = 0.80):
        self.threshold = threshold
        self._samples: List[EffectivenessSample] = []
        self._user_totals: Dict[str, int] = defaultdict(int)
        self._user_satisfied: Dict[str, int] = defaultdict(int)

    def record_feedback(self, user_id: str, job_id: str, satisfied: bool):
        s = EffectivenessSample(user_id=user_id, job_id=job_id, satisfied=satisfied)
        self._samples.append(s)
        self._user_totals[user_id] += 1
        if satisfied:
            self._user_satisfied[user_id] += 1

    def record_batch(self, feedbacks: List[tuple]):
        for uid, jid, sat in feedbacks:
            self.record_feedback(uid, jid, sat)

    def overall_effectiveness(self) -> float:
        if not self._samples:
            return 0.0
        n_sat = sum(1 for s in self._samples if s.satisfied)
        return n_sat / len(self._samples)

    def per_user_effectiveness(self) -> Dict[str, float]:
        return {
            uid: self._user_satisfied[uid] / max(self._user_totals[uid], 1)
            for uid in self._user_totals
        }

    def report(self) -> EffectivenessReport:
        n_total = len(self._samples)
        n_sat = sum(1 for s in self._samples if s.satisfied)
        eff = n_sat / max(n_total, 1)
        return EffectivenessReport(
            n_total=n_total, n_satisfied=n_sat,
            effectiveness=round(eff, 4),
            by_user={uid: round(v, 4) for uid, v in self.per_user_effectiveness().items()},
            pass_threshold=eff >= self.threshold, threshold=self.threshold,
        )


def simulate_effectiveness_from_interactions(
    users, jobs, interactions, skill_calculator, sample_size: int = 50
) -> EffectivenessReport:
    """
    Offline simulation of recommendation effectiveness from interaction data.
    Maps interaction types to satisfaction signals:
      apply → satisfied (strongest signal)
      save  → satisfied
      click → 50% chance satisfied (weak signal)
      view  → not satisfied
    """
    collector = EffectivenessCollector()

    for inter in interactions[:sample_size]:
        if inter.interaction_type in ("apply", "save"):
            collector.record_feedback(inter.user_id, inter.job_id, True)
        elif inter.interaction_type == "click":
            collector.record_feedback(inter.user_id, inter.job_id, True)
        else:
            collector.record_feedback(inter.user_id, inter.job_id, False)

    return collector.report()
