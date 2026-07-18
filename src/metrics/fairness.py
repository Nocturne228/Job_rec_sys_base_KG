"""Subgroup quality and exposure diagnostics for job recommendation."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List


def subgroup_quality(
    records: Iterable[dict], group_key: str = "group", metric_key: str = "ndcg"
) -> Dict[str, dict]:
    grouped: Dict[str, List[float]] = defaultdict(list)
    for record in records:
        grouped[str(record[group_key])].append(float(record[metric_key]))
    return {
        group: {"count": len(values), "mean": sum(values) / max(len(values), 1)}
        for group, values in grouped.items()
    }


def exposure_parity(
    exposures: Iterable[dict], group_key: str = "group"
) -> Dict[str, float]:
    counts: Dict[str, int] = defaultdict(int)
    users: Dict[str, set] = defaultdict(set)
    for exposure in exposures:
        group = str(exposure[group_key])
        counts[group] += 1
        users[group].add(exposure["user_id"])
    per_user = {group: counts[group] / max(len(users[group]), 1) for group in counts}
    if not per_user:
        return {"max_min_ratio": 1.0}
    minimum = min(per_user.values())
    maximum = max(per_user.values())
    return {**per_user, "max_min_ratio": maximum / max(minimum, 1e-12)}


def experience_group(years: float) -> str:
    if years < 1:
        return "entry"
    if years < 3:
        return "junior"
    if years < 6:
        return "mid"
    return "senior"
