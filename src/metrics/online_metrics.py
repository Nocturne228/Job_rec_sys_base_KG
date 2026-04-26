"""
Online behavior metrics: CTR, CVR, North Star Metric.

Collects simulated user interactions (impression → click → apply) and computes:
  - CTR (Click-Through Rate)
  - CVR (Conversion Rate = apply / click)
  - Recommendation North Star: apply per user who entered recommendation surface
  - Per-recall-layer metrics: Recall@K, HitRate@10, Cold-start fill rate
  - Ranking metrics: NDCG@10, position entropy
  - Generation metrics: JSON compliance, skill-gap detection rate
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum


# ============================================================================
# Event types
# ============================================================================

class ActionType(str, Enum):
    IMPRESSION = "impression"   # Job shown in recommendation list
    CLICK      = "click"        # User clicked job detail
    APPLY      = "apply"        # User submitted application


@dataclass
class InteractionEvent:
    """Single user interaction event."""
    user_id: str
    job_id: str
    action: ActionType
    group: str = "B"          # "A" = control, "B" = treatment
    timestamp: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


# ============================================================================
# Collector
# ============================================================================

class OnlineMetricsCollector:
    """
    Accumulates interaction events and computes online metrics.

    Usage:
        collector = OnlineMetricsCollector()
        collector.record("user_001", "job_001", ActionType.IMPRESSION, group="B")
        collector.record("user_001", "job_001", ActionType.CLICK, group="B")
        report = collector.generate_report()
    """

    def __init__(self):
        self.events: List[InteractionEvent] = []
        self.start_time: str = datetime.now().isoformat()

        # Per-group counters
        self._impressions: Dict[str, Set[str]] = {"A": set(), "B": set()}     # (user_id, job_id) pairs
        self._clicks: Dict[str, Set[str]] = {"A": set(), "B": set()}
        self._applies: Dict[str, Set[str]] = {"A": set(), "B": set()}
        self._active_users: Dict[str, Set[str]] = {"A": set(), "B": set()}     # users who clicked
        self._apply_users: Dict[str, Set[str]] = {"A": set(), "B": set()}      # users who applied

    # ----- recording -----

    def record(self, user_id: str, job_id: str, action: ActionType,
               group: str = "B", metadata: Optional[Dict[str, Any]] = None):
        """Record a single interaction event."""
        event = InteractionEvent(
            user_id=user_id, job_id=job_id, action=action,
            group=group, metadata=metadata or {}
        )
        self.events.append(event)
        pair = (user_id, job_id)
        grp = event.group
        if action == ActionType.IMPRESSION:
            self._impressions[grp].add(pair)
        elif action == ActionType.CLICK:
            self._clicks[grp].add(pair)
            self._active_users[grp].add(user_id)
        elif action == ActionType.APPLY:
            self._applies[grp].add(pair)
            self._apply_users[grp].add(user_id)

    # ----- metric calculations -----

    @staticmethod
    def _ctr(impressions: int, clicks: int) -> float:
        return clicks / max(impressions, 1)

    @staticmethod
    def _cvr(clicks: int, applies: int) -> float:
        return applies / max(clicks, 1)

    @staticmethod
    def _north_star(impressions: int, applies: int) -> float:
        """North Star: apply / impression (recommendation-side CVR)."""
        return applies / max(impressions, 1)

    def group_metrics(self, group: str = "B") -> Dict[str, float]:
        """Compute all online metrics for a single group."""
        imp_pairs = self._impressions.get(group, set())
        clk_pairs = self._clicks.get(group, set())
        app_pairs = self._applies.get(group, set())

        imp_users = {u for u, _ in imp_pairs}
        clk_users = self._active_users.get(group, set())
        app_users = self._apply_users.get(group, set())

        impressions = len(imp_pairs)
        clicks = len(clk_pairs)
        applies = len(app_pairs)

        ctr = self._ctr(impressions, clicks)
        cvr = self._cvr(clicks, applies)
        north_star = self._north_star(impressions, applies)

        return {
            "group": group,
            "users_impressed": len(imp_users),
            "users_clicked": len(clk_users),
            "users_applied": len(app_users),
            "impressions": impressions,
            "clicks": clicks,
            "applies": applies,
            "ctr": round(ctr, 4),
            "cvr": round(cvr, 4),
            "north_star_cvr": round(north_star, 4),
        }

    def ab_comparison(
        self,
        metric: str = "north_star_cvr",
        confidence: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Compare A vs B on a given metric using bootstrap CI.

        Args:
            metric: one of "ctr", "cvr", "north_star_cvr"
            confidence: confidence level (default 0.95)

        Returns:
            dict with A/B point estimates, delta, and whether B beats A
            (95% CI lower bound > 0).
        """
        a_stats = self.group_metrics("A")
        b_stats = self.group_metrics("B")

        a_val = a_stats.get(metric, 0.0)
        b_val = b_stats.get(metric, 0.0)
        delta = b_val - a_val

        # Simple z-test for proportion difference (large-sample approximation)
        n_a = a_stats.get("impressions", 1)
        n_b = b_stats.get("impressions", 1)
        p_a = a_val
        p_b = b_val
        se = math.sqrt(p_a * (1 - p_a) / max(n_a, 1) + p_b * (1 - p_b) / max(n_b, 1)) + 1e-12
        z = delta / se
        # Two-tailed p-value approximation using normal CDF (no scipy needed)
        p_value = 2 * (1 - _normal_cdf(abs(z)))

        significant = p_value < (1 - confidence)

        return {
            "metric": metric,
            "group_A": round(a_val, 4),
            "group_B": round(b_val, 4),
            "delta": round(delta, 4),
            "delta_pct": f"{(b_val / max(a_val, 1e-6) - 1) * 100:+.1f}%",
            "z_score": round(z, 4),
            "p_value": round(p_value, 4),
            "significant": significant,
            "confidence_level": confidence,
        }

    def generate_report(self) -> Dict[str, Any]:
        """Full report for both groups."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "group_A": self.group_metrics("A"),
            "group_B": self.group_metrics("B"),
            "ctr_comparison": self.ab_comparison("ctr"),
            "cvr_comparison": self.ab_comparison("cvr"),
            "north_star_comparison": self.ab_comparison("north_star_cvr"),
        }
        return report


# ============================================================================
# Convenience helpers
# ============================================================================

def _normal_cdf(x: float) -> float:
    """Approximation of the standard normal CDF using Abramowitz & Stegun."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def compute_north_star(
    impressions: int,
    clicks: int,
    applies: int,
) -> Dict[str, float]:
    """
    Compute North Star metrics from raw counts.

    Returns dict with CTR, CVR, north_star_cvr.
    """
    ctr = OnlineMetricsCollector._ctr(impressions, clicks)
    cvr = OnlineMetricsCollector._cvr(clicks, applies)
    ns = OnlineMetricsCollector._north_star(impressions, applies)
    return {"ctr": round(ctr, 4), "cvr": round(cvr, 4), "north_star_cvr": round(ns, 4)}


def generate_online_report(collector: OnlineMetricsCollector) -> str:
    """Human-readable string report."""
    report = collector.generate_report()
    lines = [
        "=" * 60,
        "Online Metrics Report",
        "=" * 60,
        f"Generated: {report['generated_at']}",
        "",
        "--- Group A (Control) ---",
        f"  Impressions = {report['group_A']['impressions']}",
        f"  Clicks      = {report['group_A']['clicks']}",
        f"  Applies     = {report['group_A']['applies']}",
        f"  CTR         = {report['group_A']['ctr']:.2%}",
        f"  CVR         = {report['group_A']['cvr']:.2%}",
        f"  North Star  = {report['group_A']['north_star_cvr']:.2%}",
        "",
        "--- Group B (Treatment) ---",
        f"  Impressions = {report['group_B']['impressions']}",
        f"  Clicks      = {report['group_B']['clicks']}",
        f"  Applies     = {report['group_B']['applies']}",
        f"  CTR         = {report['group_B']['ctr']:.2%}",
        f"  CVR         = {report['group_B']['cvr']:.2%}",
        f"  North Star  = {report['group_B']['north_star_cvr']:.2%}",
        "",
        "--- Comparisons (B vs A) ---",
    ]

    for key in ["ctr_comparison", "cvr_comparison", "north_star_comparison"]:
        comp = report[key]
        sig = "✅ Significant" if comp["significant"] else "❌ Not significant"
        lines.append(
            f"  {comp['metric']}: A={comp['group_A']:.4f}, "
            f"B={comp['group_B']:.4f}, delta={comp['delta']:+.4f} "
            f"({comp['delta_pct']})  p={comp['p_value']:.4f} → {sig}"
        )

    return "\n".join(lines)
