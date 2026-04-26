"""
A/B Testing framework for the job recommendation system.

Implements:
  - ABTest: statistical test (z-test, t-test, Mann-Whitney U, bootstrap)
  - ABExperiment: experiment lifecycle (design, power analysis, assign, analyze)

Covers all tests described in RecSys.md §12.5:
  - CVR → z-test for proportions
  - Avg clicks → Welch's t-test
  - NDCG → Mann-Whitney U test
  - Sample size calculation (NormalIndPower)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


# ============================================================================
# Statistical test primitives (no scipy)
# ============================================================================

def _normal_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun)."""
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x) / math.sqrt(2)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


def z_test_proportions(p1: float, n1: int, p2: float, n2: int,
                       alternative: str = "greater") -> Dict[str, float]:
    """Two-sample z-test for proportions.

    Args:
        p1, n1: control group proportion and sample size
        p2, n2: treatment group proportion and sample size
        alternative: "greater", "less", "two_sided"
    """
    se = math.sqrt(p1 * (1 - p1) / max(n1, 1) + p2 * (1 - p2) / max(n2, 1)) + 1e-12
    z = (p2 - p1) / se

    if alternative == "greater":
        p_value = 1 - _normal_cdf(z)
    elif alternative == "less":
        p_value = _normal_cdf(z)
    else:
        p_value = 2 * (1 - _normal_cdf(abs(z)))

    return {"z": round(z, 4), "p_value": round(p_value, 4), "se": round(se, 6)}


def welch_t_test(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Welch's t-test (unequal variance)."""
    if len(xs) < 2 or len(ys) < 2:
        return {"t": 0.0, "df": 0, "p_value": 1.0}

    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    var_x = sum((v - mx) ** 2 for v in xs) / (len(xs) - 1)
    var_y = sum((v - my) ** 2 for v in ys) / (len(ys) - 1)

    se = math.sqrt(var_x / len(xs) + var_y / len(ys)) + 1e-12
    t_stat = (my - mx) / se

    # Welch-Satterthwaite degrees of freedom
    num = (var_x / len(xs) + var_y / len(ys)) ** 2
    den = ((var_x / len(xs)) ** 2 / (len(xs) - 1) +
           (var_y / len(ys)) ** 2 / (len(ys) - 1)) + 1e-12
    df = num / den

    p_value = 2 * (1 - _normal_cdf(abs(t_stat)))

    return {"t": round(t_stat, 4), "df": round(df, 2), "p_value": round(p_value, 4)}


def mann_whitney_u(xs: List[float], ys: List[float]) -> Dict[str, float]:
    """Mann-Whitney U test (two-sided, large-sample normal approximation)."""
    n1, n2 = len(xs), len(ys)
    if n1 == 0 or n2 == 0:
        return {"U": 0.0, "z": 0.0, "p_value": 1.0}

    combined = sorted(
        [(v, 0) for v in xs] + [(v, 1) for v in ys],
        key=lambda x: x[0],
    )
    ranks_x = 0.0
    for rank_i, (_, grp) in enumerate(combined, 1):
        if grp == 0:
            ranks_x += rank_i

    U = ranks_x - n1 * (n1 + 1) / 2

    # Normal approximation
    mu = n1 * n2 / 2
    sigma = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12) + 1e-12
    z = (U - mu) / sigma
    p_value = 2 * (1 - _normal_cdf(abs(z)))

    return {"U": round(U, 2), "z": round(z, 4), "p_value": round(p_value, 4)}


def bootstrap_ci(
    xs: List[float], ys: List[float],
    statistic="mean_diff", n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap confidence interval for difference in means.

    Returns delta estimate and CI bounds.
    """
    rng = random.Random(42)
    deltas = []
    for _ in range(n_bootstrap):
        sx = [rng.choice(xs) for _ in range(len(xs))]
        sy = [rng.choice(ys) for _ in range(len(ys))]
        deltas.append(sum(sy) / len(sy) - sum(sx) / len(sx))

    deltas.sort()
    alpha = 1 - confidence
    lo_idx = int(alpha / 2 * n_bootstrap)
    hi_idx = int((1 - alpha / 2) * n_bootstrap)

    delta_obs = sum(ys) / len(ys) - sum(xs) / len(xs)
    return {
        "delta": round(delta_obs, 6),
        "ci_lower": round(deltas[lo_idx], 6),
        "ci_upper": round(deltas[hi_idx], 6),
        "confidence": confidence,
        "significant": deltas[lo_idx] * deltas[hi_idx] > 0,
    }


def sample_size_proportion(
    baseline: float, mde: float, alpha: float = 0.05,
    power: float = 0.8, ratio: float = 1.0,
) -> int:
    """Sample size per group to detect a change from baseline → baseline + mde.

    Uses normal approximation with Z_alpha/2 + Z_beta.
    """
    z_alpha = _normal_ppf(1 - alpha / 2)
    z_beta = _normal_ppf(power)

    p1 = baseline
    p2 = baseline + mde
    p_bar = (p1 + p2 * ratio) / (1 + ratio)

    se_alt = math.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)
    se_null = math.sqrt(p_bar * (1 - p_bar) * (1 + 1 / ratio))

    n = ((z_alpha * se_null + z_beta * se_alt) / (p2 - p1)) ** 2
    return max(1, int(math.ceil(n)))


def _normal_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro)."""
    # Rational approximation for 0 < p < 1
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if abs(p - 0.5) < 1e-10:
        return 0.0

    # Use iterative Newton-Raphson
    x = 0.0
    if p > 0.5:
        x = 0.436  # seed
    else:
        x = -0.436

    for _ in range(20):
        cdf = _normal_cdf(x)
        pdf = math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
        if pdf < 1e-15:
            break
        x = x - (cdf - p) / pdf

    return x


# ============================================================================
# ABTest: run a single test given collected data
# ============================================================================

@dataclass
class ABTestResult:
    """Result of a single metric comparison."""
    metric: str
    test_method: str
    p_value: float
    significant: bool
    details: Dict[str, Any]


class ABTest:
    """Run statistical tests between control (A) and treatment (B)."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compare_proportions(self, metric: str,
                             a_count: int, a_total: int,
                             b_count: int, b_total: int,
                             alternative: str = "greater") -> ABTestResult:
        """z-test for proportions (CVR, CTR)."""
        p_a = a_count / max(a_total, 1)
        p_b = b_count / max(b_total, 1)
        res = z_test_proportions(p_a, a_total, p_b, b_total, alternative)
        return ABTestResult(
            metric=metric,
            test_method="z_test_proportions",
            p_value=res["p_value"],
            significant=res["p_value"] < self.alpha,
            details={
                "p_A": round(p_a, 4),
                "p_B": round(p_b, 4),
                "delta_pct": round((p_b / max(p_a, 1e-6) - 1) * 100, 1),
                "z_score": res["z"],
                "se": res["se"],
            },
        )

    def compare_means(self, metric: str,
                       a_values: List[float], b_values: List[float]) -> ABTestResult:
        """Welch's t-test for means (avg clicks, avg dwell time)."""
        res = welch_t_test(a_values, b_values)
        return ABTestResult(
            metric=metric,
            test_method="welch_t_test",
            p_value=res["p_value"],
            significant=res["p_value"] < self.alpha,
            details={
                "mean_A": round(sum(a_values) / max(len(a_values), 1), 4),
                "mean_B": round(sum(b_values) / max(len(b_values), 1), 4),
                "t": res["t"],
                "df": res["df"],
            },
        )

    def compare_distributions(self, metric: str,
                               a_values: List[float], b_values: List[float]) -> ABTestResult:
        """Mann-Whitney U test (NDCG, ranking metrics)."""
        res = mann_whitney_u(a_values, b_values)
        return ABTestResult(
            metric=metric,
            test_method="mann_whitney_u",
            p_value=res["p_value"],
            significant=res["p_value"] < self.alpha,
            details={
                "U_stat": res["U"],
                "z": res["z"],
            },
        )

    def confidence_interval(self, metric: str,
                             a_values: List[float], b_values: List[float],
                             confidence: float = 0.95) -> ABTestResult:
        """Bootstrap CI for delta in means."""
        res = bootstrap_ci(a_values, b_values, confidence=confidence)
        return ABTestResult(
            metric=metric,
            test_method="bootstrap_ci",
            p_value=0.0,  # bootstrap CI doesn't produce p-value
            significant=res["significant"],
            details={
                "delta": res["delta"],
                "ci_lower": res["ci_lower"],
                "ci_upper": res["ci_upper"],
                "confidence": confidence,
            },
        )


# ============================================================================
# ABExperiment: full experiment lifecycle
# ============================================================================

@dataclass
class ExperimentDesign:
    """A/B experiment design parameters."""
    name: str
    baseline_metric: float        # e.g. current CVR = 0.02
    mde: float                    # minimum detectable effect (absolute), e.g. 0.005
    alpha: float = 0.05
    power: float = 0.8
    ratio: float = 1.0            # A:B = 1:1
    duration_days: int = 14
    primary_metric: str = "cvr"   # which metric drives sample size
    secondary_metrics: List[str] = field(default_factory=lambda: ["ctr", "ndcg@10"])


@dataclass
class UserAssignment:
    user_id: str
    group: str       # "A" or "B"


class ABExperiment:
    """
    Full A/B experiment lifecycle.

    1. Design (power analysis → required sample size)
    2. Assign users (hash-based deterministic assignment)
    3. Analyze (run tests)
    """

    def __init__(self, design: ExperimentDesign):
        self.design = design
        self.alpha = design.alpha

        # Sample size calculation
        self.required_per_group = sample_size_proportion(
            baseline=design.baseline_metric,
            mde=design.mde,
            alpha=design.alpha,
            power=design.power,
            ratio=design.ratio,
        )

        self.assignments: Dict[str, str] = {}
        self._metrics_a: Dict[str, List[float]] = {}
        self._metrics_b: Dict[str, List[float]] = {}
        self._counts_a: Dict[str, int] = {}
        self._counts_b: Dict[str, int] = {}
        self._totals_a: Dict[str, int] = {}
        self._totals_b: Dict[str, int] = {}
        self.created_at: str = datetime.now().isoformat()

    # ----- 1. Power analysis -----

    def get_power_analysis(self) -> Dict[str, Any]:
        return {
            "experiment": self.design.name,
            "baseline_metric": self.design.baseline_metric,
            "mde": self.design.mde,
            "mde_pct": f"{(self.design.mde / max(self.design.baseline_metric, 1e-6)) * 100:.1f}%",
            "alpha": self.alpha,
            "power": self.design.power,
            "required_per_group": self.required_per_group,
            "total_required": self.required_per_group * 2,
            "estimated_duration_days": self.design.duration_days,
        }

    # ----- 2. User assignment -----

    def assign_user(self, user_id: str) -> str:
        """Deterministic hash-based assignment (consistent across sessions)."""
        if user_id in self.assignments:
            return self.assignments[user_id]

        h = hash(user_id) % 1000
        group = "B" if h >= 500 else "A"
        self.assignments[user_id] = group
        return group

    def assign_batch(self, user_ids: List[str]) -> Dict[str, int]:
        counts = {"A": 0, "B": 0}
        for uid in user_ids:
            counts[self.assign_user(uid)] += 1
        return counts

    def enrolled_users(self) -> Dict[str, int]:
        return {"A": sum(1 for g in self.assignments.values() if g == "A"),
                "B": sum(1 for g in self.assignments.values() if g == "B")}

    # ----- 3. Metric logging -----

    def log_impression(self, user_id: str):
        grp = self.assignments.get(user_id)
        if grp is None:
            return
        self._totals_a.setdefault(grp, 0)
        self._totals_b.setdefault(grp, 0)
        if grp == "A":
            self._totals_a["total"] = self._totals_a.get("total", 0) + 1
        else:
            self._totals_b["total"] = self._totals_b.get("total", 0) + 1

    def log_click(self, user_id: str, n_clicks: int = 1):
        grp = self.assignments.get(user_id)
        if grp is None:
            return
        target = self._metrics_a if grp == "A" else self._metrics_b
        target.setdefault("clicks", []).append(n_clicks)

    def log_apply(self, user_id: str):
        grp = self.assignments.get(user_id)
        if grp is None:
            return
        target = self._counts_a if grp == "A" else self._counts_b
        target["applies"] = target.get("applies", 0) + 1

    def log_ndcg(self, user_id: str, ndcg: float):
        grp = self.assignments.get(user_id)
        if grp is None:
            return
        target = self._metrics_a if grp == "A" else self._metrics_b
        target.setdefault("ndcg@10", []).append(ndcg)

    # ----- 4. Analysis -----

    def analyze(self) -> Dict[str, ABTestResult]:
        """Run all tests and return dict of metric → result."""
        tester = ABTest(alpha=self.alpha)
        results = {}

        clicks_a = self._metrics_a.get("clicks", [])
        clicks_b = self._metrics_b.get("clicks", [])
        ndcg_a = self._metrics_a.get("ndcg@10", [])
        ndcg_b = self._metrics_b.get("ndcg@10", [])

        applies_a = self._counts_a.get("applies", 0)
        applies_b = self._counts_b.get("applies", 0)
        total_a = self._totals_a.get("total", 1)
        total_b = self._totals_b.get("total", 1)

        # CVR: z-test for proportions
        if applies_a > 0 or applies_b > 0:
            results["cvr"] = tester.compare_proportions(
                "cvr", applies_a, total_a, applies_b, total_b
            )

        # Avg clicks: Welch's t-test
        if clicks_a and clicks_b:
            results["avg_clicks"] = tester.compare_means("avg_clicks", clicks_a, clicks_b)

        # NDCG@10: Mann-Whitney U
        if ndcg_a and ndcg_b:
            results["ndcg@10"] = tester.compare_distributions("ndcg@10", ndcg_a, ndcg_b)

        # Bootstrap CI for clicks
        if clicks_a and clicks_b:
            results["clicks_ci"] = tester.confidence_interval(
                "click_ci", clicks_a, clicks_b, confidence=0.95
            )

        return results

    def report(self) -> Dict[str, Any]:
        """Full experiment report."""
        results = self.analyze()
        return {
            "experiment": self.design.name,
            "created_at": self.created_at,
            "power_analysis": self.get_power_analysis(),
            "enrolled": self.enrolled_users(),
            "results": {
                k: {
                    "metric": v.metric,
                    "test": v.test_method,
                    "p_value": v.p_value,
                    "significant": v.significant,
                    "details": v.details,
                }
                for k, v in results.items()
            },
        }


def print_experiment_report(exp: ABExperiment) -> str:
    """Formatted string report."""
    rpt = exp.report()
    pa = rpt["power_analysis"]
    lines = [
        "=" * 60,
        f"Experiment: {rpt['experiment']}",
        f"Created: {rpt['created_at']}",
        "",
        "--- Power Analysis ---",
        f"  Baseline: {pa['baseline_metric']}",
        f"  MDE: {pa['mde']} ({pa['mde_pct']})  →  {pa['required_per_group']} /group",
        f"  Total required: {pa['total_required']}",
        "",
        f"Enrolled: A={rpt['enrolled']['A']}, B={rpt['enrolled']['B']}",
        "",
        "--- Results ---",
    ]
    for name, r in rpt["results"].items():
        sig = "✅ Significant" if r["significant"] else "❌ Not significant"
        lines.append(f"  {name} ({r['test']}): p={r['p_value']:.4f} → {sig}")

    return "\n".join(lines)
