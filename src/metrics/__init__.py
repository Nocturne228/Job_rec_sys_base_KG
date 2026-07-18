"""
Metrics and evaluation modules for the job recommendation system.
Includes A/B testing, online metrics, LLM evaluation, and effectiveness tracking.
"""

from .ab_test import ABExperiment, ABTest, ExperimentDesign, sample_size_proportion
from .effectiveness import (
    EffectivenessCollector,
    EffectivenessReport,
    simulate_effectiveness_from_interactions,
)
from .event_store import EventStore
from .fairness import experience_group, exposure_parity, subgroup_quality
from .llm_eval import LLMJudgeEvaluator, RuleBasedScorer
from .online_metrics import ActionType, OnlineMetricsCollector

__all__ = [
    "ABTest",
    "ABExperiment",
    "ExperimentDesign",
    "sample_size_proportion",
    "OnlineMetricsCollector",
    "ActionType",
    "LLMJudgeEvaluator",
    "RuleBasedScorer",
    "EffectivenessCollector",
    "EffectivenessReport",
    "simulate_effectiveness_from_interactions",
    "EventStore",
    "exposure_parity",
    "experience_group",
    "subgroup_quality",
]
