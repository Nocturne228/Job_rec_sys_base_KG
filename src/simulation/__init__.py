"""离线合成用户评估：潜在匹配判断与可观察行为分层建模。"""

from .behavior import BehaviorConfig, PositionAwareBehaviorModel
from .judge import (
    PROMPT_VERSION,
    DeterministicPersonaJudge,
    PromptedLLMPersonaJudge,
    build_judgment_prompt,
)
from .runner import RecommendationSlate, run_simulation
from .schemas import (
    BehaviorOutcome,
    JudgmentEnvelope,
    Persona,
    PersonaJobJudgment,
    SimulationArtifact,
    SimulationRecord,
    SimulationSummary,
)

__all__ = [
    "PROMPT_VERSION",
    "BehaviorConfig",
    "BehaviorOutcome",
    "DeterministicPersonaJudge",
    "JudgmentEnvelope",
    "Persona",
    "PersonaJobJudgment",
    "PositionAwareBehaviorModel",
    "PromptedLLMPersonaJudge",
    "RecommendationSlate",
    "SimulationRecord",
    "SimulationArtifact",
    "SimulationSummary",
    "build_judgment_prompt",
    "run_simulation",
]
