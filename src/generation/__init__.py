"""证据受限的兴趣扩展；外部 LLM 可选，默认确定性降级。"""

from .profile_expansion import (
    DeterministicProfileExpander,
    ExpansionResult,
    InterestProfile,
    OpenAICompatibleProfileExpander,
    ResilientProfileExpander,
)

__all__ = [
    "DeterministicProfileExpander",
    "ExpansionResult",
    "InterestProfile",
    "OpenAICompatibleProfileExpander",
    "ResilientProfileExpander",
]
