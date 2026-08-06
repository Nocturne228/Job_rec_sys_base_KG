import pytest

from src.data import Skill
from src.generation import (
    DeterministicProfileExpander,
    InterestProfile,
    ResilientProfileExpander,
)


def test_interest_expansion_only_uses_terms_with_input_evidence():
    fallback = DeterministicProfileExpander(
        [
            Skill(id="python", name="Python", category="programming"),
            Skill(id="java", name="Java", category="programming"),
        ]
    )
    result = ResilientProfileExpander(fallback).expand(
        "Experienced with Python", ["Backend role with SQL"]
    )
    assert result.mode == "deterministic_fallback"
    assert result.profile.interests == ["Python"]
    assert result.profile.evidence[0].source_text == "Experienced with Python"

    with pytest.raises(ValueError, match="needs input evidence"):
        InterestProfile(interests=["Kubernetes"])
