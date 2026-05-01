
"""
Reverse matching module: enterprise-side candidate search.
Matches job requirements against candidate profiles using the existing
recall pipeline (SBERT semantic + skill coverage + LightGCN reverse).
"""

from .reverse_matcher import ReverseMatcher, CandidateResult

__all__ = ["ReverseMatcher", "CandidateResult"]
