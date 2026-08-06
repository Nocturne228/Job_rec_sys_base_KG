"""请求级无状态线性融合与可加解释。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

FEATURE_NAMES = ("lightgcn", "text", "skill")


@dataclass(frozen=True)
class RankingFeatures:
    lightgcn: float = 0.0
    text: float = 0.0
    skill: float = 0.0


class LinearFusionRanker:
    """在单次候选集合内归一化，避免跨请求共享拟合状态。"""

    def __init__(self, weights: Mapping[str, float] | None = None) -> None:
        raw = dict(weights or {"lightgcn": 0.4, "text": 0.3, "skill": 0.3})
        if set(raw) != set(FEATURE_NAMES):
            raise ValueError(f"weights must contain exactly {FEATURE_NAMES}")
        if any(value < 0 for value in raw.values()) or sum(raw.values()) <= 0:
            raise ValueError("ranking weights must be non-negative with a positive sum")
        total = sum(raw.values())
        self.weights = {name: float(raw[name] / total) for name in FEATURE_NAMES}

    @staticmethod
    def _matrix(rows: Sequence[RankingFeatures]) -> np.ndarray:
        return np.asarray(
            [[row.lightgcn, row.text, row.skill] for row in rows], dtype=float
        )

    @staticmethod
    def _normalize(matrix: np.ndarray) -> np.ndarray:
        if matrix.size == 0:
            return matrix.copy()
        minimum = matrix.min(axis=0)
        span = matrix.max(axis=0) - minimum
        return np.divide(
            matrix - minimum,
            span,
            out=np.zeros_like(matrix),
            where=span > 0,
        )

    def rank_with_explanations(
        self, rows: Sequence[RankingFeatures]
    ) -> list[tuple[int, float, dict[str, float]]]:
        normalized = self._normalize(self._matrix(rows))
        weight_vector = np.asarray([self.weights[name] for name in FEATURE_NAMES])
        contributions = normalized * weight_vector
        scores = contributions.sum(axis=1)
        order = np.argsort(scores, kind="stable")[::-1]
        return [
            (
                int(index),
                float(scores[index]),
                {
                    name: float(contributions[index, column])
                    for column, name in enumerate(FEATURE_NAMES)
                },
            )
            for index in order
        ]

    def rank(self, rows: Sequence[RankingFeatures]) -> list[tuple[int, float]]:
        return [(index, score) for index, score, _ in self.rank_with_explanations(rows)]
