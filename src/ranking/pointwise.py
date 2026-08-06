"""带可加 logit 解释的轻量 pointwise 学习排序。"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from .feature_builder import FEED_FEATURE_NAMES, FeedRankingFeatures


class PointwiseRanker:
    def __init__(
        self,
        means: Sequence[float],
        scales: Sequence[float],
        coefficients: Sequence[float],
        intercept: float,
    ) -> None:
        size = len(FEED_FEATURE_NAMES)
        if not all(len(values) == size for values in (means, scales, coefficients)):
            raise ValueError("pointwise ranker parameters do not match feature schema")
        self.means = np.asarray(means, dtype=float)
        self.scales = np.asarray(scales, dtype=float)
        self.scales[self.scales == 0] = 1.0
        self.coefficients = np.asarray(coefficients, dtype=float)
        self.intercept = float(intercept)

    @classmethod
    def fit(
        cls, rows: Sequence[FeedRankingFeatures], labels: Sequence[int]
    ) -> "PointwiseRanker":
        matrix = np.asarray([row.values() for row in rows], dtype=float)
        target = np.asarray(labels, dtype=int)
        if matrix.shape[0] < 2 or len(np.unique(target)) < 2:
            raise ValueError(
                "pointwise training needs both positive and negative exposures"
            )
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0] = 1.0
        normalized = (matrix - means) / scales
        model = LogisticRegression(
            class_weight="balanced", max_iter=300, random_state=42
        )
        model.fit(normalized, target)
        return cls(
            means.tolist(),
            scales.tolist(),
            model.coef_[0].tolist(),
            float(model.intercept_[0]),
        )

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PointwiseRanker":
        if config.get("feature_names") != list(FEED_FEATURE_NAMES):
            raise ValueError("unsupported pointwise feature schema")
        return cls(
            config["means"],
            config["scales"],
            config["coefficients"],
            config["intercept"],
        )

    def to_config(self) -> dict[str, Any]:
        return {
            "kind": "standardized_logistic",
            "feature_names": list(FEED_FEATURE_NAMES),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
        }

    def rank_with_explanations(
        self, rows: Sequence[FeedRankingFeatures]
    ) -> list[tuple[int, float, dict[str, float]]]:
        matrix = np.asarray([row.values() for row in rows], dtype=float)
        normalized = (matrix - self.means) / self.scales
        contributions = normalized * self.coefficients
        logits = contributions.sum(axis=1) + self.intercept
        order = np.argsort(logits, kind="stable")[::-1]
        return [
            (
                int(index),
                float(logits[index]),
                {
                    **{
                        name: float(contributions[index, column])
                        for column, name in enumerate(FEED_FEATURE_NAMES)
                    },
                    "intercept": self.intercept,
                },
            )
            for index in order
        ]
