"""从训练期曝光拟合 pointwise 排序模型。"""

from __future__ import annotations

from typing import Any

from src.data import DataLoader, GraphEntities

from .feature_builder import FeatureBuilder, FeedRankingFeatures
from .pointwise import PointwiseRanker


def fit_pointwise_from_exposures(
    data: GraphEntities, loader: DataLoader, builder: FeatureBuilder
) -> tuple[PointwiseRanker, dict[str, Any]]:
    users = {user.id: user for user in data.users}
    jobs = list(data.jobs)
    cache: dict[str, dict[str, FeedRankingFeatures]] = {}
    rows: list[FeedRankingFeatures] = []
    labels: list[int] = []
    for exposure in loader.train_exposures:
        user = users[exposure.user_id]
        if user.id not in cache:
            cache[user.id] = builder.build(
                user_id=user.id,
                resume_text=user.resume_text or "",
                user_skills=user.skills,
                jobs=jobs,
                known_user=True,
            )
        rows.append(cache[user.id][exposure.job_id])
        labels.append(int(exposure.engaged))
    ranker = PointwiseRanker.fit(rows, labels)
    positives = sum(labels)
    return ranker, {
        "n_exposures": len(labels),
        "n_positive": positives,
        "positive_rate": positives / max(len(labels), 1),
        "label_definition": "clicked OR saved OR applied OR dwell_seconds>=20",
    }
