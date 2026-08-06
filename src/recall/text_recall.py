"""默认离线可复现的文本召回，不下载预训练权重。"""

# mypy: disable-error-code=import-untyped

from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


class TextRecall:
    """使用 feature hashing 表达岗位文本并计算余弦相似度。

    该实现用于展示文本候选信号和冷启动路径，不声称具备预训练语义模型的效果。
    """

    def __init__(self, n_features: int = 512) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            lowercase=True,
            ngram_range=(1, 2),
        )
        self._job_ids: list[str] = []
        self._job_texts: list[str] = []
        self._job_matrix = None

    def add_job(self, job_id: str, text: str) -> None:
        self._job_ids.append(job_id)
        self._job_texts.append(text)
        self._job_matrix = None

    def recommend_for_text(
        self,
        text: str,
        k: int = 10,
        job_ids: Iterable[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not self._job_ids or k <= 0:
            return []
        if self._job_matrix is None:
            self._job_matrix = self.vectorizer.transform(self._job_texts)
        matrix = self._job_matrix
        assert matrix is not None
        allowed = set(job_ids) if job_ids is not None else None
        query = self.vectorizer.transform([text])
        scores = (query @ matrix.T).toarray().ravel()
        ranked = np.argsort(scores)[::-1]
        results: list[tuple[str, float]] = []
        for index in ranked:
            job_id = self._job_ids[int(index)]
            if allowed is not None and job_id not in allowed:
                continue
            results.append((job_id, float(scores[int(index)])))
            if len(results) >= k:
                break
        return results

    def stats(self) -> dict[str, int | str]:
        return {
            "encoder": "feature_hashing",
            "n_jobs": len(self._job_ids),
            "n_features": self.vectorizer.n_features,
        }
