"""从已发布 bundle 构造不含训练期已见岗位的离线推荐列表。"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch

from src.data import DataLoader, generate_mock_data
from src.models.bundle import ModelBundle, StaticSkillWeighter
from src.ranking import LinearFusionRanker, RankingFeatures, SkillCoverageCalculator
from src.recall import LightGCN, SBERTRecall
from src.recall.lightgcn import prepare_adj_matrix

from .runner import RecommendationSlate
from .schemas import Persona


def _level(value: object) -> str:
    return str(getattr(value, "value", value))


def build_published_hybrid_slates(
    bundle_path: str | Path = "models/jobrec_bundle.json",
    user_limit: int = 20,
    top_k: int = 10,
) -> Tuple[List[RecommendationSlate], ModelBundle]:
    """复用发布契约构造 known-user hybrid 列表，且先屏蔽训练期岗位。"""

    if user_limit < 1 or top_k < 1:
        raise ValueError("user_limit and top_k must be positive")
    bundle = ModelBundle.load(bundle_path)
    data = generate_mock_data(20, 50, seed=bundle.data_seed)
    loader = DataLoader(data, random_seed=bundle.data_seed)
    if (
        loader.user_id_to_idx != bundle.user_id_to_idx
        or loader.job_id_to_idx != bundle.job_id_to_idx
    ):
        raise RuntimeError("Model-bundle mappings do not match generated data")

    model = LightGCN.load(bundle.lightgcn_checkpoint)
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(
            prepare_adj_matrix(loader.get_sparse_graph())
        )

    semantic = SBERTRecall(use_faiss=False, use_pretrained=False)
    for user in data.users:
        semantic.add_user(user.id, user.resume_text or "")
    for job in data.jobs:
        semantic.add_job(job.id, job.description)

    coverage = SkillCoverageCalculator(
        gat_weighter=StaticSkillWeighter(bundle.skill_weights)
    )
    ranker = LinearFusionRanker(
        weights=bundle.ranking_weights,
        normalization_mode="query",
    )
    jobs_by_id = {job.id: job for job in data.jobs}
    slates: List[RecommendationSlate] = []
    for user in loader.users[:user_limit]:
        seen = set(bundle.train_items_by_user.get(user.id, []))
        candidates = [job for job in loader.jobs if job.id not in seen]
        candidate_ids = [job.id for job in candidates]
        semantic_scores = dict(
            semantic.recommend_for_user(
                user.id,
                k=len(candidate_ids),
                job_ids=candidate_ids,
            )
        )
        user_idx = bundle.user_id_to_idx[user.id]
        collaborative = (user_embeddings[user_idx] @ item_embeddings.T).detach().cpu()
        features = []
        for job in candidates:
            skill_result = coverage.calculate_coverage(
                {key: _level(value) for key, value in user.skills.items()},
                {key: _level(value) for key, value in job.required_skills.items()},
                {key: _level(value) for key, value in job.preferred_skills.items()},
            )
            weighted = skill_result.get("gat_coverage_score")
            features.append(
                RankingFeatures(
                    lightgcn_score=float(collaborative[bundle.job_id_to_idx[job.id]]),
                    sbert_score=semantic_scores.get(job.id, 0.0),
                    skill_coverage=float(
                        weighted
                        if weighted is not None
                        else skill_result["coverage_score"]
                    ),
                )
            )
        order = ranker.rank(features)[:top_k]
        target_titles = sorted(
            {
                jobs_by_id[job_id].title
                for job_id in bundle.train_items_by_user.get(user.id, [])
                if job_id in jobs_by_id
            }
        )
        persona = Persona(
            persona_id=user.id,
            education=user.education,
            experience_years=user.experience_years,
            skills={key: _level(value) for key, value in user.skills.items()},
            target_titles=target_titles,
            preferences=["skills fit", "role relevance", "growth potential"],
        )
        slates.append(
            RecommendationSlate(
                persona=persona,
                jobs=[candidates[index] for index in order],
            )
        )
    return slates, bundle
