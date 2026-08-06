"""共享候选池和切分协议下的最小基线与融合实验。"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

import numpy as np

from src.data import DataLoader, JobPosting, generate_mock_data
from src.models import data_fingerprint
from src.ranking import (
    DiversityReranker,
    FeatureBuilder,
    LinearFusionRanker,
    RankingFeatures,
    fit_pointwise_from_exposures,
)
from src.recall import TextRecall, merge_recall_routes
from src.utils.training import rank_scores, train_lightgcn


def _metrics(ranked: list[int], relevant: set[int], k: int = 10) -> dict[str, float]:
    hits = [int(item in relevant) for item in ranked[:k]]
    recall = sum(hits) / max(len(relevant), 1)
    dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
    idcg = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    first = next((position + 1 for position, hit in enumerate(hits) if hit), None)
    return {
        "recall@10": recall,
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "mrr@10": 1.0 / first if first else 0.0,
    }


def _distribution_metrics(
    ranked: list[int], jobs: list[JobPosting], k: int = 10
) -> dict[str, float]:
    selected = [jobs[index] for index in ranked[:k]]
    distances: list[float] = []
    for left_index, left in enumerate(selected):
        left_skills = set(left.required_skills) | set(left.preferred_skills)
        for right in selected[left_index + 1 :]:
            right_skills = set(right.required_skills) | set(right.preferred_skills)
            union = left_skills | right_skills
            similarity = len(left_skills & right_skills) / len(union) if union else 0.0
            distances.append(1.0 - similarity)
    newest = max(datetime.fromisoformat(job.posted_at) for job in jobs)
    fresh = sum(
        (newest - datetime.fromisoformat(job.posted_at)).days <= 14 for job in selected
    )
    return {
        "intra_list_diversity@10": mean(distances) if distances else 0.0,
        "fresh_job_share@10": fresh / max(len(selected), 1),
    }


def _single_seed(
    seed: int, epochs: int
) -> tuple[dict[str, dict[str, float]], dict[str, object]]:
    data = generate_mock_data(40, 100, seed=seed)
    loader = DataLoader(data, random_seed=seed)
    trained = train_lightgcn(loader, epochs=epochs, seed=seed)
    user_embeddings = trained["user_embeddings"].numpy()
    item_embeddings = trained["item_embeddings"].numpy()
    text = TextRecall()
    for job in loader.jobs:
        text.add_job(job.id, job.description)
    ranker = LinearFusionRanker()
    feature_builder = FeatureBuilder(
        data,
        loader,
        text,
        trained["user_embeddings"],
        trained["item_embeddings"],
    )
    pointwise, pointwise_stats = fit_pointwise_from_exposures(
        data, loader, feature_builder
    )
    reranker = DiversityReranker()
    popularity = np.asarray(loader.train_R.sum(axis=0)).ravel()
    per_model: dict[str, list[dict[str, float]]] = defaultdict(list)
    catalogues: dict[str, set[int]] = defaultdict(set)

    for user_idx in loader.test_users:
        user = loader.users[user_idx]
        relevant = set(loader.test_R[user_idx].indices.tolist())
        seen = set(loader.train_R[user_idx].indices.tolist())
        lightgcn = user_embeddings[user_idx] @ item_embeddings.T
        allowed_indices = [index for index in range(loader.n_jobs) if index not in seen]
        allowed_jobs = [loader.jobs[index] for index in allowed_indices]
        feature_by_job = feature_builder.build(
            user_id=user.id,
            resume_text=user.resume_text or "",
            user_skills=user.skills,
            jobs=allowed_jobs,
            known_user=True,
        )
        text_scores = np.asarray(
            [
                feature_by_job[job.id].text if job.id in feature_by_job else 0.0
                for job in loader.jobs
            ]
        )
        skill_scores = np.asarray(
            [
                feature_by_job[job.id].skill if job.id in feature_by_job else 0.0
                for job in loader.jobs
            ]
        )
        random_scores = np.random.default_rng(seed * 10_000 + user_idx).random(
            loader.n_jobs
        )
        fusion_rows = [
            RankingFeatures(
                lightgcn=feature_by_job[job.id].lightgcn,
                text=feature_by_job[job.id].text,
                skill=feature_by_job[job.id].skill,
            )
            for job in allowed_jobs
        ]
        fusion_scores = np.full(loader.n_jobs, -np.inf)
        for index, score in ranker.rank(fusion_rows):
            fusion_scores[allowed_indices[index]] = score
        pointwise_scores = np.full(loader.n_jobs, -np.inf)
        pointwise_rows = [feature_by_job[job.id] for job in allowed_jobs]
        for index, score, _ in pointwise.rank_with_explanations(pointwise_rows):
            pointwise_scores[allowed_indices[index]] = score
        score_sets = {
            "random": random_scores,
            "popularity": popularity,
            "skill": skill_scores,
            "text": text_scores,
            "lightgcn": lightgcn,
            "fusion": fusion_scores,
            "pointwise": pointwise_scores,
        }
        for name, scores in score_sets.items():
            ranked = rank_scores(scores, seen)
            per_model[name].append(
                {
                    **_metrics(ranked, relevant),
                    **_distribution_metrics(ranked, loader.jobs),
                }
            )
            catalogues[name].update(ranked[:10])

        candidates = merge_recall_routes(
            feature_builder.recall_routes(feature_by_job, known_user=True),
            [job.id for job in allowed_jobs],
            per_route_k=min(20, len(allowed_jobs)),
        )
        allowed_by_id = {job.id: job for job in allowed_jobs}
        candidate_jobs = [allowed_by_id[row.job_id] for row in candidates]
        candidate_features = [feature_by_job[job.id] for job in candidate_jobs]
        reranked = reranker.rerank(
            pointwise.rank_with_explanations(candidate_features),
            candidate_jobs,
            top_k=10,
        )
        multistage_ranked = [
            loader.job_id_to_idx[candidate_jobs[index].id] for index, _, _ in reranked
        ]
        metrics = {
            **_metrics(multistage_ranked, relevant),
            **_distribution_metrics(multistage_ranked, loader.jobs),
        }
        metrics["candidate_recall@80"] = len(
            relevant & {loader.job_id_to_idx[row.job_id] for row in candidates}
        ) / max(len(relevant), 1)
        per_model["multistage_feed"].append(metrics)
        catalogues["multistage_feed"].update(multistage_ranked[:10])

    return {
        name: {
            **{key: mean(row[key] for row in rows) for key in rows[0]},
            "catalogue_coverage@10": len(catalogues[name]) / max(loader.n_jobs, 1),
        }
        for name, rows in per_model.items()
    }, pointwise_stats


def run_experiment_suite(
    seeds: Iterable[int] = (11, 19, 23, 31, 42),
    epochs: int = 15,
    output_path: str = "results/experiment_summary.json",
) -> dict[str, object]:
    seeds = tuple(seeds)
    seed_results = {str(seed): _single_seed(seed, epochs) for seed in seeds}
    raw = {seed: result[0] for seed, result in seed_results.items()}
    ranking_training = {seed: result[1] for seed, result in seed_results.items()}
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    for model in next(iter(raw.values())):
        aggregate[model] = {}
        for metric in next(iter(raw.values()))[model]:
            values = [result[model][metric] for result in raw.values()]
            aggregate[model][metric] = {"mean": mean(values), "std": pstdev(values)}
    reference_data = generate_mock_data(40, 100, seed=seeds[0])
    payload: dict[str, object] = {
        "schema_version": 3,
        "protocol": "per-user-temporal-holdout-seen-masked-multistage",
        "data_kind": "fixed-seed semi-synthetic",
        "reference_data_sha256": data_fingerprint(reference_data),
        "seeds": list(seeds),
        "epochs": epochs,
        "raw": raw,
        "ranking_training": ranking_training,
        "aggregate": aggregate,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
