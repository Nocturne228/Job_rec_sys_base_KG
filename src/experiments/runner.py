"""Reproducible baseline, ablation, and subgroup evaluation suite."""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List

import numpy as np
import torch

from src.data import DataLoader, generate_mock_data
from src.ranking import GATSkillWeighter, SkillCoverageCalculator
from src.recall import LightGCN, SBERTRecall
from src.recall.lightgcn import prepare_adj_matrix
from src.utils.training import train_lightgcn


def _normalize(values: np.ndarray) -> np.ndarray:
    span = values.max() - values.min()
    return (values - values.min()) / span if span > 0 else np.zeros_like(values)


def _metrics(ranked: List[int], relevant: set[int], k: int = 10) -> Dict[str, float]:
    top = ranked[:k]
    hits = [int(item in relevant) for item in top]
    recall = sum(hits) / max(len(relevant), 1)
    dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
    idcg = sum(1 / math.log2(position + 2) for position in range(min(k, len(relevant))))
    first = next((position + 1 for position, hit in enumerate(hits) if hit), None)
    return {
        "recall@10": recall,
        "ndcg@10": dcg / idcg if idcg else 0.0,
        "mrr@10": 1.0 / first if first else 0.0,
    }


def _gat_calculator(data):
    kg_data = {
        "skills": [
            {"name": s.id, "display_name": s.name, "level": 1, "domain": s.category}
            for s in data.skills
        ],
        "prerequisites": [
            (r.source_skill_id, r.target_skill_id, r.confidence)
            for r in data.skill_relations
        ],
        "job_associations": {
            job.id: list(job.required_skills) + list(job.preferred_skills)
            for job in data.jobs
        },
    }
    weighter = GATSkillWeighter(kg_data=kg_data, num_features=16)
    weighter.train(n_epochs=30, lr=1e-3, weight_decay=1e-4, verbose=False)
    return SkillCoverageCalculator(gat_weighter=weighter)


def _single_seed(seed: int, epochs: int = 15) -> Dict[str, Dict[str, float]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    data = generate_mock_data(40, 100, seed=seed)
    loader = DataLoader(data, random_seed=seed)
    model = LightGCN(loader.n_users, loader.n_jobs, embedding_dim=32, n_layers=2)
    trained = train_lightgcn(model, loader, n_epochs=epochs, verbose=False)
    adjacency = prepare_adj_matrix(loader.get_sparse_graph())
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model(adjacency)

    sbert = SBERTRecall(use_faiss=False, use_pretrained=False)
    for user in loader.users:
        sbert.add_user(user.id, user.resume_text or "")
    for job in loader.jobs:
        sbert.add_job(job.id, job.description)

    uniform = SkillCoverageCalculator()
    gat = _gat_calculator(data)
    popularity = np.asarray(loader.train_R.sum(axis=0)).ravel()
    per_model = defaultdict(list)
    catalogues = defaultdict(set)

    for user_idx in loader.test_users:
        user = loader.users[user_idx]
        relevant = set(loader.test_R[user_idx].indices.tolist())
        seen = set(loader.train_R[user_idx].indices.tolist())
        lg = (user_embeddings[user_idx] @ item_embeddings.T).cpu().numpy()
        semantic_by_id = dict(sbert.recommend_for_user(user.id, k=loader.n_jobs))
        semantic = np.array([semantic_by_id.get(job.id, 0.0) for job in loader.jobs])
        skill = np.array(
            [
                uniform.calculate_coverage(
                    user.skills, job.required_skills, job.preferred_skills
                )["coverage_score"]
                for job in loader.jobs
            ]
        )
        gat_skill = np.array(
            [
                gat.calculate_coverage(
                    user.skills, job.required_skills, job.preferred_skills
                ).get("gat_coverage_score", 0.0)
                for job in loader.jobs
            ]
        )
        random_scores = np.random.default_rng(seed * 10_000 + user_idx).random(
            loader.n_jobs
        )
        score_sets = {
            "B0_random": random_scores,
            "B1_popularity": popularity,
            "B2_skill": skill,
            "B3_sbert": semantic,
            "B4_lightgcn": lg,
            "E1_lg_sbert": 0.7 * _normalize(lg) + 0.3 * _normalize(semantic),
            "E2_plus_skill": 0.4 * _normalize(lg)
            + 0.3 * _normalize(semantic)
            + 0.3 * skill,
            "E3_plus_gat": 0.4 * _normalize(lg)
            + 0.3 * _normalize(semantic)
            + 0.3 * gat_skill,
        }
        for name, scores in score_sets.items():
            safe = np.asarray(scores, dtype=float).copy()
            if seen:
                safe[list(seen)] = -np.inf
            ranked = np.argsort(safe)[::-1].tolist()
            metric = _metrics(ranked, relevant)
            per_model[name].append(metric)
            catalogues[name].update(ranked[:10])

    results = {}
    for name, rows in per_model.items():
        results[name] = {
            key: mean(row[key] for row in rows)
            for key in ("recall@10", "ndcg@10", "mrr@10")
        }
        results[name]["catalogue_coverage@10"] = len(catalogues[name]) / max(
            loader.n_jobs, 1
        )
    return results


def run_experiment_suite(
    seeds: Iterable[int] = (11, 19, 23, 31, 42),
    epochs: int = 15,
    output_path: str = "results/experiment_summary.json",
) -> Dict[str, dict]:
    seeds = tuple(seeds)
    raw = {str(seed): _single_seed(seed, epochs=epochs) for seed in seeds}
    models = next(iter(raw.values())).keys()
    aggregate: Dict[str, dict] = {}
    for model in models:
        aggregate[model] = {}
        for metric in next(iter(raw.values()))[model]:
            values = [result[model][metric] for result in raw.values()]
            aggregate[model][metric] = {"mean": mean(values), "std": pstdev(values)}
    payload = {
        "seeds": list(seeds),
        "epochs": epochs,
        "raw": raw,
        "aggregate": aggregate,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
