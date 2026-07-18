#!/usr/bin/env python3
"""Train offline artifacts and publish a coherent versioned serving bundle."""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from src.data import DataLoader, generate_mock_data
from src.models import ModelBundle
from src.ranking import GATSkillWeighter
from src.utils.training import train_full_pipeline


def build(output: str, checkpoint: str, seed: int, epochs: int) -> ModelBundle:
    data = generate_mock_data(20, 50, seed=seed)
    loader = DataLoader(data, random_seed=seed)
    results = train_full_pipeline(
        loader,
        config={
            "lightgcn_embedding_dim": 64,
            "lightgcn_n_layers": 3,
            "lightgcn_dropout": 0.0,
            "learning_rate": 0.001,
            "weight_decay": 1e-4,
            "n_epochs": epochs,
            "device": "cpu",
        },
    )
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    results["model"].save(checkpoint)

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
    weighter.train(n_epochs=50, lr=1e-3, weight_decay=1e-4, verbose=False)
    skill_weights = {
        skill.id: weighter.get_skill_weight(skill.id) for skill in data.skills
    }
    train_items = {
        loader.idx_to_user_id[user_idx]: [
            loader.idx_to_job_id[item] for item in loader.train_R[user_idx].indices
        ]
        for user_idx in range(loader.n_users)
    }
    bundle = ModelBundle(
        model_version=f"semi-synthetic-v1-seed{seed}",
        created_at=datetime.now(timezone.utc).isoformat(),
        data_seed=seed,
        lightgcn_checkpoint=os.path.relpath(checkpoint, Path(output).parent),
        user_id_to_idx=loader.user_id_to_idx,
        job_id_to_idx=loader.job_id_to_idx,
        train_items_by_user=train_items,
        skill_weights=skill_weights,
    )
    bundle.save(output)
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/jobrec_bundle.json")
    parser.add_argument("--checkpoint", default="models/lightgcn_model.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    built = build(args.output, args.checkpoint, args.seed, args.epochs)
    print(f"Published {built.model_version} to {args.output}")
