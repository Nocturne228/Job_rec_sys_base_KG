#!/usr/bin/env python3
"""训练并以不可变 checkpoint + 原子 bundle 指针发布服务产物。"""

from __future__ import annotations

import argparse
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.data import DataLoader, generate_mock_data
from src.models import (
    ModelBundle,
    data_fingerprint,
    serving_fingerprint,
    sha256_file,
)
from src.ranking import FeatureBuilder, fit_pointwise_from_exposures
from src.recall import TextRecall
from src.utils.training import train_lightgcn


def build(output: str, seed: int, epochs: int) -> ModelBundle:
    data = generate_mock_data(20, 50, seed=seed)
    loader = DataLoader(data, random_seed=seed)
    config = {
        "embedding_dim": 32,
        "n_layers": 2,
        "epochs": epochs,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "seed": seed,
    }
    trained = train_lightgcn(loader, **config)
    text_config = {"kind": "feature_hashing", "n_features": 512}
    text = TextRecall(n_features=text_config["n_features"])
    for job in data.jobs:
        text.add_job(job.id, job.description)
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
    ranking_model = pointwise.to_config()
    ranking_model["training"] = pointwise_stats
    reranking_config = {"diversity_penalty": 0.12, "freshness_bonus": 0.05}
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".lightgcn-", suffix=".pt.tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary_checkpoint = Path(temporary_name)
    try:
        trained["model"].save(str(temporary_checkpoint))
        checkpoint_hash = sha256_file(temporary_checkpoint)
        checkpoint = target.parent / f"lightgcn-{checkpoint_hash[:12]}.pt"
        if checkpoint.exists():
            temporary_checkpoint.unlink()
        else:
            os.replace(temporary_checkpoint, checkpoint)
        checkpoint.chmod(0o644)

        data_hash = data_fingerprint(data)
        serving_hash = serving_fingerprint(
            data_sha256=data_hash,
            checkpoint_sha256=checkpoint_hash,
            ranking_model=ranking_model,
            text_config=text_config,
            reranking_config=reranking_config,
        )
        model_version = f"jobrec-feed-{serving_hash[:16]}"
        seen = {
            loader.idx_to_user_id[user_idx]: [
                loader.idx_to_job_id[item] for item in loader.train_R[user_idx].indices
            ]
            for user_idx in range(loader.n_users)
        }
        bundle = ModelBundle(
            model_version=model_version,
            created_at=datetime.now(timezone.utc).isoformat(),
            data_seed=seed,
            data_sha256=data_hash,
            checkpoint_path=checkpoint.name,
            checkpoint_sha256=checkpoint_hash,
            serving_sha256=serving_hash,
            training_config=config,
            user_id_to_idx=loader.user_id_to_idx,
            job_id_to_idx=loader.job_id_to_idx,
            train_items_by_user=seen,
            ranking_model=ranking_model,
            text_config=text_config,
            reranking_config=reranking_config,
        )
        temporary_bundle = target.with_suffix(target.suffix + ".tmp")
        bundle.save(temporary_bundle)
        os.replace(temporary_bundle, target)
        return bundle
    finally:
        temporary_checkpoint.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/jobrec_bundle.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    published = build(args.output, args.seed, args.epochs)
    print(f"Published {published.model_version} to {args.output}")
