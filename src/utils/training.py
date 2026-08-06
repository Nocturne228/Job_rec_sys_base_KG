"""LightGCN 的可复现 BPR 训练与留出评估。"""

from __future__ import annotations

import math
import random
from typing import Any, Iterable

import numpy as np
import torch

from src.data.loader import DataLoader
from src.recall.lightgcn import LightGCN, prepare_adj_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sample_unobserved_negatives(
    train_r: torch.Tensor, user_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """为每个用户采一个训练期未观察岗位，并返回仍有合法负样本的位置。"""
    negatives: list[torch.Tensor] = []
    positions: list[int] = []
    for position, user_id in enumerate(user_ids.tolist()):
        candidates = torch.nonzero(train_r[user_id] == 0).flatten()
        if candidates.numel():
            negatives.append(candidates[torch.randint(candidates.numel(), (1,))])
            positions.append(position)
    if not negatives:
        empty = torch.empty(0, dtype=torch.long, device=train_r.device)
        return empty, empty
    return (
        torch.cat(negatives).to(train_r.device),
        torch.tensor(positions, dtype=torch.long, device=train_r.device),
    )


def evaluate_embeddings(
    user_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    test_r: torch.Tensor,
    train_r: torch.Tensor,
    k: int = 10,
) -> dict[str, float]:
    """在同一全岗位候选池上评估，并屏蔽训练期已见岗位。"""
    test_users = torch.nonzero(test_r.sum(dim=1) > 0).flatten()
    if not test_users.numel():
        return {"recall@10": 0.0, "ndcg@10": 0.0, "mrr@10": 0.0}
    scores = user_embeddings[test_users] @ item_embeddings.T
    scores = scores.masked_fill(train_r[test_users] > 0, -torch.inf)
    effective_k = min(k, item_embeddings.shape[0])
    ranked = torch.topk(scores, k=effective_k, dim=1).indices
    recalls: list[float] = []
    ndcgs: list[float] = []
    mrrs: list[float] = []
    for row, user_id in enumerate(test_users):
        relevant = set(torch.nonzero(test_r[user_id] > 0).flatten().tolist())
        hits = [int(int(item) in relevant) for item in ranked[row]]
        recalls.append(sum(hits) / max(len(relevant), 1))
        dcg = sum(hit / math.log2(position + 2) for position, hit in enumerate(hits))
        idcg = sum(
            1 / math.log2(position + 2)
            for position in range(min(len(relevant), effective_k))
        )
        ndcgs.append(dcg / idcg if idcg else 0.0)
        first = next((position + 1 for position, hit in enumerate(hits) if hit), None)
        mrrs.append(1.0 / first if first else 0.0)
    return {
        "recall@10": float(np.mean(recalls)),
        "ndcg@10": float(np.mean(ndcgs)),
        "mrr@10": float(np.mean(mrrs)),
    }


def train_lightgcn(
    loader: DataLoader,
    *,
    embedding_dim: int = 32,
    n_layers: int = 2,
    epochs: int = 15,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    model = LightGCN(
        loader.n_users,
        loader.n_jobs,
        embedding_dim=embedding_dim,
        n_layers=n_layers,
    )
    adjacency = prepare_adj_matrix(loader.get_sparse_graph())
    train_r = torch.tensor(loader.train_R.toarray())
    test_r = torch.tensor(loader.test_R.toarray())
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positive_pairs = torch.nonzero(train_r > 0)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        user_embeddings, item_embeddings = model(adjacency)
        user_ids = positive_pairs[:, 0]
        positive_ids = positive_pairs[:, 1]
        negative_ids, positions = sample_unobserved_negatives(train_r, user_ids)
        if not negative_ids.numel():
            break
        loss = model.bpr_loss(
            user_embeddings,
            item_embeddings,
            user_ids[positions],
            positive_ids[positions],
            negative_ids,
        )
        regularization = weight_decay * (
            model.user_embedding.weight.square().sum()
            + model.item_embedding.weight.square().sum()
        )
        total = loss + regularization
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
        losses.append(float(total.detach()))
    model.eval()
    with torch.no_grad():
        users, items = model(adjacency)
    return {
        "model": model,
        "user_embeddings": users,
        "item_embeddings": items,
        "losses": losses,
        "metrics": evaluate_embeddings(users, items, test_r, train_r),
    }


def rank_scores(scores: Iterable[float], seen: set[int]) -> list[int]:
    safe = np.asarray(list(scores), dtype=float)
    if seen:
        safe[list(seen)] = -np.inf
    return np.argsort(safe, kind="stable")[::-1].tolist()
