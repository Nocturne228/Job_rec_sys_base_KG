"""
Training utilities for LightGCN model.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import time
from tqdm import tqdm

from recall.lightgcn import LightGCN, prepare_adj_matrix
from data.loader import DataLoader as GraphDataLoader
from config.settings import get_settings


def create_data_loaders(data: Any, test_ratio: float = 0.2) -> Tuple[Any, Any]:
    """
    Create data loaders for training and testing.

    Args:
        data: GraphEntities or similar data
        test_ratio: Ratio of test data

    Returns:
        Tuple of (train_loader, test_loader)
    """
    # This is a simplified version - in practice would use PyTorch DataLoader
    # For now, return the DataLoader instances
    if hasattr(data, '__class__') and data.__class__.__name__ == 'GraphEntities':
        # Create DataLoader from GraphEntities
        data_loader = GraphDataLoader(data)
        return data_loader, data_loader  # Same loader for both in this simplified version
    else:
        # Assume it's already a DataLoader
        return data, data


def train_lightgcn(model: LightGCN,
                  data_loader: Any,
                  n_epochs: int = 100,
                  learning_rate: float = 0.001,
                  weight_decay: float = 1e-4,
                  device: str = "cpu",
                  verbose: bool = True) -> Dict[str, Any]:
    """
    Train LightGCN model.

    Args:
        model: LightGCN model
        data_loader: DataLoader with training data
        n_epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        device: Device to train on
        verbose: Whether to print progress

    Returns:
        Dictionary with training history and metrics
    """
    settings = get_settings()
    model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.Adam(model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay)

    # Prepare adjacency matrix
    adj_matrix = data_loader.get_sparse_graph()
    adj_tensor = prepare_adj_matrix(adj_matrix, device)

    # Get training data
    train_R, test_R, test_users = data_loader.get_train_test_data()

    # Convert to tensors
    train_R_tensor = torch.tensor(train_R.toarray(), device=device)
    test_R_tensor = torch.tensor(test_R.toarray(), device=device)

    # Training history
    history = {
        'loss': [],
        'epoch_time': [],
        'train_metrics': [],
        'test_metrics': []
    }

    # Early stopping
    best_loss = float('inf')
    patience = 10
    patience_counter = 0

    # Training loop
    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Forward pass to get embeddings
        user_embeddings, item_embeddings = model(adj_tensor)

        # Sample training pairs
        n_users, n_items = train_R_tensor.shape
        batch_size = min(1024, n_users * n_items // 10)

        # Simple sampling: get positive interactions
        pos_pairs = torch.nonzero(train_R_tensor > 0)
        if len(pos_pairs) == 0:
            continue

        # Sample batch
        batch_indices = torch.randint(0, len(pos_pairs), (batch_size,))
        batch_pairs = pos_pairs[batch_indices]

        user_ids = batch_pairs[:, 0]
        pos_item_ids = batch_pairs[:, 1]

        # Sample negative items
        neg_item_ids = torch.randint(0, n_items, (batch_size,), device=device)

        # Compute loss
        loss = model.bpr_loss(user_embeddings, item_embeddings,
                             user_ids, pos_item_ids, neg_item_ids)

        # Add regularization
        reg_loss = weight_decay * (
            model.user_embedding.weight.norm(2).pow(2) +
            model.item_embedding.weight.norm(2).pow(2)
        )
        total_loss = loss + reg_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Record history
        epoch_time = time.time() - epoch_start
        history['loss'].append(total_loss.item())
        history['epoch_time'].append(epoch_time)

        # Evaluate periodically
        if (epoch + 1) % 10 == 0 or epoch == n_epochs - 1:
            train_metrics = evaluate_model(model, train_R_tensor, adj_tensor, device, k_values=[20])
            test_metrics = evaluate_model(model, test_R_tensor, adj_tensor, device, k_values=[20])

            history['train_metrics'].append({
                'epoch': epoch,
                **train_metrics
            })
            history['test_metrics'].append({
                'epoch': epoch,
                **test_metrics
            })

            if verbose:
                print(f"Epoch {epoch+1}/{n_epochs}, "
                      f"Loss: {total_loss.item():.4f}, "
                      f"Recall@20: {test_metrics.get('recall@20', 0):.4f}, "
                      f"NDCG@20: {test_metrics.get('ndcg@20', 0):.4f}")

            # Early stopping check
            current_loss = total_loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        elif verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss.item():.4f}")

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train_metrics = evaluate_model(model, train_R_tensor, adj_tensor, device)
        final_test_metrics = evaluate_model(model, test_R_tensor, adj_tensor, device)

    results = {
        'model': model,
        'history': history,
        'final_train_metrics': final_train_metrics,
        'final_test_metrics': final_test_metrics,
        'n_epochs_trained': len(history['loss']),
        'best_loss': min(history['loss']) if history['loss'] else float('inf')
    }

    return results


def evaluate_model(model: LightGCN,
                  R: torch.Tensor,
                  adj_matrix: torch.Tensor,
                  device: str = "cpu",
                  k_values: List[int] = [5, 10, 20]) -> Dict[str, float]:
    """
    Evaluate model performance.

    Args:
        model: LightGCN model
        R: Interaction matrix (binary or weighted)
        adj_matrix: Adjacency matrix
        device: Device to evaluate on
        k_values: List of k values for metrics

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        # Get embeddings
        user_embeddings, item_embeddings = model(adj_matrix)

        # Get test users (those with interactions in R)
        test_users = torch.nonzero(R.sum(dim=1) > 0).squeeze()
        if test_users.dim() == 0:
            test_users = test_users.unsqueeze(0)

        if len(test_users) == 0:
            return {f'recall@{k}': 0.0 for k in k_values}

        # Compute scores for all items for test users
        user_vectors = user_embeddings[test_users]  # (n_test_users, embedding_dim)
        scores = torch.matmul(user_vectors, item_embeddings.T)  # (n_test_users, n_items)

        # Get ground truth
        ground_truth = R[test_users] > 0  # Binary matrix

        # Compute metrics
        metrics = {}
        for k in k_values:
            # Get top-k items for each user
            _, topk_indices = torch.topk(scores, k=k, dim=1)

            # Compute recall@k
            recall_sum = 0.0
            ndcg_sum = 0.0

            for i in range(len(test_users)):
                user_topk = topk_indices[i]
                user_gt = torch.nonzero(ground_truth[i]).squeeze()

                if user_gt.dim() == 0:
                    user_gt = user_gt.unsqueeze(0)

                # Recall@k
                hits = torch.isin(user_topk, user_gt).sum().item()
                recall = hits / min(k, len(user_gt)) if len(user_gt) > 0 else 0.0
                recall_sum += recall

                # NDCG@k
                dcg = 0.0
                for j, item in enumerate(user_topk):
                    if torch.isin(item, user_gt):
                        dcg += 1.0 / torch.log2(torch.tensor(j + 2.0, device=device))

                # Ideal DCG
                ideal_hits = min(k, len(user_gt))
                idcg = sum(1.0 / torch.log2(torch.tensor(j + 2.0, device=device))
                          for j in range(ideal_hits))

                ndcg = dcg / idcg if idcg > 0 else 0.0
                ndcg_sum += ndcg.item()

            metrics[f'recall@{k}'] = recall_sum / len(test_users)
            metrics[f'ndcg@{k}'] = ndcg_sum / len(test_users)

        # Compute precision@k and mrr
        for k in k_values:
            # Precision@k: directly computed as hits / k
            precision_sum = 0.0
            for i in range(len(test_users)):
                user_topk = topk_indices[i]
                user_gt = torch.nonzero(ground_truth[i]).squeeze()
                if user_gt.dim() == 0:
                    user_gt = user_gt.unsqueeze(0)
                hits = torch.isin(user_topk, user_gt.to(user_topk.device)).sum().item()
                precision_sum += hits / k
            metrics[f'precision@{k}'] = precision_sum / len(test_users)

            # MRR@k: mean reciprocal rank
            mrr_sum = 0.0
            for i in range(len(test_users)):
                user_topk = topk_indices[i]
                user_gt = torch.nonzero(ground_truth[i]).squeeze()
                if user_gt.dim() == 0:
                    user_gt = user_gt.unsqueeze(0)
                if len(user_gt) == 0 or user_gt.numel() == 0:
                    continue
                # Find first hit
                hits_mask = torch.isin(user_topk, user_gt.to(user_topk.device))
                hit_positions = torch.nonzero(hits_mask)
                if hit_positions.numel() > 0:
                    first_hit = hit_positions[0].item() + 1  # 1-indexed
                    mrr_sum += 1.0 / first_hit
            metrics[f'mrr@{k}'] = mrr_sum / len(test_users)

        # HitRate@k: fraction of users with at least one hit in top-k
        for k in k_values:
            hit_count = 0
            for i in range(len(test_users)):
                user_topk = topk_indices[i]
                user_gt = torch.nonzero(ground_truth[i]).squeeze()
                if user_gt.dim() == 0:
                    user_gt = user_gt.unsqueeze(0)
                if len(user_gt) > 0 and user_gt.numel() > 0:
                    if torch.isin(user_topk, user_gt.to(user_topk.device)).any():
                        hit_count += 1
            metrics[f'hitrate@{k}'] = hit_count / len(test_users)

        # Catalog Coverage@k: fraction of unique items recommended across all users
        for k in k_values:
            all_recommended = set()
            for i in range(len(test_users)):
                items = topk_indices[i].tolist()
                all_recommended.update(items[:k])
            metrics[f'coverage@{k}'] = len(all_recommended) / model.n_items

        # Compute AUC (simplified)
        try:
            # Sample some negative items for AUC calculation
            n_samples = min(1000, scores.numel())
            flat_scores = scores.flatten()
            flat_labels = ground_truth.flatten().float()

            # Random sample for efficiency
            indices = torch.randint(0, len(flat_scores), (n_samples,))
            sample_scores = flat_scores[indices]
            sample_labels = flat_labels[indices]

            # Sort by score
            sorted_indices = torch.argsort(sample_scores, descending=True)
            sorted_labels = sample_labels[sorted_indices]

            # Compute AUC using trapezoidal rule
            cum_sum = torch.cumsum(sorted_labels, dim=0)
            auc = torch.sum(cum_sum * (1.0 - sorted_labels)) / (
                torch.sum(sorted_labels) * torch.sum(1.0 - sorted_labels)
            )
            metrics['auc'] = auc.item() if not torch.isnan(auc) else 0.0
        except:
            metrics['auc'] = 0.0

    return metrics


def train_full_pipeline(data: Any,
                       config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Train full pipeline: LightGCN + optionally other components.

    Args:
        data: Training data
        config: Configuration dictionary

    Returns:
        Dictionary with trained models and metrics
    """
    settings = get_settings()

    # Use config or settings
    if config is None:
        config = {
            'lightgcn_embedding_dim': settings.model.lightgcn_embedding_dim,
            'lightgcn_n_layers': settings.model.lightgcn_n_layers,
            'lightgcn_dropout': settings.model.lightgcn_dropout,
            'learning_rate': settings.model.lightgcn_learning_rate,
            'weight_decay': settings.model.lightgcn_weight_decay,
            'n_epochs': 50,
            'device': settings.system.device.value
        }

    # Create data loader
    if hasattr(data, '__class__') and data.__class__.__name__ == 'GraphEntities':
        data_loader = GraphDataLoader(data)
    else:
        data_loader = data

    # Get data dimensions (DataLoader uses n_users and n_jobs)
    n_users = data_loader.n_users
    n_items = data_loader.n_jobs

    # Create model
    model = LightGCN(
        n_users=n_users,
        n_items=n_items,
        embedding_dim=config['lightgcn_embedding_dim'],
        n_layers=config['lightgcn_n_layers'],
        dropout=config['lightgcn_dropout'],
        device=config['device']
    )

    # Train model
    results = train_lightgcn(
        model=model,
        data_loader=data_loader,
        n_epochs=config['n_epochs'],
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=config['device'],
        verbose=True
    )

    # Add configuration to results
    results['config'] = config
    results['data_stats'] = {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': data_loader.R.nnz
    }

    return results


def save_training_results(results: Dict[str, Any], path: str) -> None:
    """Save training results to file."""
    import pickle

    # Don't save the model in the results (save separately)
    saved_results = results.copy()
    if 'model' in saved_results:
        del saved_results['model']

    with open(path, 'wb') as f:
        pickle.dump(saved_results, f)


def load_training_results(path: str) -> Dict[str, Any]:
    """Load training results from file."""
    import pickle
    with open(path, 'rb') as f:
        return pickle.load(f)