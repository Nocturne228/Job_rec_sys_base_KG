"""
LightGCN implementation for collaborative filtering recall.
Based on the paper: "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import sparse
from typing import List, Tuple, Optional, Dict, Any


class LightGCN(nn.Module):
    """LightGCN model for collaborative filtering."""

    def __init__(self,
                 n_users: int,
                 n_items: int,
                 embedding_dim: int = 64,
                 n_layers: int = 3,
                 dropout: float = 0.0,
                 device: str = "cpu"):
        """
        Initialize LightGCN.

        Args:
            n_users: Number of users
            n_items: Number of items (jobs)
            embedding_dim: Dimension of embedding vectors (d=64)
            n_layers: Number of propagation layers (K=3)
            dropout: Dropout rate
            device: Device to run on
        """
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.device = device

        # Initialize user and item embeddings (Eq. 1 in paper)
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)

        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # Layer weights (alpha_k in Eq. 6, typically 1/(K+1))
        self.alpha = 1.0 / (n_layers + 1)

        # Loss function will be BPR loss (implemented separately)

    def forward(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation through LightGCN.

        Args:
            adj_matrix: Normalized adjacency matrix of shape (n_users + n_items, n_users + n_items)

        Returns:
            user_embeddings: Final user embeddings after propagation
            item_embeddings: Final item embeddings after propagation
        """
        # Initial embeddings (Eq. 1)
        all_embeddings = [torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)]

        # Propagation through layers (Eq. 2-3)
        for layer_idx in range(self.n_layers):
            # Message passing: e^{(k+1)} = A * e^{(k)}
            layer_embeddings = torch.spmm(adj_matrix, all_embeddings[-1])

            # Apply dropout if specified
            if self.dropout > 0 and self.training:
                layer_embeddings = F.dropout(layer_embeddings, p=self.dropout)

            all_embeddings.append(layer_embeddings)

        # Combine all layers (Eq. 6: readout with equal weights)
        final_embeddings = torch.stack(all_embeddings, dim=0).mean(dim=0)

        # Split into user and item embeddings
        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]

        return user_embeddings, item_embeddings

    def get_embeddings(self, adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get user and item embeddings (convenience wrapper for forward)."""
        return self.forward(adj_matrix)

    def predict(self,
                user_embeddings: torch.Tensor,
                item_embeddings: torch.Tensor,
                user_ids: torch.Tensor,
                item_ids: torch.Tensor) -> torch.Tensor:
        """
        Predict scores for user-item pairs.

        Args:
            user_embeddings: User embeddings tensor
            item_embeddings: Item embeddings tensor
            user_ids: User indices
            item_ids: Item indices

        Returns:
            scores: Predicted scores for user-item pairs
        """
        user_vecs = user_embeddings[user_ids]
        item_vecs = item_embeddings[item_ids]

        # Dot product scoring (Eq. 7)
        scores = (user_vecs * item_vecs).sum(dim=1)

        return scores

    def recommend_for_user(self,
                          user_idx: int,
                          user_embeddings: torch.Tensor,
                          item_embeddings: torch.Tensor,
                          k: int = 10,
                          exclude_interacted: bool = True,
                          interacted_items: Optional[List[int]] = None) -> Tuple[List[int], List[float]]:
        """
        Generate recommendations for a user.

        Args:
            user_idx: User index
            user_embeddings: User embeddings tensor
            item_embeddings: Item embeddings tensor
            k: Number of recommendations
            exclude_interacted: Whether to exclude interacted items
            interacted_items: List of item indices the user has interacted with

        Returns:
            item_indices: Recommended item indices
            scores: Recommendation scores
        """
        user_vec = user_embeddings[user_idx].unsqueeze(0)  # Shape: (1, embedding_dim)

        # Compute scores for all items
        scores = torch.matmul(user_vec, item_embeddings.T).squeeze(0)  # Shape: (n_items,)

        # Exclude interacted items if specified
        if exclude_interacted and interacted_items is not None:
            scores[interacted_items] = -float('inf')

        # Get top-k items
        topk_scores, topk_indices = torch.topk(scores, min(k, len(scores)))

        return topk_indices.cpu().tolist(), topk_scores.cpu().tolist()

    def bpr_loss(self,
                 user_embeddings: torch.Tensor,
                 item_embeddings: torch.Tensor,
                 user_ids: torch.Tensor,
                 pos_item_ids: torch.Tensor,
                 neg_item_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute BPR loss (Bayesian Personalized Ranking).

        Args:
            user_embeddings: User embeddings
            item_embeddings: Item embeddings
            user_ids: User indices
            pos_item_ids: Positive item indices
            neg_item_ids: Negative item indices

        Returns:
            loss: BPR loss value
        """
        # Get embeddings
        user_vecs = user_embeddings[user_ids]
        pos_vecs = item_embeddings[pos_item_ids]
        neg_vecs = item_embeddings[neg_item_ids]

        # Compute scores
        pos_scores = (user_vecs * pos_vecs).sum(dim=1)
        neg_scores = (user_vecs * neg_vecs).sum(dim=1)

        # BPR loss: -ln sigmoid(pos_score - neg_score)
        diff = pos_scores - neg_scores
        loss = -torch.log(torch.sigmoid(diff) + 1e-8).mean()

        # L2 regularization (optional)
        # reg_loss = self._l2_loss(user_vecs, pos_vecs, neg_vecs)
        # loss += reg_loss

        return loss

    def _l2_loss(self, *tensors) -> torch.Tensor:
        """Compute L2 regularization loss."""
        loss = 0.0
        for tensor in tensors:
            loss += tensor.norm(2).pow(2)
        return loss

    def save(self, path: str) -> None:
        """Save model state."""
        torch.save({
            'state_dict': self.state_dict(),
            'n_users': self.n_users,
            'n_items': self.n_items,
            'embedding_dim': self.embedding_dim,
            'n_layers': self.n_layers,
            'dropout': self.dropout,
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> 'LightGCN':
        """Load model from saved state."""
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            n_users=checkpoint['n_users'],
            n_items=checkpoint['n_items'],
            embedding_dim=checkpoint['embedding_dim'],
            n_layers=checkpoint['n_layers'],
            dropout=checkpoint['dropout'],
            device=device
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        return model


def prepare_adj_matrix(sparse_adj: sparse.csr_matrix, device: str = "cpu") -> torch.Tensor:
    """
    Prepare adjacency matrix for LightGCN.

    Args:
        sparse_adj: Sparse adjacency matrix from scipy
        device: Device to place tensor on

    Returns:
        adj_tensor: Sparse tensor for PyTorch
    """
    # Convert to COO format
    coo = sparse_adj.tocoo()

    # Create indices and values
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    values = torch.tensor(coo.data, dtype=torch.float)

    # Create sparse tensor
    adj_tensor = torch.sparse_coo_tensor(indices, values, coo.shape, device=device)

    return adj_tensor.coalesce()