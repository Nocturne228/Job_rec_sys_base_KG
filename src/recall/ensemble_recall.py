"""
Ensemble recall that combines LightGCN and SBERT recall results.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

from .lightgcn import LightGCN
from .sbert_recall import SBERTRecall


@dataclass
class RecallResult:
    """Container for recall results."""
    job_id: str
    lightgcn_score: float = 0.0
    sbert_score: float = 0.0
    combined_score: float = 0.0
    rank: int = 0


class EnsembleRecall:
    """Ensemble recall combining LightGCN and SBERT."""

    def __init__(self,
                 lightgcn_model: LightGCN,
                 sbert_recall: SBERTRecall,
                 lightgcn_weight: float = 0.7,
                 sbert_weight: float = 0.3,
                 fusion_method: str = "weighted_sum"):
        """
        Initialize ensemble recall.

        Args:
            lightgcn_model: Trained LightGCN model
            sbert_recall: SBERT recall model
            lightgcn_weight: Weight for LightGCN scores
            sbert_weight: Weight for SBERT scores
            fusion_method: Fusion method ("weighted_sum", "product", "rank_combination")
        """
        self.lightgcn = lightgcn_model
        self.sbert = sbert_recall
        self.lightgcn_weight = lightgcn_weight
        self.sbert_weight = sbert_weight
        self.fusion_method = fusion_method

        # Validate weights
        total_weight = lightgcn_weight + sbert_weight
        if abs(total_weight - 1.0) > 1e-6:
            # Normalize weights
            self.lightgcn_weight = lightgcn_weight / total_weight
            self.sbert_weight = sbert_weight / total_weight

    def recommend_for_user(self,
                          user_id: str,
                          user_idx: Optional[int] = None,
                          user_embeddings: Optional[np.ndarray] = None,
                          item_embeddings: Optional[np.ndarray] = None,
                          k: int = 10,
                          exclude_interacted: bool = True,
                          interacted_items: Optional[List[str]] = None) -> List[RecallResult]:
        """
        Generate ensemble recommendations for a user.

        Args:
            user_id: User ID
            user_idx: User index in LightGCN (required if using LightGCN)
            user_embeddings: User embeddings from LightGCN (optional)
            item_embeddings: Item embeddings from LightGCN (optional)
            k: Number of recommendations
            exclude_interacted: Whether to exclude interacted items
            interacted_items: List of job IDs the user has interacted with

        Returns:
            List of RecallResult objects sorted by combined score
        """
        # Get LightGCN recommendations if user_idx is provided
        lightgcn_recommendations = []
        if user_idx is not None and user_embeddings is not None and item_embeddings is not None:
            # Convert interacted job IDs to indices if a mapping is provided
            interacted_indices = None
            if exclude_interacted and interacted_items:
                # Caller can pass a dict {job_id: index} or list of indices directly
                if isinstance(interacted_items, dict):
                    # Mapping from data loader: job_id -> integer index
                    interacted_indices = [
                        v for v in interacted_items.values() if isinstance(v, int)
                    ]
                elif isinstance(interacted_items, list) and all(isinstance(x, int) for x in interacted_items):
                    interacted_indices = interacted_items

            # Get LightGCN recommendations
            item_indices, lg_scores = self.lightgcn.recommend_for_user(
                user_idx=user_idx,
                user_embeddings=user_embeddings,
                item_embeddings=item_embeddings,
                k=k * 2,  # Get more candidates for fusion
                exclude_interacted=exclude_interacted,
                interacted_items=interacted_indices
            )

        # Get SBERT recommendations
        sbert_recommendations = self.sbert.recommend_for_user(user_id, k=k * 2)

        # Combine recommendations
        combined = self._fuse_recommendations(
            lightgcn_recommendations,
            sbert_recommendations,
            k=k
        )

        return combined

    def _fuse_recommendations(self,
                             lightgcn_recs: List[Tuple[str, float]],
                             sbert_recs: List[Tuple[str, float]],
                             k: int = 10) -> List[RecallResult]:
        """
        Fuse recommendations from LightGCN and SBERT.

        Args:
            lightgcn_recs: List of (job_id, score) from LightGCN
            sbert_recs: List of (job_id, score) from SBERT
            k: Number of final recommendations

        Returns:
            Fused recommendations
        """
        # Create score dictionaries
        lg_scores = {job_id: score for job_id, score in lightgcn_recs}
        sb_scores = {job_id: score for job_id, score in sbert_recs}

        # Get all unique job IDs
        all_job_ids = set(lg_scores.keys()) | set(sb_scores.keys())

        # Normalize scores to [0, 1] range
        def normalize_scores(scores_dict):
            if not scores_dict:
                return {}
            values = np.array(list(scores_dict.values()))
            if values.max() == values.min():
                return {k: 0.5 for k in scores_dict.keys()}
            normalized = (values - values.min()) / (values.max() - values.min())
            return {k: v for k, v in zip(scores_dict.keys(), normalized)}

        lg_norm = normalize_scores(lg_scores)
        sb_norm = normalize_scores(sb_scores)

        # Combine scores based on fusion method
        combined_scores = {}
        for job_id in all_job_ids:
            lg_score = lg_norm.get(job_id, 0.0)
            sb_score = sb_norm.get(job_id, 0.0)

            if self.fusion_method == "weighted_sum":
                combined = self.lightgcn_weight * lg_score + self.sbert_weight * sb_score
            elif self.fusion_method == "product":
                combined = (lg_score ** self.lightgcn_weight) * (sb_score ** self.sbert_weight)
            elif self.fusion_method == "rank_combination":
                # Get ranks (lower rank is better)
                lg_rank = self._get_rank(job_id, lightgcn_recs)
                sb_rank = self._get_rank(job_id, sbert_recs)

                # Convert ranks to scores (higher is better)
                lg_rank_score = 1.0 / (lg_rank + 1) if lg_rank is not None else 0.0
                sb_rank_score = 1.0 / (sb_rank + 1) if sb_rank is not None else 0.0

                combined = self.lightgcn_weight * lg_rank_score + self.sbert_weight * sb_rank_score
            else:
                raise ValueError(f"Unknown fusion method: {self.fusion_method}")

            combined_scores[job_id] = combined

        # Sort by combined score
        sorted_jobs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        # Create RecallResult objects
        results = []
        for rank, (job_id, combined_score) in enumerate(sorted_jobs, 1):
            result = RecallResult(
                job_id=job_id,
                lightgcn_score=lg_scores.get(job_id, 0.0),
                sbert_score=sb_scores.get(job_id, 0.0),
                combined_score=combined_score,
                rank=rank
            )
            results.append(result)

        return results

    def _get_rank(self, job_id: str, recommendations: List[Tuple[str, float]]) -> Optional[int]:
        """Get rank of a job in recommendations list (0-indexed)."""
        for rank, (jid, _) in enumerate(recommendations):
            if jid == job_id:
                return rank
        return None

    def evaluate_ensemble(self,
                         user_ids: List[str],
                         true_positives: Dict[str, List[str]],
                         k_values: List[int] = [5, 10, 20]) -> Dict[str, Dict[int, float]]:
        """
        Evaluate ensemble recall performance.

        Args:
            user_ids: List of user IDs to evaluate
            true_positives: Dict mapping user_id to list of relevant job IDs
            k_values: List of k values for recall@k

        Returns:
            Dictionary of metrics for each k value
        """
        metrics = {k: {"recall": 0.0, "precision": 0.0} for k in k_values}

        for user_id in user_ids:
            # Get recommendations (simplified - need proper LightGCN indices)
            recommendations = self.recommend_for_user(user_id, k=max(k_values))

            # Get recommended job IDs
            rec_job_ids = [rec.job_id for rec in recommendations]

            # Get true positive job IDs for this user
            true_pos = set(true_positives.get(user_id, []))

            # Calculate metrics for each k
            for k in k_values:
                rec_at_k = set(rec_job_ids[:k])
                relevant_at_k = rec_at_k & true_pos

                recall = len(relevant_at_k) / max(len(true_pos), 1)
                precision = len(relevant_at_k) / k

                metrics[k]["recall"] += recall
                metrics[k]["precision"] += precision

        # Average over users
        n_users = len(user_ids)
        for k in k_values:
            metrics[k]["recall"] /= n_users
            metrics[k]["precision"] /= n_users

        return metrics

    def get_fusion_parameters(self) -> Dict[str, Any]:
        """Get current fusion parameters."""
        return {
            "lightgcn_weight": self.lightgcn_weight,
            "sbert_weight": self.sbert_weight,
            "fusion_method": self.fusion_method,
            "total_weight": self.lightgcn_weight + self.sbert_weight
        }

    def update_weights(self, lightgcn_weight: float, sbert_weight: float) -> None:
        """Update fusion weights."""
        self.lightgcn_weight = lightgcn_weight
        self.sbert_weight = sbert_weight

        # Normalize
        total_weight = lightgcn_weight + sbert_weight
        if total_weight > 0:
            self.lightgcn_weight = lightgcn_weight / total_weight
            self.sbert_weight = sbert_weight / total_weight