"""
GAT-based Skill Weighting Module.
Uses Graph Attention Network to compute learnable skill importance weights.
These weights are incorporated into the skill coverage feature for explainable ranking.
"""

import logging
import numpy as np
import torch
from models.gat import GraphAttentionNetwork, SkillFeatureBuilder

logger = logging.getLogger(__name__)


class GATSkillWeighter:
    """
    GAT wrapper that produces per-skill importance weights for the ranking layer.

    Workflow:
    1. Build skill features from KG data (SkillFeatureBuilder, 16-dim)
    2. Train GAT with pseudo-labels (PageRank + job frequency, MSE loss)
    3. Run GAT forward pass to get skill importance scores
    4. Cache scores for use in SkillCoverageCalculator
    5. Provide explainability: top attention edges per skill

    Integration with ranking:
        skill_coverage_score = Σ_{s ∈ missing_skills} GAT_weight(s) × priority(s)
    """

    def __init__(self, kg_data=None, hidden_dim=32, num_heads=4, num_features=16,
                 edge_attr_dim=1, dropout=0.6):
        self.kg_data = kg_data
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.edge_attr_dim = edge_attr_dim
        self.dropout = dropout

        self.model = None
        self.skill_id_map = None
        self.skill_scores = None      # [num_skills] numpy array
        self.skill_names_by_id = {}   # reverse mapping

        # Feature cache for training
        self._x = None
        self._edge_index = None
        self._edge_attr = None

        if kg_data is not None:
            self._build()

    def _build(self):
        """Build skill features, GAT model, and compute importance scores."""
        logger.info(f"[GATSkillWeighter] Building feature matrix from KG data...")

        feature_builder = SkillFeatureBuilder(num_features=self.num_features)
        self._x, self._edge_index, self._edge_attr, self.skill_id_map = (
            feature_builder.build_from_kg_data(self.kg_data)
        )

        # Reverse mapping for explainability
        for name, idx in self.skill_id_map.items():
            self.skill_names_by_id[idx] = name

        # Initialize GAT model
        self.model = GraphAttentionNetwork(
            num_skill_features=self.num_features,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            edge_attr_dim=self._edge_attr.shape[1] if self._edge_attr.numel() > 0 else 1,
            dropout=self.dropout
        )

        # Default: run inference with untrained weights to cache initial scores
        self.model.eval()
        with torch.no_grad():
            self.skill_scores = (
                self.model.compute_node_importance(self._x, self._edge_index, self._edge_attr).numpy()
            )

    def train(self, n_epochs=200, lr=1e-3, weight_decay=1e-4,
              device="cpu", verbose=True):
        """
        Train GAT with pseudo-labels constructed from PageRank + job frequency.

        Pseudo-label for skill s:
            score(s) = 0.6 * PageRank(s) + 0.4 * normalized_job_freq(s)
        This combines structural centrality (how well-connected the skill is
        in the prerequisite graph) with market signal (how often it appears in jobs).

        After training, the model is set to eval() and scores are cached.

        Args:
            n_epochs: training epochs (default 200)
            lr: learning rate
            weight_decay: L2 regularization
            device: "cpu" or "cuda"
            verbose: print progress
        """
        if self._x is None:
            raise RuntimeError("No KG data. Call _build() or provide kg_data first.")

        n_skills = self._x.shape[0]

        # --- Construct pseudo-labels ---
        # Extract PageRank from feature[4] (already computed by SkillFeatureBuilder)
        page_rank = self._x[:, 4].clone()  # [num_skills]
        # Skill frequency from feature[0]
        skill_freq = self._x[:, 0].clone()  # [num_skills]

        # Combine: 60% PageRank + 40% frequency
        pseudo_labels = 0.6 * page_rank + 0.4 * skill_freq
        # Re-normalize to [0, 1]
        p_min, p_max = pseudo_labels.min(), pseudo_labels.max()
        if p_max - p_min > 1e-8:
            pseudo_labels = (pseudo_labels - p_min) / (p_max - p_min)

        if verbose:
            logger.info(
                f"[GATSkillWeighter] Training with pseudo-labels on {n_skills} skills. "
                f"Label range: [{pseudo_labels.min():.4f}, {pseudo_labels.max():.4f}]"
            )

        # --- Train ---
        history = self.model.train_with_pseudo_labels(
            x=self._x,
            edge_index=self._edge_index,
            edge_attr=self._edge_attr,
            pseudo_labels=pseudo_labels,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
            verbose=verbose,
        )

        logger.info(f"[GATSkillWeighter] GAT training complete. Best loss={min(history['loss']):.4f}")

        # --- Cache scores ---
        with torch.no_grad():
            self.skill_scores = (
                self.model.compute_node_importance(self._x, self._edge_index, self._edge_attr).numpy()
            )

        logger.info(
            f"[GATSkillWeighter] Computed importance scores for {len(self.skill_scores)} skills. "
            f"Mean={self.skill_scores.mean():.4f}, Max={self.skill_scores.max():.4f}"
        )

        return history

    def get_skill_weight(self, skill_name):
        """
        Get the GAT-computed importance weight for a specific skill.
        Used in ranking layer's skill coverage calculation.

        Args:
            skill_name: skill string (e.g., "Python", "Machine Learning")

        Returns:
            float: importance score in [0, 1], or 0.0 if skill not in KG
        """
        if self.skill_scores is None or skill_name not in self.skill_id_map:
            logger.debug(f"[GATSkillWeighter] Skill '{skill_name}' not found in KG, weight=0.0")
            return 0.0

        skill_id = self.skill_id_map[skill_name]
        return float(self.skill_scores[skill_id])

    def get_top_k_skills(self, k=10):
        """
        Get top-k most important skills by GAT score (for explainability).

        Returns:
            list[tuple[str, float]]: [(skill_name, weight), ...], sorted by weight desc
        """
        if self.skill_scores is None:
            return []

        # Get indices sorted by score
        top_ids = np.argsort(self.skill_scores)[::-1][:k]
        return [(self.skill_names_by_id[idx], float(self.skill_scores[idx])) for idx in top_ids]

    def get_explainability_report(self, skill_name):
        """
        Generate explainability info for a specific skill.

        Returns:
            dict: {
                "skill": str,
                "importance_score": float,
                "percentile": float,       # How this skill ranks vs. all others
                "top_reasons": list[str]   # Human-readable explanations
            }
        """
        if skill_name not in self.skill_id_map:
            return {"skill": skill_name, "importance_score": 0.0, "error": "Skill not in KG"}

        skill_id = self.skill_id_map[skill_name]
        score = float(self.skill_scores[skill_id])
        percentile = float(np.mean(self.skill_scores <= score))

        # Generate human-readable reasons
        top_skills = self.get_top_k_skills(k=5)
        rank = next((i + 1 for i, (sn, _) in enumerate(top_skills) if sn == skill_name), "unranked")

        reasons = []
        if percentile >= 0.9:
            reasons.append(f"{skill_name} is in the top 10% most impactful skills across all jobs")
        elif percentile >= 0.7:
            reasons.append(f"{skill_name} is a highly referenced skill in the skill prerequisite graph")

        if rank <= 5 and isinstance(rank, int):
            reasons.append(f"Ranked #{rank} overall in skill importance")

        if not reasons:
            reasons.append(f"{skill_name} contributes moderately to career path decisions")

        return {
            "skill": skill_name,
            "importance_score": round(score, 4),
            "percentile": round(percentile, 2),
            "rank": rank,
            "top_reasons": reasons
        }

    @staticmethod
    def integrate_with_ranking(gat_weights, base_ranking_scores, coverage_factor=0.1):
        """
        Integrate GAT skill weights into the overall ranking score.

        Formula:
            final_score = α · Sim_graph + β · Sim_semantic + γ · Σ GAT_weight(s) · I(s ∈ skills)
                          where coverage_factor = γ

        Args:
            gat_weights: dict {skill_name: float} — GAT importance scores
            base_ranking_scores: dict {job_id: float} — current linear rankings
            coverage_factor: how much GAT scores influence final ranking (default 0.1)

        Returns:
            dict {job_id: float} — adjusted ranking scores
        """
        adjusted = {}
        max_gat_score = max(gat_weights.values(), default=1.0) + 1e-8

        for job_id, base_score in base_ranking_scores.items():
            # Parse job skills from job_id or metadata would be needed here
            # For now, the caller integrates gat_weights at the SkillCoverageCalculator level
            adjusted[job_id] = base_score  # Placeholder — actual integration in ranking layer

        return adjusted
