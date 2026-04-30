"""
Linear fusion ranking model.
Combines multiple recall scores using weighted linear combination.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class RankingFeatures:
    """Features for ranking a job for a user."""
    # Recall scores
    lightgcn_score: float  # Graph similarity score
    sbert_score: float     # Semantic similarity score
    skill_coverage: float  # Skill coverage ratio (0-1)

    # Additional features (optional)
    popularity_score: float = 0.0  # Job popularity
    salary_score: float = 0.0      # Salary attractiveness (normalized)
    company_prestige: float = 0.0  # Company reputation score


class LinearFusionRanker:
    """Linear fusion ranker with grid search optimization."""

    def __init__(self,
                 weights: Optional[Dict[str, float]] = None,
                 normalize_scores: bool = True):
        """
        Initialize linear fusion ranker.

        Args:
            weights: Dictionary of feature weights. If None, uses default weights.
            normalize_scores: Whether to normalize scores to [0, 1] range
        """
        if weights is None:
            # Default weights from README: ω1·Sim_Graph + ω2·Sim_Semantic + ω3·Coverage_Skill
            self.weights = dict(
                lightgcn_score=0.4,      # ω1
                sbert_score=0.3,         # ω2
                skill_coverage=0.3,      # ω3
                popularity_score=0.0,
                salary_score=0.0,
                company_prestige=0.0,
            )
        else:
            # Always ensure all keys exist — caller may only pass a subset
            self.weights = dict(
                lightgcn_score=weights.get('lightgcn_score', 0.0),
                sbert_score=weights.get('sbert_score', 0.0),
                skill_coverage=weights.get('skill_coverage', 0.0),
                popularity_score=weights.get('popularity_score', 0.0),
                salary_score=weights.get('salary_score', 0.0),
                company_prestige=weights.get('company_prestige', 0.0),
            )

        self.normalize_scores = normalize_scores
        self.feature_stats = {}  # For normalization

    def rank(self,
             features_list: List[RankingFeatures],
             return_scores: bool = False) -> List[int]:
        """
        Rank jobs based on linear fusion of features.

        Args:
            features_list: List of RankingFeatures for each job
            return_scores: Whether to return scores along with ranks

        Returns:
            List of indices sorted by rank (best first)
        """
        if not features_list:
            return []

        # Convert to numpy array
        features_array = self._features_to_array(features_list)

        # Normalize if requested
        if self.normalize_scores:
            features_array = self._normalize_features(features_array, update_stats=False)

        # Apply weights and compute scores
        scores = self._compute_scores(features_array)

        # Get sorted indices (descending score)
        sorted_indices = np.argsort(scores)[::-1]

        if return_scores:
            return sorted_indices, scores[sorted_indices]
        else:
            return sorted_indices.tolist()

    def rank_with_features(self,
                          user_id: str,
                          job_ids: List[str],
                          lightgcn_scores: List[float],
                          sbert_scores: List[float],
                          skill_coverages: List[float],
                          additional_features: Optional[Dict[str, List[float]]] = None) -> List[Tuple[str, float]]:
        """
        Convenience method to rank jobs with raw scores.

        Args:
            user_id: User ID
            job_ids: List of job IDs
            lightgcn_scores: List of LightGCN scores
            sbert_scores: List of SBERT scores
            skill_coverages: List of skill coverage scores
            additional_features: Optional additional features

        Returns:
            List of (job_id, final_score) tuples sorted by score descending
        """
        # Create RankingFeatures objects
        features_list = []
        n_jobs = len(job_ids)

        for i in range(n_jobs):
            # Get additional features if provided
            popularity = additional_features.get('popularity', [0.0] * n_jobs)[i] if additional_features else 0.0
            salary = additional_features.get('salary', [0.0] * n_jobs)[i] if additional_features else 0.0
            prestige = additional_features.get('prestige', [0.0] * n_jobs)[i] if additional_features else 0.0

            features = RankingFeatures(
                lightgcn_score=lightgcn_scores[i],
                sbert_score=sbert_scores[i],
                skill_coverage=skill_coverages[i],
                popularity_score=popularity,
                salary_score=salary,
                company_prestige=prestige
            )
            features_list.append(features)

        # Rank jobs
        sorted_indices, sorted_scores = self.rank(features_list, return_scores=True)

        # Create results
        results = []
        for idx, score in zip(sorted_indices, sorted_scores):
            results.append((job_ids[idx], float(score)))

        return results

    def _features_to_array(self, features_list: List[RankingFeatures]) -> np.ndarray:
        """Convert list of RankingFeatures to numpy array."""
        n_samples = len(features_list)
        n_features = 6  # lightgcn_score, sbert_score, skill_coverage, popularity_score, salary_score, company_prestige

        array = np.zeros((n_samples, n_features))
        for i, features in enumerate(features_list):
            array[i] = [
                features.lightgcn_score,
                features.sbert_score,
                features.skill_coverage,
                features.popularity_score,
                features.salary_score,
                features.company_prestige
            ]

        return array

    def _normalize_features(self, features_array: np.ndarray, update_stats: bool = True) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if update_stats or not self.feature_stats:
            # Compute min and max for each feature
            mins = features_array.min(axis=0)
            maxs = features_array.max(axis=0)
            ranges = maxs - mins

            # Avoid division by zero
            ranges[ranges == 0] = 1.0

            self.feature_stats = {
                'mins': mins,
                'maxs': maxs,
                'ranges': ranges
            }

        # Normalize
        mins = self.feature_stats['mins']
        ranges = self.feature_stats['ranges']
        normalized = (features_array - mins) / ranges

        return normalized

    def _compute_scores(self, features_array: np.ndarray) -> np.ndarray:
        """Compute final scores using linear combination."""
        # Create weight vector
        weight_vector = np.array([
            self.weights['lightgcn_score'],
            self.weights['sbert_score'],
            self.weights['skill_coverage'],
            self.weights['popularity_score'],
            self.weights['salary_score'],
            self.weights['company_prestige']
        ])

        # Ensure weights sum to 1 (optional)
        if np.sum(weight_vector) > 0:
            weight_vector = weight_vector / np.sum(weight_vector)

        # Compute weighted sum
        scores = np.dot(features_array, weight_vector)

        return scores

    def grid_search(self,
                   X_train: List[RankingFeatures],
                   y_train: List[float],  # Target scores or labels
                   param_grid: Optional[Dict[str, List[float]]] = None,
                   cv: int = 5) -> Dict[str, Any]:
        """
        Perform grid search to find optimal weights.

        Args:
            X_train: Training features
            y_train: Training labels (e.g., click-through rate, relevance scores)
            param_grid: Parameter grid for search. If None, uses default grid.
            cv: Number of cross-validation folds

        Returns:
            Dictionary with best parameters and results
        """
        # Default parameter grid
        if param_grid is None:
            param_grid = {
                'lightgcn_weight': [0.2, 0.3, 0.4, 0.5, 0.6],
                'sbert_weight': [0.2, 0.3, 0.4, 0.5],
                'skill_coverage_weight': [0.1, 0.2, 0.3, 0.4]
            }

        # Simple implementation (in production would use sklearn's GridSearchCV)
        best_score = -np.inf
        best_params = None

        # Generate all parameter combinations
        from itertools import product
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for param_combination in product(*param_values):
            params = dict(zip(param_names, param_combination))

            # Skip if sum > 1 (optional constraint)
            if sum(param_combination) > 1.0:
                continue

            # Update weights
            temp_weights = self.weights.copy()
            temp_weights['lightgcn_score'] = params.get('lightgcn_weight', temp_weights['lightgcn_score'])
            temp_weights['sbert_score'] = params.get('sbert_weight', temp_weights['sbert_score'])
            temp_weights['skill_coverage'] = params.get('skill_coverage_weight', temp_weights['skill_coverage'])

            # Evaluate with simple cross-validation
            score = self._cross_validate(X_train, y_train, temp_weights, cv=cv)

            if score > best_score:
                best_score = score
                best_params = params

        # Update weights with best parameters
        if best_params:
            self.weights['lightgcn_score'] = best_params.get('lightgcn_weight', self.weights['lightgcn_score'])
            self.weights['sbert_score'] = best_params.get('sbert_weight', self.weights['sbert_score'])
            self.weights['skill_coverage'] = best_params.get('skill_coverage_weight', self.weights['skill_coverage'])

        return {
            'best_params': best_params,
            'best_score': best_score,
            'final_weights': self.weights
        }

    def _cross_validate(self,
                       X: List[RankingFeatures],
                       y: List[float],
                       weights: Dict[str, float],
                       cv: int = 5) -> float:
        """Simple cross-validation for weight evaluation."""
        n_samples = len(X)
        fold_size = n_samples // cv

        scores = []

        for fold in range(cv):
            # Split indices
            start = fold * fold_size
            end = (fold + 1) * fold_size if fold < cv - 1 else n_samples

            val_indices = list(range(start, end))
            train_indices = list(set(range(n_samples)) - set(val_indices))

            # Split data
            X_train = [X[i] for i in train_indices]
            X_val = [X[i] for i in val_indices]
            y_train = [y[i] for i in train_indices]
            y_val = [y[i] for i in val_indices]

            # Train temporary ranker
            temp_ranker = LinearFusionRanker(weights=weights, normalize_scores=self.normalize_scores)

            # Convert to features array
            X_train_array = temp_ranker._features_to_array(X_train)
            X_val_array = temp_ranker._features_to_array(X_val)

            # Normalize using training stats
            if self.normalize_scores:
                train_min = X_train_array.min(axis=0)
                train_max = X_train_array.max(axis=0)
                train_range = train_max - train_min
                train_range[train_range == 0] = 1.0

                X_train_norm = (X_train_array - train_min) / train_range
                X_val_norm = (X_val_array - train_min) / train_range

                X_train_array = X_train_norm
                X_val_array = X_val_norm

            # Make predictions
            train_scores = temp_ranker._compute_scores(X_train_array)
            val_scores = temp_ranker._compute_scores(X_val_array)

            # Evaluate correlation with target (could use other metrics)
            from scipy.stats import spearmanr
            if len(y_val) > 1:
                correlation, _ = spearmanr(val_scores, y_val)
                scores.append(abs(correlation))  # Use absolute value

        return np.mean(scores) if scores else 0.0

    def get_weights(self) -> Dict[str, float]:
        """Get current weights."""
        return self.weights.copy()

    def set_weights(self, weights: Dict[str, float]) -> None:
        """Set weights."""
        self.weights.update(weights)

    def explain_ranking(self,
                       features: RankingFeatures,
                       top_n: int = 3) -> Dict[str, Any]:
        """Explain ranking decision for a single job."""
        # Convert to array
        features_array = self._features_to_array([features])

        # Normalize if needed
        if self.normalize_scores:
            features_array = self._normalize_features(features_array, update_stats=False)

        # Compute contributions
        weight_vector = np.array([
            self.weights['lightgcn_score'],
            self.weights['sbert_score'],
            self.weights['skill_coverage'],
            self.weights['popularity_score'],
            self.weights['salary_score'],
            self.weights['company_prestige']
        ])

        # Normalize weight vector
        if np.sum(weight_vector) > 0:
            weight_vector = weight_vector / np.sum(weight_vector)

        contributions = features_array[0] * weight_vector
        total_score = np.sum(contributions)

        # Get top contributing features
        feature_names = ['lightgcn_score', 'sbert_score', 'skill_coverage',
                        'popularity_score', 'salary_score', 'company_prestige']
        feature_contributions = list(zip(feature_names, contributions.tolist()))
        feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        return {
            'total_score': float(total_score),
            'contributions': dict(feature_contributions[:top_n]),
            'weighted_features': dict(zip(feature_names, contributions.tolist())),
            'raw_features': {
                'lightgcn_score': features.lightgcn_score,
                'sbert_score': features.sbert_score,
                'skill_coverage': features.skill_coverage,
                'popularity_score': features.popularity_score,
                'salary_score': features.salary_score,
                'company_prestige': features.company_prestige
            }
        }
