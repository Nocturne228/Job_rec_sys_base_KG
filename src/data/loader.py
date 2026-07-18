"""
Data loader for constructing interaction graphs and preparing data for models.
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from .models import GraphEntities, JobPosting, Skill, User


class DataLoader:
    """Load data and construct user-job interaction graph for LightGCN."""

    def __init__(
        self,
        data: GraphEntities,
        min_interactions: int = 1,
        test_ratio: float = 0.2,
        random_seed: int = 42,
    ):
        self.data = data
        self.min_interactions = min_interactions
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        # Create mappings
        self.user_id_to_idx: Dict[str, int] = {}
        self.job_id_to_idx: Dict[str, int] = {}
        self.idx_to_user_id: Dict[int, str] = {}
        self.idx_to_job_id: Dict[int, str] = {}

        self._process_data()

    def _process_data(self) -> None:
        """Process data and create mappings."""
        # Filter users and jobs with sufficient interactions
        user_interaction_counts = defaultdict(int)
        job_interaction_counts = defaultdict(int)

        for interaction in self.data.interactions:
            user_interaction_counts[interaction.user_id] += 1
            job_interaction_counts[interaction.job_id] += 1

        # Filter users and jobs
        self.users = [
            u
            for u in self.data.users
            if user_interaction_counts[u.id] >= self.min_interactions
        ]
        self.jobs = [
            j
            for j in self.data.jobs
            if job_interaction_counts[j.id] >= self.min_interactions
        ]

        # Create mappings
        self.user_id_to_idx = {user.id: idx for idx, user in enumerate(self.users)}
        self.job_id_to_idx = {job.id: idx for idx, job in enumerate(self.jobs)}
        self.idx_to_user_id = {
            idx: user_id for user_id, idx in self.user_id_to_idx.items()
        }
        self.idx_to_job_id = {idx: job_id for job_id, idx in self.job_id_to_idx.items()}

        # Build interaction matrix
        self.n_users = len(self.users)
        self.n_jobs = len(self.jobs)
        self.R = sparse.lil_matrix((self.n_users, self.n_jobs), dtype=np.float32)

        for interaction in self.data.interactions:
            if (
                interaction.user_id in self.user_id_to_idx
                and interaction.job_id in self.job_id_to_idx
            ):
                user_idx = self.user_id_to_idx[interaction.user_id]
                job_idx = self.job_id_to_idx[interaction.job_id]

                # Assign weights based on interaction type
                weight = {"view": 0.5, "click": 1.0, "save": 1.5, "apply": 2.0}.get(
                    interaction.interaction_type, 1.0
                )

                self.R[user_idx, job_idx] = max(self.R[user_idx, job_idx], weight)

        self.R = self.R.tocsr()

        # Build train/test split
        self._create_train_test_split()

    def _create_train_test_split(self) -> None:
        """Create a reproducible per-user holdout split.

        Each user with at least two interactions keeps at least one training
        edge. A global random split can place all of a user's observations in
        test, inadvertently evaluating a cold-start user instead of ranking.
        """
        if not 0.0 <= self.test_ratio < 1.0:
            raise ValueError("test_ratio must be in [0, 1).")

        rng = np.random.default_rng(self.random_seed)
        train_interactions: List[Tuple[int, int]] = []
        test_interactions: List[Tuple[int, int]] = []
        for user_idx in range(self.n_users):
            item_indices = self.R[user_idx].indices.copy()
            if len(item_indices) < 2:
                train_interactions.extend(
                    (user_idx, item_idx) for item_idx in item_indices
                )
                continue

            rng.shuffle(item_indices)
            n_test = min(
                max(1, int(round(len(item_indices) * self.test_ratio))),
                len(item_indices) - 1,
            )
            test_interactions.extend(
                (user_idx, item_idx) for item_idx in item_indices[:n_test]
            )
            train_interactions.extend(
                (user_idx, item_idx) for item_idx in item_indices[n_test:]
            )

        # Create train matrix
        self.train_R = sparse.lil_matrix((self.n_users, self.n_jobs), dtype=np.float32)
        for u, i in train_interactions:
            self.train_R[u, i] = self.R[u, i]
        self.train_R = self.train_R.tocsr()

        # Create test matrix (only positive interactions)
        self.test_R = sparse.lil_matrix((self.n_users, self.n_jobs), dtype=np.float32)
        for u, i in test_interactions:
            self.test_R[u, i] = self.R[u, i]
        self.test_R = self.test_R.tocsr()

        # Test user indices
        self.test_users = list(set([u for u, _ in test_interactions]))

    def get_sparse_graph(self) -> sparse.csr_matrix:
        """Get sparse adjacency matrix for LightGCN."""
        # Create bipartite adjacency matrix
        # A = [0, R; R^T, 0]
        n_total = self.n_users + self.n_jobs

        # Top-right block: R
        A = sparse.lil_matrix((n_total, n_total), dtype=np.float32)
        A[: self.n_users, self.n_users :] = self.train_R

        # Bottom-left block: R^T
        A[self.n_users :, : self.n_users] = self.train_R.T

        # Convert to CSR
        A = A.tocsr()

        # Add self-loops
        A = A + sparse.eye(n_total, dtype=np.float32)

        # Normalize adjacency matrix (D^(-1/2) A D^(-1/2))
        rowsum = np.array(A.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        D_inv_sqrt = sparse.diags(d_inv_sqrt)

        normalized_A = D_inv_sqrt @ A @ D_inv_sqrt

        return normalized_A

    def get_user_job_mappings(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Get user and job ID to index mappings."""
        return self.user_id_to_idx, self.job_id_to_idx

    def get_train_test_data(
        self,
    ) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, List[int]]:
        """Get train and test matrices and test user indices."""
        return self.train_R, self.test_R, self.test_users


class GraphLoader:
    """Simulate Neo4j graph queries for skill-based retrieval."""

    def __init__(self, data: GraphEntities):
        self.data = data

        # Build skill mappings
        self.skill_id_to_obj: Dict[str, Skill] = {
            skill.id: skill for skill in data.skills
        }

        # Build user-skill graph
        self.user_skills: Dict[str, Dict[str, str]] = {}  # user_id -> {skill_id: level}
        for user in data.users:
            self.user_skills[user.id] = user.skills

        # Build job-skill graph
        self.job_required_skills: Dict[str, Dict[str, str]] = (
            {}
        )  # job_id -> {skill_id: min_level}
        self.job_preferred_skills: Dict[str, Dict[str, str]] = (
            {}
        )  # job_id -> {skill_id: min_level}
        for job in data.jobs:
            self.job_required_skills[job.id] = job.required_skills
            self.job_preferred_skills[job.id] = job.preferred_skills

        self.skill_relations = list(data.skill_relations)
        self._relation_lookup = {
            (edge.source_skill_id, edge.target_skill_id): edge
            for edge in self.skill_relations
        }

    def get_user_skills(self, user_id: str) -> Dict[str, str]:
        """Get skills for a user."""
        return self.user_skills.get(user_id, {})

    def get_job_skills(self, job_id: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Get required and preferred skills for a job."""
        required = self.job_required_skills.get(job_id, {})
        preferred = self.job_preferred_skills.get(job_id, {})
        return required, preferred

    def get_skill_gap(self, user_id: str, job_id: str) -> Dict[str, Tuple[str, str]]:
        """
        Calculate skill gap between user and job.
        Returns: {skill_id: (user_level, required_level)}
        """
        user_skills = self.get_user_skills(user_id)
        required_skills, preferred_skills = self.get_job_skills(job_id)

        skill_gap = {}

        # Check required skills
        for skill_id, required_level in required_skills.items():
            user_level = user_skills.get(skill_id)
            if not user_level:
                # User doesn't have this skill
                skill_gap[skill_id] = (None, required_level)
            else:
                # Compare levels (simplified comparison)
                level_order = {
                    "beginner": 1,
                    "intermediate": 2,
                    "advanced": 3,
                    "expert": 4,
                }
                if level_order.get(user_level, 0) < level_order.get(required_level, 0):
                    skill_gap[skill_id] = (user_level, required_level)

        return skill_gap

    def find_shortest_paths(
        self, user_id: str, job_id: str, max_path_length: int = 3
    ) -> List[List[str]]:
        """
        Simulate finding shortest paths in skill graph via BFS on the
        prerequisite edges defined in job_associations.

        In a real Neo4j implementation, this would be:
            MATCH path = shortestPath((uSkill)-[*1..3]-(jSkill)) RETURN path
        """
        user_skill_names = list(self.get_user_skills(user_id).keys())
        required_skills, _ = self.get_job_skills(job_id)
        job_skill_names = list(required_skills.keys())

        paths: List[List[str]] = []

        # Only typed prerequisite edges can justify a learning sequence.
        adj: Dict[str, set] = {}
        for skill in self.data.skills:
            adj[skill.id] = set()
        for edge in self.skill_relations:
            if edge.relation_type == "PREREQUISITE_OF":
                adj.setdefault(edge.source_skill_id, set()).add(edge.target_skill_id)

        # --- BFS from each user skill to find paths to job skills ---
        job_skill_set = set(job_skill_names)

        for start_skill in user_skill_names:
            if start_skill not in job_skill_set:
                # BFS to find shortest path to any job-required skill
                visited: Dict[str, List[str]] = {start_skill: [start_skill]}
                queue = [start_skill]
                found = False
                while queue and not found:
                    next_queue = []
                    for curr in queue:
                        for neighbor in adj.get(curr, set()):
                            if neighbor in visited:
                                continue
                            new_path = visited[curr] + [neighbor]
                            if len(new_path) > max_path_length:
                                continue
                            visited[neighbor] = new_path
                            if neighbor in job_skill_set:
                                paths.append(new_path)
                                found = True  # only need one shortest path per start
                            else:
                                next_queue.append(neighbor)
                    queue = next_queue
            else:
                # User already has a required skill — direct match
                paths.append([start_skill])

        # Deduplicate and limit
        seen = set()
        unique_paths = []
        for p in paths:
            key = tuple(p)
            if key not in seen:
                seen.add(key)
                unique_paths.append(p)

        return unique_paths[:5]

    def find_paths_for_skills(
        self, user_skills: Dict[str, str], job_id: str, max_path_length: int = 3
    ) -> List[Dict[str, Any]]:
        """Find provenance-aware prerequisite paths for a request-scoped resume."""
        required, _ = self.get_job_skills(job_id)
        adj: Dict[str, set] = {skill.id: set() for skill in self.data.skills}
        for edge in self.skill_relations:
            if edge.relation_type == "PREREQUISITE_OF":
                adj.setdefault(edge.source_skill_id, set()).add(edge.target_skill_id)
        targets = set(required)
        results: List[Dict[str, Any]] = []
        for start in user_skills:
            visited = {start: [start]}
            queue = [start]
            while queue:
                current = queue.pop(0)
                for neighbor in sorted(adj.get(current, set())):
                    if neighbor in visited:
                        continue
                    path = visited[current] + [neighbor]
                    if len(path) - 1 > max_path_length:
                        continue
                    visited[neighbor] = path
                    if neighbor in targets and neighbor not in user_skills:
                        results.append(
                            {"skills": path, "evidence": self.get_path_evidence(path)}
                        )
                    else:
                        queue.append(neighbor)
        unique = {tuple(item["skills"]): item for item in results}
        return list(unique.values())[:5]

    def get_path_evidence(self, path: List[str]) -> List[Dict[str, Any]]:
        """Return typed edge provenance for a learning path."""
        evidence = []
        for source, target in zip(path, path[1:]):
            edge = self._relation_lookup.get((source, target))
            if edge is None:
                continue
            evidence.append(edge.model_dump())
        return evidence

    def get_recommended_learning_path(
        self, user_id: str, job_id: str
    ) -> Dict[str, Any]:
        """Generate a recommended learning path based on skill gaps."""
        skill_gap = self.get_skill_gap(user_id, job_id)
        paths = self.find_shortest_paths(user_id, job_id)

        # Calculate skill coverage
        user_skills = self.get_user_skills(user_id)
        required_skills, _ = self.get_job_skills(job_id)

        coverage = len(set(user_skills.keys()) & set(required_skills.keys())) / max(
            len(required_skills), 1
        )

        return {
            "skill_gap": skill_gap,
            "paths": [
                {"skills": path, "evidence": self.get_path_evidence(path)}
                for path in paths
                if len(path) > 1
            ],
            "skill_coverage": coverage,
            "missing_skills": list(skill_gap.keys()),
            "gap_count": len(skill_gap),
        }
