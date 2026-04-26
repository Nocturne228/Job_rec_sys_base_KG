"""
Data loader for constructing interaction graphs and preparing data for models.
"""
import numpy as np
from scipy import sparse
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

from .models import GraphEntities, User, JobPosting, Skill


class DataLoader:
    """Load data and construct user-job interaction graph for LightGCN."""

    def __init__(self, data: GraphEntities, min_interactions: int = 1):
        self.data = data
        self.min_interactions = min_interactions

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
        self.users = [u for u in self.data.users
                     if user_interaction_counts[u.id] >= self.min_interactions]
        self.jobs = [j for j in self.data.jobs
                    if job_interaction_counts[j.id] >= self.min_interactions]

        # Create mappings
        self.user_id_to_idx = {user.id: idx for idx, user in enumerate(self.users)}
        self.job_id_to_idx = {job.id: idx for idx, job in enumerate(self.jobs)}
        self.idx_to_user_id = {idx: user_id for user_id, idx in self.user_id_to_idx.items()}
        self.idx_to_job_id = {idx: job_id for job_id, idx in self.job_id_to_idx.items()}

        # Build interaction matrix
        self.n_users = len(self.users)
        self.n_jobs = len(self.jobs)
        self.R = sparse.lil_matrix((self.n_users, self.n_jobs), dtype=np.float32)

        for interaction in self.data.interactions:
            if (interaction.user_id in self.user_id_to_idx and
                interaction.job_id in self.job_id_to_idx):
                user_idx = self.user_id_to_idx[interaction.user_id]
                job_idx = self.job_id_to_idx[interaction.job_id]

                # Assign weights based on interaction type
                weight = {
                    "view": 0.5,
                    "click": 1.0,
                    "save": 1.5,
                    "apply": 2.0
                }.get(interaction.interaction_type, 1.0)

                self.R[user_idx, job_idx] = max(self.R[user_idx, job_idx], weight)

        self.R = self.R.tocsr()

        # Build train/test split
        self._create_train_test_split()

    def _create_train_test_split(self, test_ratio: float = 0.2) -> None:
        """Create train/test split for evaluation."""
        # Get all non-zero interactions
        rows, cols = self.R.nonzero()
        interactions = list(zip(rows, cols))

        # Shuffle
        np.random.seed(42)
        np.random.shuffle(interactions)

        # Split
        test_size = int(len(interactions) * test_ratio)
        test_interactions = interactions[:test_size]
        train_interactions = interactions[test_size:]

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
        A[:self.n_users, self.n_users:] = self.train_R

        # Bottom-left block: R^T
        A[self.n_users:, :self.n_users] = self.train_R.T

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

    def get_train_test_data(self) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, List[int]]:
        """Get train and test matrices and test user indices."""
        return self.train_R, self.test_R, self.test_users


class GraphLoader:
    """Simulate Neo4j graph queries for skill-based retrieval."""

    def __init__(self, data: GraphEntities):
        self.data = data

        # Build skill mappings
        self.skill_id_to_obj: Dict[str, Skill] = {skill.id: skill for skill in data.skills}

        # Build user-skill graph
        self.user_skills: Dict[str, Dict[str, str]] = {}  # user_id -> {skill_id: level}
        for user in data.users:
            self.user_skills[user.id] = user.skills

        # Build job-skill graph
        self.job_required_skills: Dict[str, Dict[str, str]] = {}  # job_id -> {skill_id: min_level}
        self.job_preferred_skills: Dict[str, Dict[str, str]] = {}  # job_id -> {skill_id: min_level}
        for job in data.jobs:
            self.job_required_skills[job.id] = job.required_skills
            self.job_preferred_skills[job.id] = job.preferred_skills

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
                level_order = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}
                if level_order.get(user_level, 0) < level_order.get(required_level, 0):
                    skill_gap[skill_id] = (user_level, required_level)

        return skill_gap

    def find_shortest_paths(self, user_id: str, job_id: str, max_path_length: int = 3) -> List[List[str]]:
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

        # --- Build prerequisite adjacency from data ---
        # Map skill name -> set of neighbor names (bidirectional for BFS)
        adj: Dict[str, set] = {}
        for skill in self.data.skills:
            adj[skill.name] = set()
        for interaction in self.data.interactions:
            # Interactions don't carry prerequisite info; use all skills as nodes
            pass
        # Since mock data has no explicit prerequisite edges stored on GraphEntities,
        # we use a simplified co-occurrence proxy: two skills are connected if they
        # are both required by the same job
        for job in self.data.jobs:
            job_skills = list(job.required_skills.keys())
            for i, s1 in enumerate(job_skills):
                for s2 in job_skills[i + 1:]:
                    adj.setdefault(s1, set()).add(s2)
                    adj.setdefault(s2, set()).add(s1)

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

    def get_recommended_learning_path(self, user_id: str, job_id: str) -> Dict[str, Any]:
        """Generate a recommended learning path based on skill gaps."""
        skill_gap = self.get_skill_gap(user_id, job_id)
        paths = self.find_shortest_paths(user_id, job_id)

        # Calculate skill coverage
        user_skills = self.get_user_skills(user_id)
        required_skills, _ = self.get_job_skills(job_id)

        coverage = len(set(user_skills.keys()) & set(required_skills.keys())) / max(len(required_skills), 1)

        return {
            "skill_gap": skill_gap,
            "paths": paths,
            "skill_coverage": coverage,
            "missing_skills": list(skill_gap.keys()),
            "gap_count": len(skill_gap)
        }