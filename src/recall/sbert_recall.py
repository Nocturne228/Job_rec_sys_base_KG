"""
Semantic recall using Sentence-BERT (SBERT) for cold-start scenarios.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import logging

# Try to import sentence-transformers, but provide fallback for simulation
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False
    logging.warning("sentence-transformers not available. Using simulated embeddings.")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("faiss not available. Using brute-force similarity search.")


class SBERTRecall:
    """Semantic recall using SBERT embeddings."""

    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "cpu",
                 use_faiss: bool = True):
        """
        Initialize SBERT recall model.

        Args:
            model_name: Name of SBERT model to use
            device: Device to run model on
            use_faiss: Whether to use Faiss for efficient similarity search
        """
        self.model_name = model_name
        self.device = device
        self.use_faiss = use_faiss and FAISS_AVAILABLE

        # Initialize model
        if SBERT_AVAILABLE:
            self.model = SentenceTransformer(model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
        else:
            # Simulate model with random embeddings
            self.model = None
            self.embedding_dim = 384  # Typical MiniLM dimension

        # Storage for embeddings
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.job_embeddings: Dict[str, np.ndarray] = {}

        # FAISS index
        self.faiss_index = None
        self.job_ids = []

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to embedding vector."""
        if self.model is not None:
            # Use real SBERT model
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        else:
            # Simulate embedding (deterministic from text hash)
            # In a real system, this would be actual SBERT embeddings
            seed = hash(text) % (2**32)
            np.random.seed(seed)
            embedding = np.random.randn(self.embedding_dim).astype(np.float32)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            return embedding

    def add_user(self, user_id: str, resume_text: str) -> None:
        """Add user with resume text."""
        embedding = self.encode_text(resume_text)
        self.user_embeddings[user_id] = embedding

    def add_job(self, job_id: str, job_description: str) -> None:
        """Add job with description."""
        embedding = self.encode_text(job_description)
        self.job_embeddings[job_id] = embedding

        # Update FAISS index if using it
        if self.use_faiss:
            self._update_faiss_index()

    def _update_faiss_index(self) -> None:
        """Update FAISS index with current job embeddings."""
        if not self.job_embeddings:
            return

        self.job_ids = list(self.job_embeddings.keys())
        embeddings = np.array([self.job_embeddings[jid] for jid in self.job_ids], dtype=np.float32)

        # Create or update index
        if self.faiss_index is None:
            # Create new index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
            self.faiss_index.add(embeddings)
        else:
            # Rebuild index (simplified - in production would use incremental updates)
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
            self.faiss_index.add(embeddings)

    def compute_similarity(self, user_id: str, job_id: str) -> float:
        """Compute cosine similarity between user and job embeddings."""
        if user_id not in self.user_embeddings or job_id not in self.job_embeddings:
            return 0.0

        user_vec = self.user_embeddings[user_id]
        job_vec = self.job_embeddings[job_id]

        # Cosine similarity
        similarity = np.dot(user_vec, job_vec) / (np.linalg.norm(user_vec) * np.linalg.norm(job_vec))
        return float(similarity)

    def recommend_for_user(self,
                          user_id: str,
                          k: int = 10,
                          job_ids: Optional[List[str]] = None) -> List[Tuple[str, float]]:
        """
        Recommend jobs for a user based on semantic similarity.

        Args:
            user_id: User ID
            k: Number of recommendations
            job_ids: Optional list of job IDs to consider (if None, consider all jobs)

        Returns:
            List of (job_id, similarity_score) tuples, sorted by score descending
        """
        if user_id not in self.user_embeddings:
            return []

        user_vec = self.user_embeddings[user_id].reshape(1, -1).astype(np.float32)

        # Get job IDs to consider
        if job_ids is None:
            job_ids = list(self.job_embeddings.keys())
            embeddings = np.array([self.job_embeddings[jid] for jid in job_ids], dtype=np.float32)
        else:
            embeddings = np.array([self.job_embeddings[jid] for jid in job_ids if jid in self.job_embeddings],
                                 dtype=np.float32)
            # Filter job_ids to those with embeddings
            job_ids = [jid for jid in job_ids if jid in self.job_embeddings]

        if len(job_ids) == 0:
            return []

        # Use FAISS for efficient search if available
        if self.use_faiss and self.faiss_index is not None:
            # Search using FAISS
            distances, indices = self.faiss_index.search(user_vec, min(k, len(job_ids)))

            results = []
            for idx, dist in zip(indices[0], distances[0]):
                if idx >= 0 and idx < len(job_ids):
                    results.append((job_ids[idx], float(dist)))
            return results

        else:
            # Brute-force similarity calculation
            similarities = []
            for job_id in job_ids:
                job_vec = self.job_embeddings[job_id]
                similarity = np.dot(user_vec.flatten(), job_vec) / (
                    np.linalg.norm(user_vec) * np.linalg.norm(job_vec))
                similarities.append((job_id, float(similarity)))

            # Sort by similarity descending
            similarities.sort(key=lambda x: x[1], reverse=True)

            return similarities[:k]

    def batch_recommend(self,
                       user_ids: List[str],
                       k: int = 10,
                       job_ids: Optional[List[str]] = None) -> Dict[str, List[Tuple[str, float]]]:
        """Recommend jobs for multiple users."""
        results = {}
        for user_id in user_ids:
            results[user_id] = self.recommend_for_user(user_id, k, job_ids)
        return results

    def get_embedding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored embeddings."""
        return {
            "n_users": len(self.user_embeddings),
            "n_jobs": len(self.job_embeddings),
            "embedding_dim": self.embedding_dim,
            "using_faiss": self.use_faiss and self.faiss_index is not None,
            "sbert_available": SBERT_AVAILABLE,
            "faiss_available": FAISS_AVAILABLE
        }

    def save_embeddings(self, path: str) -> None:
        """Save embeddings to disk."""
        import pickle
        data = {
            'user_embeddings': self.user_embeddings,
            'job_embeddings': self.job_embeddings,
            'job_ids': self.job_ids,
            'embedding_dim': self.embedding_dim,
            'model_name': self.model_name
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def load_embeddings(self, path: str) -> None:
        """Load embeddings from disk."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.user_embeddings = data['user_embeddings']
        self.job_embeddings = data['job_embeddings']
        self.job_ids = data['job_ids']
        self.embedding_dim = data['embedding_dim']
        self.model_name = data['model_name']

        # Rebuild FAISS index if needed
        if self.use_faiss and self.job_embeddings:
            self._update_faiss_index()