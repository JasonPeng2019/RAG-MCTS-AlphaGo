"""
ANN.py - Approximate Nearest Neighbor search for Go position embeddings

Uses FAISS for efficient similarity search of position embeddings based on
policy, ownership, and other features from game_analyzer.py's GoStateEmbedding.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import faiss


class PositionEmbeddingStore:
    """
    Stores and searches position embeddings using approximate nearest neighbor.

    The embedding vector is constructed from:
    - policy (362 values for 19x19 board + pass)
    - ownership (361 values for 19x19 board)
    - winrate (1 value)
    - score_lead (1 value)

    Total dimension: 725
    """

    def __init__(self, dimension: Optional[int] = None, index_type: str = 'L2'):
        """
        Initialize the ANN index.

        Args:
            dimension: Embedding dimension (default: 725 for policy+ownership+winrate+score)
            index_type: 'L2' for L2 distance or 'IP' for inner product (cosine similarity)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.embeddings_data = []  # Store full embedding dicts for retrieval
        self.is_trained = False

    def _create_embedding_vector(self, embedding_dict: dict) -> np.ndarray:
        """
        Convert GoStateEmbedding dict to a fixed-size vector for ANN search.

        Args:
            embedding_dict: Dict from GoStateEmbedding.to_dict()

        Returns:
            Numpy array of shape (dimension,)
        """
        components = []

        # Policy vector (362 values: 19x19 + pass)
        policy = embedding_dict.get('policy', [])
        if policy is None:
            policy = []
        policy_vec = np.array(policy, dtype=np.float32)
        if len(policy_vec) < 362:
            policy_vec = np.pad(policy_vec, (0, 362 - len(policy_vec)))
        elif len(policy_vec) > 362:
            policy_vec = policy_vec[:362]
        components.append(policy_vec)

        # Ownership vector (361 values: 19x19)
        ownership = embedding_dict.get('ownership', [])
        if ownership is None:
            ownership = []
        ownership_vec = np.array(ownership, dtype=np.float32)
        if len(ownership_vec) < 361:
            ownership_vec = np.pad(ownership_vec, (0, 361 - len(ownership_vec)))
        elif len(ownership_vec) > 361:
            ownership_vec = ownership_vec[:361]
        components.append(ownership_vec)

        # Scalar features
        winrate = embedding_dict.get('winrate', 0.5)
        score_lead = embedding_dict.get('score_lead', 0.0)
        components.append(np.array([winrate, score_lead], dtype=np.float32))

        # Concatenate all components
        embedding = np.concatenate(components)

        # Set dimension if not already set
        if self.dimension is None:
            self.dimension = len(embedding)

        return embedding

    def add_embeddings(self, embeddings: List[dict]):
        """
        Add multiple position embeddings to the index.

        Args:
            embeddings: List of dicts from GoStateEmbedding.to_dict()
        """
        if not embeddings:
            return

        # Convert to vectors
        vectors = np.array([self._create_embedding_vector(emb) for emb in embeddings],
                          dtype=np.float32)

        # Create index if it doesn't exist
        if self.index is None:
            if self.index_type == 'L2':
                self.index = faiss.IndexFlatL2(self.dimension)
            elif self.index_type == 'IP':
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(f"Unknown index_type: {self.index_type}")

        # Add vectors to index
        self.index.add(vectors)

        # Store original embedding data
        self.embeddings_data.extend(embeddings)
        self.is_trained = True

    def search(self, query_embedding: dict, k: int = 1) -> List[Tuple[dict, float]]:
        """
        Find k nearest neighbors to the query embedding.

        Args:
            query_embedding: Dict from GoStateEmbedding.to_dict()
            k: Number of nearest neighbors to return

        Returns:
            List of (embedding_dict, distance) tuples, sorted by distance (closest first)
        """
        if not self.is_trained or self.index is None:
            raise RuntimeError("Index not initialized. Call add_embeddings() first.")

        # Convert query to vector
        query_vec = self._create_embedding_vector(query_embedding)
        query_vec = query_vec.reshape(1, -1)

        # Search
        distances, indices = self.index.search(query_vec, k)

        # Return results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.embeddings_data):
                results.append((self.embeddings_data[idx], float(dist)))

        return results

    def search_by_sym_hash(self, sym_hash: str) -> Optional[dict]:
        """
        Exact lookup by symmetry hash.

        Args:
            sym_hash: The sym_hash to search for

        Returns:
            The embedding dict if found, None otherwise
        """
        for emb in self.embeddings_data:
            if emb.get('sym_hash') == sym_hash:
                return emb
        return None

    def size(self) -> int:
        """Return the number of embeddings in the store."""
        return len(self.embeddings_data)

    def save_index(self, filepath: str):
        """
        Save the FAISS index to disk.

        Args:
            filepath: Path to save the index
        """
        if self.index is not None:
            faiss.write_index(self.index, filepath)

    def load_index(self, filepath: str):
        """
        Load a FAISS index from disk.

        Args:
            filepath: Path to load the index from
        """
        self.index = faiss.read_index(filepath)
        self.is_trained = True
        if self.dimension is None:
            self.dimension = self.index.d


# Example usage
if __name__ == "__main__":
    # Create sample embeddings (matching GoStateEmbedding.to_dict() format)
    sample_embeddings = [
        {
            'sym_hash': 'hash1',
            'state_hash': 'state1',
            'policy': np.random.rand(362).tolist(),
            'ownership': np.random.rand(361).tolist(),
            'winrate': 0.55,
            'score_lead': 2.3,
            'move_infos': [],
            'komi': 7.5,
            'query_id': 'query_1',
        },
        {
            'sym_hash': 'hash2',
            'state_hash': 'state2',
            'policy': np.random.rand(362).tolist(),
            'ownership': np.random.rand(361).tolist(),
            'winrate': 0.48,
            'score_lead': -1.2,
            'move_infos': [],
            'komi': 7.5,
            'query_id': 'query_2',
        },
        {
            'sym_hash': 'hash3',
            'state_hash': 'state3',
            'policy': np.random.rand(362).tolist(),
            'ownership': np.random.rand(361).tolist(),
            'winrate': 0.52,
            'score_lead': 0.5,
            'move_infos': [],
            'komi': 7.5,
            'query_id': 'query_3',
        },
    ]

    # Initialize store and add embeddings
    store = PositionEmbeddingStore(index_type='L2')
    store.add_embeddings(sample_embeddings)

    print(f"Added {store.size()} embeddings to the store")

    # Create a query embedding (similar to first one)
    query = {
        'sym_hash': 'query_hash',
        'state_hash': 'query_state',
        'policy': sample_embeddings[0]['policy'],  # Same as first for testing
        'ownership': sample_embeddings[0]['ownership'],
        'winrate': 0.54,  # Slightly different
        'score_lead': 2.4,
        'move_infos': [],
        'komi': 7.5,
        'query_id': 'test_query',
    }

    # Search for nearest neighbor
    results = store.search(query, k=3)

    print(f"\nQuery embedding: {query['query_id']}")
    print(f"Winrate: {query['winrate']}, Score: {query['score_lead']}")
    print("\nNearest neighbors:")
    for i, (embedding, distance) in enumerate(results, 1):
        print(f"{i}. {embedding['query_id']} (distance: {distance:.4f})")
        print(f"   Winrate: {embedding['winrate']}, Score: {embedding['score_lead']}")
        print(f"   Sym hash: {embedding['sym_hash']}")
