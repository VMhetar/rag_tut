"""
vector_store.py
Handles storage and retrieval of embeddings.
"""

import numpy as np

class VectorStore:
    def __init__(self):
        self.embeddings = None
        self.chunks = []
    
    def add(self, embeddings, chunks):
        """
        embeddings: numpy array (N, D)
        chunks: listof chunk dictionaries
        """
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack((self.embeddings, embeddings))
        self.chunks.extend(chunks)
    
    def search(self, query_embedding, top_k=3):
        """
        query_embeddings: numpy array (D,) or (1, D)
        returns top_k most similar chunks
        """
        if(len(query_embedding.shape)==2):
            query_embedding = query_embedding[0]
        # cosine similarity (doct product because normalized)
        scores = np.dot(self.embeddings, query_embedding)

        # get top_k indices sorted descending
        top_indices = np.argsort(scores)[-top_k:][::-1]
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "chunk": self.chunks[idx]
            })

        return results
