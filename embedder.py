"""
embedder.py
Handles text embedding generation.
"""

from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts):
    """
    texts: list of strings
    returns numpy array of embeddings
    """
    if isinstance(texts, str):
        texts = [texts]

    embeddings = model.encode(texts)
    return np.array(embeddings)
def normalize_embeddings(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return vectors / norms