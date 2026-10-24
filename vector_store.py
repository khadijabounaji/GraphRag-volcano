"""
vector_store.py — Stockage et indexation des vecteurs pour le projet RAG.
Gere la sauvegarde, le chargement et la recherche par similarite cosinus.
Format : JSON lisible (UTF-8), pas de pickle binaire.
"""

import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    """
    Store de vecteurs pour le retrieval.
    Stocke les chunks, leurs embeddings, et permet la recherche par similarite.
    Sauvegarde en JSON lisible (UTF-8).
    """

    def __init__(self, chunks=None, embeddings=None):
        """
        Args:
            chunks: liste de strings (les chunks de texte)
            embeddings: np.ndarray de shape (n_chunks, dim)
        """
        self.chunks = chunks if chunks is not None else []
        self.embeddings = embeddings if embeddings is not None else np.array([])

    def save(self, path="vector_store.json"):
        """Sauvegarde le store (chunks + embeddings) dans un fichier JSON lisible."""
        data = []
        for i, (chunk, emb) in enumerate(zip(self.chunks, self.embeddings)):
            data.append({
                "chunk_id":  i + 1,
                "texte":     chunk.strip(),
                "embedding": emb.tolist()
            })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  [VectorStore] Sauvegarde -> {path} ({len(self.chunks)} chunks, JSON lisible)")

    @classmethod
    def load(cls, path="vector_store.json"):
        """Charge un store depuis un fichier JSON lisible."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [item["texte"] for item in data]
        embeddings = np.array([item["embedding"] for item in data], dtype=np.float32)
        store = cls(chunks=chunks, embeddings=embeddings)
        print(f"  [VectorStore] Charge depuis {path} ({len(store.chunks)} chunks)")
        return store

    def search(self, query_vector, k=5):
        """
        Recherche les k chunks les plus proches du vecteur query.

        Args:
            query_vector: np.ndarray de shape (1, dim) ou (dim,)
            k: nombre de resultats a retourner

        Returns:
            list of tuples (index, score, chunk_text)
        """
        if len(self.embeddings) == 0:
            return []

        q = np.array(query_vector)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        sims = cosine_similarity(q, self.embeddings)[0]
        top_k_idx = np.argsort(sims)[::-1][:k]

        results = []
        for idx in top_k_idx:
            results.append((int(idx), float(sims[idx]), self.chunks[idx]))
        return results

    def __len__(self):
        return len(self.chunks)

    def __repr__(self):
        dim = self.embeddings.shape[1] if len(self.embeddings) > 0 else 0
        return f"VectorStore(chunks={len(self.chunks)}, dim={dim})"
