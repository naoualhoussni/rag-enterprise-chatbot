import time
import math
from typing import Optional, Dict, Any, List


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticCache:
    """Cache sémantique en mémoire (Niveau 4).
    
    Permet de renvoyer instantanément (<10ms) une réponse si une question
    identique ou sémantiquement équivalente (similarité cosinus >= seuil) a déjà été posée.
    """

    def __init__(self, threshold: float = 0.95, max_entries_per_session: int = 50):
        self.threshold = threshold
        self.max_entries = max_entries_per_session
        # Structure: session_id -> list of cache entries
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def _normalize_filter_key(self, filtres: Optional[List[str]]) -> str:
        if not filtres:
            return "all"
        return "|".join(sorted(filtres))

    def get(
        self,
        question: str,
        question_embedding: Optional[List[float]],
        session_id: str = "default",
        filtres_documents: Optional[List[str]] = None
    ) -> Optional[Dict[str, Any]]:
        entries = self._store.get(session_id, [])
        if not entries:
            return None

        q_clean = question.strip().lower()
        filter_key = self._normalize_filter_key(filtres_documents)

        # 1. Vérification exacte (Exact Match - ultra-rapide < 1ms)
        for entry in entries:
            if entry["filter_key"] == filter_key and entry["question_clean"] == q_clean:
                res = dict(entry["payload"])
                res["cached"] = True
                res["cache_type"] = "exact"
                res["similarity"] = 1.0
                return res

        # 2. Vérification sémantique (Cosine Similarity sur embedding)
        if question_embedding:
            best_match = None
            highest_sim = 0.0

            for entry in entries:
                if entry["filter_key"] != filter_key:
                    continue
                sim = cosine_similarity(question_embedding, entry["embedding"])
                if sim > highest_sim:
                    highest_sim = sim
                    best_match = entry

            if highest_sim >= self.threshold and best_match:
                res = dict(best_match["payload"])
                res["cached"] = True
                res["cache_type"] = "semantic"
                res["similarity"] = round(highest_sim, 3)
                return res

        return None

    def set(
        self,
        question: str,
        question_embedding: Optional[List[float]],
        payload: Dict[str, Any],
        session_id: str = "default",
        filtres_documents: Optional[List[str]] = None
    ):
        if session_id not in self._store:
            self._store[session_id] = []

        filter_key = self._normalize_filter_key(filtres_documents)

        entry = {
            "question_clean": question.strip().lower(),
            "original_question": question,
            "filter_key": filter_key,
            "embedding": question_embedding or [],
            "payload": payload,
            "timestamp": time.time()
        }

        # Évite l'accumulation infinie
        if len(self._store[session_id]) >= self.max_entries:
            self._store[session_id].pop(0)

        self._store[session_id].append(entry)

    def clear(self, session_id: Optional[str] = None):
        if session_id:
            self._store.pop(session_id, None)
        else:
            self._store.clear()


# Instance globale partagée
semantic_cache = SemanticCache(threshold=0.95)
