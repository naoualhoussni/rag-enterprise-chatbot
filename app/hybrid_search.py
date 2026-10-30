from typing import List, Dict, Any, Optional
from sqlalchemy import text
from app.models import SessionLocal, Document


def executer_recherche_hybride(
    question: str,
    question_embedding: List[float],
    session_id: str = "default",
    filtres_documents: Optional[List[str]] = None,
    limit: int = 15,
    rrf_k: int = 60
) -> List[Any]:
    """Exécute une recherche hybride (Dense pgvector + Sparse plein texte) avec fusion RRF (Reciprocal Rank Fusion).
    
    Formule RRF : score = sum(1 / (k + rank))
    Permet de capturer à la fois la proximité sémantique et les termes exacts (codes, chiffres, sigles).
    """
    with SessionLocal() as db_session:
        # 1. Branche Dense (Vectorielle pgvector via ORM natif et sûr)
        try:
            q_dense = db_session.query(
                Document.id,
                Document.source,
                Document.page,
                Document.contenu,
                Document.document_id
            ).filter(Document.session_id == session_id)

            if filtres_documents and len(filtres_documents) > 0:
                q_dense = q_dense.filter(Document.source.in_(filtres_documents))

            dense_rows = (
                q_dense
                .order_by(Document.embedding.cosine_distance(question_embedding))
                .limit(limit)
                .all()
            )
        except Exception as e:
            print(f"[HybridSearch] Erreur branche dense: {e}")
            dense_rows = []

        # 2. Branche Sparse (Plein texte / Mots-clés)
        mots = [m.replace("'", "''") for m in question.split() if len(m) > 2]
        sparse_rows = []
        if mots:
            conditions = ["session_id = :session_id"]
            params: Dict[str, Any] = {"session_id": session_id}
            if filtres_documents and len(filtres_documents) > 0:
                conditions.append("source IN :sources")
                params["sources"] = tuple(filtres_documents)
            where_clause = " AND ".join(conditions)

            like_clauses = " OR ".join([f"contenu ILIKE :mot_{i}" for i in range(min(4, len(mots)))])
            like_query = text(f"""
                SELECT id, source, page, contenu, document_id, 1 as rank
                FROM documents
                WHERE {where_clause} AND ({like_clauses})
                LIMIT :sparse_limit
            """)
            like_params = dict(params)
            for i in range(min(4, len(mots))):
                like_params[f"mot_{i}"] = f"%{mots[i]}%"
            like_params["sparse_limit"] = limit

            try:
                sparse_rows = db_session.execute(like_query, like_params).fetchall()
            except Exception as e:
                print(f"[HybridSearch] Erreur branche sparse: {e}")
                sparse_rows = []

        # 3. Fusion Reciprocal Rank Fusion (RRF)
        scores_rrf: Dict[int, float] = {}
        docs_map: Dict[int, Any] = {}

        for rank, row in enumerate(dense_rows):
            doc_id = row[0]
            scores_rrf[doc_id] = scores_rrf.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            if doc_id not in docs_map:
                docs_map[doc_id] = row

        for rank, row in enumerate(sparse_rows):
            doc_id = row[0]
            scores_rrf[doc_id] = scores_rrf.get(doc_id, 0.0) + (1.0 / (rrf_k + rank + 1))
            if doc_id not in docs_map:
                docs_map[doc_id] = row

        # Trier par score RRF décroissant
        sorted_ids = sorted(scores_rrf.keys(), key=lambda x: scores_rrf[x], reverse=True)[:limit]

        candidats_fusionnes = []
        for did in sorted_ids:
            row = docs_map[did]
            candidats_fusionnes.append(
                DocumentProxy(
                    id=row[0],
                    source=row[1],
                    page=row[2],
                    contenu=row[3],
                    document_id=row[4],
                    rrf_score=scores_rrf[did]
                )
            )

        return candidats_fusionnes


class DocumentProxy:
    """Proxy léger simulant un objet Document pour le Reranker."""
    def __init__(self, id, source, page, contenu, document_id=None, rrf_score=0.0):
        self.id = id
        self.source = source
        self.page = page
        self.contenu = contenu
        self.document_id = document_id
        self.rrf_score = rrf_score
        self.rerank_score = 0
