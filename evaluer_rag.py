"""Script d'évaluation de la qualité du RAG (Métrique de type Ragas).
Exécutez : python evaluer_rag.py [session_id] [question_test]
"""
import sys
import os

# Assure le bon chargement des modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"

from app.rag import poser_question, recuperer_contexte
from app.evaluator import evaluer_reponse_rag


def lancer_evaluation(session_id: str = "default", question: str = "De quoi parle ce document ?"):
    print("=" * 65)
    print("   RAPPORT D'ÉVALUATION DE QUALITÉ RAG (MÉTRIQUES RAGAS)")
    print("=" * 65)
    print(f"Session testée : {session_id}")
    print(f"Question       : {question}\n")

    print("1. Recherche contextuelle avec Two-Stage Retrieval (pgvector + Reranker)...")
    contexte, sources, _, details = recuperer_contexte(question, session_id=session_id)
    
    if not contexte:
        print("❌ Aucun document trouvé pour cette session. Uploadez d'abord un PDF !")
        return

    print(f"   ✓ {len(details)} fragments pertinents sélectionnés par le Reranker")
    for d in details:
        print(f"     - {d['source']} (Page {d['page']}) -> Score Reranker: {d['score']}%")

    print("\n2. Génération de la réponse via Mistral...")
    res = poser_question(question, session_id=session_id)
    reponse = res["reponse"]
    print(f"   Réponse : {reponse[:150]}...\n")

    print("3. Calcul des métriques de qualité (Fidélité, Pertinence, Précision)...")
    metrics = evaluer_reponse_rag(question, reponse, contexte)

    print("\n" + "-" * 65)
    print(f"  Fidélité / Non-hallucination (Faithfulness)   : {metrics['fidelite']}%")
    print(f"  Pertinence de la réponse (Answer Relevance)   : {metrics['pertinence_reponse']}%")
    print(f"  Précision du contexte (Context Precision)     : {metrics['precision_contexte']}%")
    print("-" * 65)
    print(f"  ⭐ SCORE GLOBAL RAG (Harmonic / Weighted)      : {metrics['score_global']}%")
    print(f"  Verdict : {metrics['analyse']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    sid = sys.argv[1] if len(sys.argv) > 1 else "default"
    q = sys.argv[2] if len(sys.argv) > 2 else "De quoi parle ce document ?"
    lancer_evaluation(sid, q)
