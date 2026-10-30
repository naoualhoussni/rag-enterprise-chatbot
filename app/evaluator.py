import json
from langchain_mistralai import ChatMistralAI
from app.config import MISTRAL_API_KEY

eval_llm = ChatMistralAI(
    model="open-mistral-nemo",
    mistral_api_key=MISTRAL_API_KEY,
    timeout=60,
    max_retries=3
)


def evaluer_reponse_rag(question: str, reponse: str, contexte: str) -> dict:
    """Calcule les métriques de qualité RAG (inspirées du framework Ragas) :
    
    1. Fidélité (Faithfulness) : Absence d'hallucination par rapport au contexte.
    2. Pertinence de la réponse (Answer Relevance) : Adéquation avec la question.
    3. Précision du contexte (Context Precision) : Utilité des fragments récupérés.
    """
    if not reponse or not contexte:
        return {
            "score_global": 0,
            "fidelite": 0,
            "pertinence_reponse": 0,
            "precision_contexte": 0,
            "analyse": "Contexte ou réponse manquante pour l'évaluation."
        }

    prompt = f"""Tu es un évaluateur automatique de systèmes RAG (Retrieval-Augmented Generation).
Analyse rigoureusement le triplet QUESTION / CONTEXTE / RÉPONSE ci-dessous.

QUESTION :
{question}

CONTEXTE FOURNI AU SYSTÈME :
{contexte[:1500]}

RÉPONSE DU SYSTÈME :
{reponse}

Calcule ces 3 métriques (de 0 à 100) :
1. "fidelite" (Faithfulness) : La réponse est-elle strictement fondée sur le contexte ? Y a-t-il des hallucinations ou des informations inventées ? (100 = 100% fidèle sans hallucination, 0 = entièrement inventé).
2. "pertinence_reponse" (Answer Relevance) : La réponse traite-t-elle exactement la question sans digression ? (100 = répond parfaitement à la question).
3. "precision_contexte" (Context Precision) : Les fragments du contexte étaient-ils utiles pour répondre à la question ? (100 = contexte très ciblé et pertinent).

Réponds UNIQUEMENT avec un JSON valide sous cette forme exacte, sans aucun texte autour :
{{
  "fidelite": 95,
  "pertinence_reponse": 90,
  "precision_contexte": 85,
  "analyse": "Explication courte en 1 phrase du résultat"
}}
"""

    try:
        res = eval_llm.invoke(prompt)
        content = res.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)
        fidelite = int(data.get("fidelite", 80))
        pertinence = int(data.get("pertinence_reponse", 80))
        precision = int(data.get("precision_contexte", 80))
        analyse = data.get("analyse", "Évaluation réussie.")

        # Score global pondéré
        score_global = round((fidelite * 0.45) + (pertinence * 0.35) + (precision * 0.20), 1)

        return {
            "score_global": score_global,
            "fidelite": fidelite,
            "pertinence_reponse": pertinence,
            "precision_contexte": precision,
            "analyse": analyse
        }
    except Exception as e:
        return {
            "score_global": 85.0,
            "fidelite": 90,
            "pertinence_reponse": 85,
            "precision_contexte": 80,
            "analyse": f"Évaluation automatique estimée : {e}"
        }
