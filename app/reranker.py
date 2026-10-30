import json
from langchain_mistralai import ChatMistralAI
from app.config import MISTRAL_API_KEY

rerank_llm = ChatMistralAI(
    model="open-mistral-nemo",
    mistral_api_key=MISTRAL_API_KEY,
    timeout=60,
    max_retries=3
)


def reranker_documents(question: str, documents: list, top_k: int = 5) -> list:
    """Réordonne les documents candidats du plus pertinent au moins pertinent.
    
    Args:
        question: La question posée par l'utilisateur.
        documents: Liste d'objets Document issus de pgvector (ex: top 15).
        top_k: Nombre final de documents à conserver (ex: 5).
        
    Returns:
        Liste des documents triés par score de pertinence décroissant,
        avec un attribut `rerank_score` (0-100) attaché à chaque document.
    """
    if not documents or len(documents) <= 1:
        for d in documents:
            d.rerank_score = 90
        return documents[:top_k]

    # Préparer les résumés/extraits des documents pour le reranker
    candidats_texte = []
    for idx, doc in enumerate(documents):
        extrait = doc.contenu[:350].replace("\n", " ").strip()
        candidats_texte.append(f"[{idx}] {extrait}")

    liste_candidats = "\n\n".join(candidats_texte)

    prompt = f"""Tu es un modèle expert en réordonnancement (Reranker) pour un moteur RAG.
Évalue la pertinence de chaque extrait de document pour répondre précisément à la QUESTION suivante.

QUESTION : {question}

EXTRAITS CANDIDATS :
{liste_candidats}

INSTRUCTION :
Pour chaque extrait [index], attribue une note de pertinence stricte de 0 à 100 :
- 90-100 : Répond directement et précisément à la question.
- 60-89 : Apporte des informations utiles ou du contexte direct.
- 20-59 : Thème similaire mais ne répond pas à la question.
- 0-19 : Non pertinent ou bruit.

Réponds UNIQUEMENT avec un objet JSON valide sous ce format exact, sans aucun texte autour :
{{"scores": [{{"index": 0, "score": 85}}, {{"index": 1, "score": 20}}]}}
"""

    try:
        res = rerank_llm.invoke(prompt)
        content = res.content.strip()
        # Nettoyer d'éventuels délimiteurs markdown ```json
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:].strip()

        data = json.loads(content)
        scores_map = {item["index"]: item["score"] for item in data.get("scores", [])}
    except Exception as e:
        print(f"Reranking fallback vers l'ordre initial : {e}")
        scores_map = {}

    # Associer les scores aux documents
    scored_docs = []
    for idx, doc in enumerate(documents):
        # Score par défaut décroissant si non noté
        score = scores_map.get(idx, max(10, 80 - idx * 5))
        doc.rerank_score = int(score)
        scored_docs.append((doc, score))

    # Trier par score décroissant
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    # Conserver uniquement les top_k meilleurs
    resultats = [doc for doc, score in scored_docs[:top_k]]
    return resultats
