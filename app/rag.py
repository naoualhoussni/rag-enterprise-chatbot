import time
from typing import Generator, Tuple, Dict, Any, List, Optional
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings

from app.models import SessionLocal, Document
from app.config import MISTRAL_API_KEY
from app.reranker import reranker_documents
from app.cache import semantic_cache
from app.hybrid_search import executer_recherche_hybride

embeddings = MistralAIEmbeddings(
    model="mistral-embed",
    mistral_api_key=MISTRAL_API_KEY,
    timeout=120,
    max_retries=5
)

llm = ChatMistralAI(
    model="open-mistral-nemo",
    mistral_api_key=MISTRAL_API_KEY,
    timeout=120,
    max_retries=5
)


def embed_query_robuste(text_query: str, max_retries: int = 5) -> List[float]:
    """Calcule l'embedding avec retry résilient en cas de saturation de capacité Mistral AI (429)."""
    for attempt in range(max_retries):
        try:
            return embeddings.embed_query(text_query)
        except Exception as e:
            err_msg = str(e).lower()
            if "429" in err_msg or "capacity" in err_msg or "overload" in err_msg or "connect" in err_msg or "eof" in err_msg:
                print(f"[RAG] Mistral saturé (tentative {attempt + 1}/{max_retries}). Pause de {attempt + 2}s...")
                time.sleep(attempt + 2)
            else:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(1)
    return [0.0] * 1024


def detecter_intention_conversationnelle(question: str) -> Optional[str]:
    """Niveau 3 : Détecte les salutations ou formules de politesse sans appel vectoriel."""
    q_lower = question.lower().strip()
    salutations = ["bonjour", "salut", "bonsoir", "coucou", "hello", "hi", "merci", "merci beaucoup", "au revoir"]
    if q_lower in salutations or q_lower in [s + " !" for s in salutations] or q_lower in [s + " ?" for s in salutations]:
        return "Bonjour ! Comment puis-je vous aider avec vos documents aujourd'hui ?"
    if q_lower in ["qui es-tu", "qui es tu", "qui es-tu ?", "qui es tu ?"]:
        return "Je suis votre assistant RAG intelligent propulsé par Mistral AI. Je réponds à vos questions en me basant strictement sur les documents PDF que vous me confiez."
    return None


def reformuler_question(question: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
    """Niveau 3 : Reformule la question pour la rendre autonome à partir du contexte récent."""
    if not chat_history or len(chat_history) == 0:
        return question

    echanges = []
    for msg in chat_history[-4:]:
        role = "Utilisateur" if msg.get("role") == "user" else "Assistant"
        content = msg.get("content", "")
        echanges.append(f"{role}: {content}")
    historique_str = "\n".join(echanges)

    prompt_reformulation = f"""En tenant compte de l'historique de conversation ci-dessous, reformule la dernière question pour qu'elle soit une question autonome et compréhensible sans le reste de la conversation.
Si la question est déjà autonome ou si l'historique n'apporte rien, renvoie la question telle quelle, sans aucun texte d'introduction ni commentaire.

Historique :
{historique_str}

Dernière question : {question}

Question autonome reformulée :"""

    try:
        res = llm.invoke(prompt_reformulation)
        reformulee = res.content.strip() if hasattr(res, 'content') else ""
        if reformulee and len(reformulee) > 3:
            return reformulee
    except Exception:
        pass
    return question


def calculer_indice_confiance(sources_details: List[Dict[str, Any]]) -> int:
    """Niveau 4 : Calcule un indice de confiance global (0 à 100%) basé sur le Reranker."""
    if not sources_details:
        return 0
    scores = [d.get("score", 70) for d in sources_details]
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    # Formule pondérée : 70% meilleur score + 30% moyenne
    confiance = int((0.7 * max_score) + (0.3 * avg_score))
    return min(99, max(15, confiance))


def recuperer_contexte(
    question: str,
    session_id: str = "default",
    chat_history: Optional[List[Dict[str, str]]] = None,
    filtres_documents: Optional[List[str]] = None,
    use_reranker: bool = True
) -> Tuple[str, List[str], str, List[Dict[str, Any]], List[float]]:
    """Niveau 4 : Récupère le contexte via Recherche Hybride (Dense + Sparse RRF) et Reranker."""
    question_recherche = reformuler_question(question, chat_history)
    question_embedding = embed_query_robuste(question_recherche)

    # Recherche Hybride (Dense pgvector + Sparse plein texte avec fusion RRF)
    candidats = executer_recherche_hybride(
        question=question_recherche,
        question_embedding=question_embedding,
        session_id=session_id,
        filtres_documents=filtres_documents,
        limit=15
    )

    if not candidats:
        # Fallback classique si la table n'a pas encore de résultats texte
        with SessionLocal() as session:
            q = session.query(Document).filter(Document.session_id == session_id)
            if filtres_documents and len(filtres_documents) > 0:
                q = q.filter(Document.source.in_(filtres_documents))
            candidats = q.order_by(Document.embedding.cosine_distance(question_embedding)).limit(15).all()

    if not candidats:
        return "", [], question_recherche, [], question_embedding

    # Two-Stage Reranking
    if use_reranker and len(candidats) > 1:
        documents_finaux = reranker_documents(question_recherche, candidats, top_k=5)
    else:
        documents_finaux = candidats[:5]

    contexte = "\n\n".join([f"[Document: {d.source} | Page {getattr(d, 'page', 1) or 1}]\n{d.contenu}" for d in documents_finaux])

    sources_brutes = []
    sources_details = []
    for d in documents_finaux:
        score = getattr(d, 'rerank_score', 85)
        score_txt = f" (Pertinence: {score}%)"
        page_val = getattr(d, 'page', None)
        page_txt = f" - Page {page_val}" if page_val else ""
        libelle = f"{d.source}{page_txt}{score_txt}"
        sources_brutes.append(libelle)
        sources_details.append({
            "source": d.source,
            "page": page_val or 1,
            "score": score,
            "document_id": getattr(d, 'document_id', None)
        })

    sources = list(dict.fromkeys(sources_brutes))
    return contexte, sources, question_recherche, sources_details, question_embedding


def poser_question(
    question: str,
    session_id: str = "default",
    chat_history: Optional[List[Dict[str, str]]] = None,
    filtres_documents: Optional[List[str]] = None
) -> Dict[str, Any]:
    # 1. Routage d'intention conversationnelle
    reponse_salutation = detecter_intention_conversationnelle(question)
    if reponse_salutation:
        return {
            "reponse": reponse_salutation,
            "sources": [],
            "sources_details": [],
            "cached": False,
            "confidence_score": 100,
            "search_type": "intent"
        }

    # 2. Vérification Cache Sémantique (Niveau 4 - Exact ou similarité > 0.95)
    cached_result = semantic_cache.get(
        question=question,
        question_embedding=None,  # Vérification exacte préalable
        session_id=session_id,
        filtres_documents=filtres_documents
    )
    if cached_result:
        return cached_result

    # 3. Recherche contextuelle Hybride
    contexte, sources, question_reformulee, details, q_embedding = recuperer_contexte(
        question, session_id=session_id, chat_history=chat_history, filtres_documents=filtres_documents
    )

    # Vérification sémantique du cache avec embedding calculé
    cached_semantic = semantic_cache.get(
        question=question,
        question_embedding=q_embedding,
        session_id=session_id,
        filtres_documents=filtres_documents
    )
    if cached_semantic:
        return cached_semantic

    if not contexte:
        return {
            "reponse": "Aucun document correspondant trouvé dans votre session. Veuillez importer un document ou vérifier vos filtres de sélection.",
            "sources": [],
            "sources_details": [],
            "cached": False,
            "confidence_score": 0,
            "search_type": "hybrid_rrf"
        }

    confidence_score = calculer_indice_confiance(details)

    # 4. Corrective RAG (CRAG) & Guardrails Anti-Hallucination
    # Si le score de pertinence maximal des documents est inférieur à 30%, on refuse poliment
    max_rerank = max([d.get("score", 0) for d in details]) if details else 0
    if max_rerank < 30 and len(details) > 0:
        reponse_guardrail = (
            "D'après les documents sélectionnés, les informations disponibles ne sont pas suffisamment pertinentes "
            f"pour répondre de manière certaine à votre question ('{question}'). "
            "Afin d'éviter toute hallucination, je vous invite à vérifier ou à importer un document traitant directement de ce sujet."
        )
        return {
            "reponse": reponse_guardrail,
            "sources": sources,
            "sources_details": details,
            "question_reformulee": question_reformulee,
            "cached": False,
            "confidence_score": 20,
            "guardrail_triggered": True,
            "search_type": "hybrid_rrf"
        }

    contexte_dialogue = ""
    if chat_history and len(chat_history) > 0:
        lignes = []
        for msg in chat_history[-3:]:
            prefix = "Utilisateur" if msg.get("role") == "user" else "Assistant"
            lignes.append(f"{prefix}: {msg.get('content', '')}")
        contexte_dialogue = "Historique récent :\n" + "\n".join(lignes) + "\n\n"

    prompt = f"""Tu es un assistant RAG expert capable d'analyser et de synthétiser des informations à travers un ou plusieurs documents PDF.
Réponds à la question en te basant UNIQUEMENT sur le contexte documentaire fourni.
Si les informations proviennent de documents différents, compare-les ou synthétise-les clairement.
Si l'information n'est pas présente dans les extraits, indique-le honnêtement.

Contexte documentaire :
{contexte}

{contexte_dialogue}Question actuelle : {question}

Réponse :"""

    response = llm.invoke(prompt)

    if isinstance(response.content, str):
        texte_reponse = response.content
    elif isinstance(response.content, list):
        texte_reponse = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in response.content])
    else:
        texte_reponse = str(response.content)

    payload = {
        "reponse": texte_reponse,
        "sources": sources,
        "sources_details": details,
        "question_reformulee": question_reformulee,
        "cached": False,
        "confidence_score": confidence_score,
        "guardrail_triggered": False,
        "search_type": "hybrid_rrf"
    }

    # Enregistrement dans le Cache Sémantique
    semantic_cache.set(
        question=question,
        question_embedding=q_embedding,
        payload=payload,
        session_id=session_id,
        filtres_documents=filtres_documents
    )

    return payload


def poser_question_stream(
    question: str,
    session_id: str = "default",
    chat_history: Optional[List[Dict[str, str]]] = None,
    filtres_documents: Optional[List[str]] = None
) -> Generator[Tuple[str, List[str], Dict[str, Any]], None, None]:
    """Streaming temps réel Niveau 4 avec Cache Sémantique, CRAG et Score de Confiance."""
    # 1. Routage conversationnel
    reponse_salutation = detecter_intention_conversationnelle(question)
    if reponse_salutation:
        yield reponse_salutation, [], {"cached": False, "confidence_score": 100, "search_type": "intent"}
        return

    # 2. Vérification Cache Sémantique
    cached = semantic_cache.get(
        question=question,
        question_embedding=None,
        session_id=session_id,
        filtres_documents=filtres_documents
    )
    if cached:
        meta = {
            "cached": True,
            "cache_type": cached.get("cache_type", "exact"),
            "confidence_score": cached.get("confidence_score", 95),
            "sources_details": cached.get("sources_details", [])
        }
        yield cached["reponse"], cached["sources"], meta
        return

    # 3. Recherche Hybride + Contexte
    contexte, sources, _, details, q_embedding = recuperer_contexte(
        question, session_id=session_id, chat_history=chat_history, filtres_documents=filtres_documents
    )

    # Vérification sémantique post-embedding
    cached_semantic = semantic_cache.get(
        question=question,
        question_embedding=q_embedding,
        session_id=session_id,
        filtres_documents=filtres_documents
    )
    if cached_semantic:
        meta = {
            "cached": True,
            "cache_type": "semantic",
            "confidence_score": cached_semantic.get("confidence_score", 95),
            "sources_details": cached_semantic.get("sources_details", [])
        }
        yield cached_semantic["reponse"], cached_semantic["sources"], meta
        return

    if not contexte:
        yield "Aucun document correspondant trouvé dans votre session. Veuillez importer un document ou vérifier vos filtres de sélection.", [], {"cached": False, "confidence_score": 0}
        return

    confidence_score = calculer_indice_confiance(details)

    # Corrective RAG (CRAG) Guardrail
    max_rerank = max([d.get("score", 0) for d in details]) if details else 0
    if max_rerank < 30 and len(details) > 0:
        reponse_guardrail = (
            "D'après les documents sélectionnés, les informations disponibles ne sont pas suffisamment pertinentes "
            f"pour répondre avec certitude à votre question ('{question}'). "
            "Afin d'éviter toute hallucination, je vous invite à importer un document directement lié à cette thématique."
        )
        yield reponse_guardrail, sources, {"cached": False, "confidence_score": 20, "guardrail_triggered": True}
        return

    contexte_dialogue = ""
    if chat_history and len(chat_history) > 0:
        lignes = []
        for msg in chat_history[-3:]:
            prefix = "Utilisateur" if msg.get("role") == "user" else "Assistant"
            lignes.append(f"{prefix}: {msg.get('content', '')}")
        contexte_dialogue = "Historique récent :\n" + "\n".join(lignes) + "\n\n"

    prompt = f"""Tu es un assistant RAG expert capable d'analyser et de synthétiser des informations à travers un ou plusieurs documents PDF.
Réponds à la question en te basant UNIQUEMENT sur le contexte documentaire fourni.
Si les informations proviennent de documents différents, compare-les ou synthétise-les clairement.
Si l'information n'est pas présente dans les extraits, indique-le honnêtement.

Contexte documentaire :
{contexte}

{contexte_dialogue}Question actuelle : {question}

Réponse :"""

    full_reponse = []
    meta = {
        "cached": False,
        "confidence_score": confidence_score,
        "sources_details": details,
        "search_type": "hybrid_rrf"
    }

    for chunk in llm.stream(prompt):
        content = chunk.content
        token = ""
        if isinstance(content, str):
            token = content
        elif isinstance(content, list):
            token = "".join([c.get("text", "") if isinstance(c, dict) else str(c) for c in content])
        full_reponse.append(token)
        yield token, sources, meta

    # Sauvegarde au cache
    texte_total = "".join(full_reponse)
    semantic_cache.set(
        question=question,
        question_embedding=q_embedding,
        payload={
            "reponse": texte_total,
            "sources": sources,
            "sources_details": details,
            "cached": False,
            "confidence_score": confidence_score
        },
        session_id=session_id,
        filtres_documents=filtres_documents
    )
