import json
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, Query, Response
from fastapi.responses import HTMLResponse, StreamingResponse, PlainTextResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import os
import tempfile
from pathlib import Path
from sqlalchemy import func

from app.models import init_db, SessionLocal, Document
from app.rag import poser_question, poser_question_stream, recuperer_contexte
from app.ingest import ingerer_pdf
from app.evaluator import evaluer_reponse_rag
from app.cache import semantic_cache

app = FastAPI(title="RAG Chatbot Enterprise - Mistral AI (Niveau 4)")

TEMPLATE_PATH = Path(__file__).parent / "templates" / "index.html"


@app.on_event("startup")
def startup():
    init_db()


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg_icon = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y=".9em" font-size="90">🤖</text></svg>"""
    return Response(content=svg_icon, media_type="image/svg+xml")


@app.get("/", response_class=HTMLResponse)
def index():
    if TEMPLATE_PATH.exists():
        return TEMPLATE_PATH.read_text(encoding="utf-8")
    return "<h1>RAG Chatbot Enterprise API is running</h1>"


class MessageItem(BaseModel):
    role: str
    content: str


class Question(BaseModel):
    texte: str
    session_id: str = "default"
    historique: List[MessageItem] = []
    filtres_documents: List[str] = []   # Filtrage sélectif par documents


class Reponse(BaseModel):
    reponse: str
    sources: List[str]
    sources_details: List[Dict[str, Any]] = []
    question_reformulee: str = ""
    cached: bool = False
    confidence_score: int = 0
    search_type: str = "hybrid_rrf"


class EvalRequest(BaseModel):
    question: str
    session_id: str = "default"
    filtres_documents: List[str] = []


class ExportReportRequest(BaseModel):
    session_id: str = "default"
    messages: List[Dict[str, Any]] = []
    filtres_documents: List[str] = []


@app.post("/chat", response_model=Reponse)
def chat(question: Question):
    history_dicts = [{"role": m.role, "content": m.content} for m in question.historique]
    return poser_question(
        question.texte,
        session_id=question.session_id,
        chat_history=history_dicts,
        filtres_documents=question.filtres_documents
    )


@app.post("/chat/stream")
def chat_stream(question: Question):
    """Streaming temps réel SSE avec métadonnées Niveau 4 (Cache Sémantique & Indice de Confiance)."""
    history_dicts = [{"role": m.role, "content": m.content} for m in question.historique]

    def event_generator():
        try:
            for token, sources, meta in poser_question_stream(
                question.texte,
                session_id=question.session_id,
                chat_history=history_dicts,
                filtres_documents=question.filtres_documents
            ):
                data = json.dumps({
                    "token": token,
                    "sources": sources,
                    "meta": meta
                })
                yield f"data: {data}\n\n"
        except Exception as e:
            err_msg = str(e)
            print(f"[chat/stream error]: {err_msg}")
            if "429" in err_msg or "capacity" in err_msg.lower():
                err_text = "\n\n⚠️ *Les serveurs de Mistral AI rencontrent une surcharge temporaire (429: backend_out_of_capacity). Veuillez patienter quelques instants et renvoyer votre question.*"
            else:
                err_text = f"\n\n⚠️ *Erreur : {err_msg}*"
            data = json.dumps({
                "token": err_text,
                "sources": [],
                "meta": {"cached": False, "confidence_score": 0}
            })
            yield f"data: {data}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/export/report")
def export_report(req: ExportReportRequest):
    """Niveau 4 : Génération d'un rapport d'audit et de synthèse documentaire en Markdown."""
    now_str = datetime.now().strftime("%d/%m/%Y à %H:%M:%S")

    # Récupérer la liste des documents de la session
    with SessionLocal() as session:
        docs = (
            session.query(Document.source, func.count(Document.id), func.max(Document.page))
            .filter(Document.session_id == req.session_id)
            .group_by(Document.source)
            .all()
        )

    lignes = [
        "# 📄 Rapport de Synthèse Documentaire & Audit RAG",
        f"**Date d'exportation :** {now_str}",
        f"**Identifiant de Session :** `{req.session_id}`",
        f"**Moteur :** Mistral AI (`open-mistral-nemo`) + Recherche Hybride (Dense pgvector + Sparse RRF)",
        "",
        "---",
        "## 📚 Documents Consultés",
    ]

    if docs:
        for src, chunks, max_page in docs:
            lignes.append(f"- **{src}** ({max_page or 1} pages, {chunks} extraits indexés)")
    else:
        lignes.append("*Aucun document enregistré dans cette session.*")

    lignes.append("")
    lignes.append("---")
    lignes.append("## 💬 Historique de l'Analyse & Réponses Sourcées")
    lignes.append("")

    if not req.messages:
        lignes.append("*Aucun échange dans cette session.*")
    else:
        for idx, msg in enumerate(req.messages, 1):
            role = "👤 Question Utilisateur" if msg.get("role") == "user" else "🤖 Réponse Assistant"
            content = msg.get("content", "")
            lignes.append(f"### {idx}. {role}")
            lignes.append(content)
            
            # Affichage des sources et confiance si disponibles
            sources = msg.get("sources", [])
            confiance = msg.get("confidence_score")
            cached = msg.get("cached", False)
            
            if sources:
                lignes.append("")
                lignes.append("**Citations & Sources :**")
                for s in sources:
                    lignes.append(f"  - 📌 {s}")
            
            badges = []
            if confiance:
                badges.append(f"Indice de confiance : {confiance}%")
            if cached:
                badges.append("⚡ Réponse issue du Cache Sémantique")
            if badges:
                lignes.append("")
                lignes.append(f"*{' | '.join(badges)}*")

            lignes.append("")

    lignes.append("---")
    lignes.append("*Rapport généré automatiquement par l'agent RAG Enterprise.*")

    rapport_markdown = "\n".join(lignes)
    return PlainTextResponse(
        content=rapport_markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f"attachment; filename=rapport_rag_{req.session_id[:8]}.md"}
    )


@app.post("/evaluate")
def evaluate_rag(req: EvalRequest):
    """Évalue la qualité RAG (Fidélité, Pertinence, Précision) avec ou sans filtres."""
    contexte, sources, _, _, _ = recuperer_contexte(
        req.question, session_id=req.session_id, filtres_documents=req.filtres_documents
    )
    if not contexte:
        return {"error": "Aucun document trouvé pour les critères sélectionnés."}
    res = poser_question(req.question, session_id=req.session_id, filtres_documents=req.filtres_documents)
    metrics = evaluer_reponse_rag(req.question, res["reponse"], contexte)
    return {
        "question": req.question,
        "reponse": res["reponse"],
        "sources": sources,
        "metrics": metrics
    }


@app.post("/upload")
def upload_multiple_pdfs(
    files: List[UploadFile] = File(...),
    session_id: str = Form("default")
):
    """Niveau 3 & 4 : Upload multi-fichiers et invalidation du cache de session."""
    resultats = []
    temp_dir = tempfile.gettempdir()

    for file in files:
        chemin = os.path.join(temp_dir, file.filename)
        with open(chemin, "wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            info = ingerer_pdf(chemin, session_id=session_id, source_name=file.filename)
            resultats.append(info)
        except Exception as e:
            print(f"[Upload] Erreur sur {file.filename}: {e}")
            return JSONResponse(
                status_code=500,
                content={"message": f"Erreur lors du traitement de '{file.filename}': {str(e)}"}
            )
        finally:
            if os.path.exists(chemin):
                os.remove(chemin)

    # Invalidation du cache pour cette session car de nouveaux documents sont arrivés
    semantic_cache.clear(session_id)

    total_chunks = sum(r["chunks"] for r in resultats)
    noms = ", ".join([r["source"] for r in resultats])
    return {
        "message": f"{len(resultats)} document(s) ingéré(s) : {noms} ({total_chunks} chunks)",
        "documents": resultats
    }


@app.get("/documents")
def list_documents(session_id: str = Query("default")):
    """Liste tous les documents de la session."""
    with SessionLocal() as session:
        docs = (
            session.query(
                Document.source,
                func.count(Document.id),
                func.max(Document.page),
                func.max(Document.document_id),
                func.max(Document.date_upload)
            )
            .filter(Document.session_id == session_id)
            .group_by(Document.source)
            .all()
        )
    return {
        "session_id": session_id,
        "total_sources": len(docs),
        "documents": [
            {
                "source": src,
                "chunks": cnt,
                "pages": max_page or 1,
                "document_id": doc_id or "n/a",
                "date_upload": date_up.strftime("%d/%m/%Y %H:%M") if date_up else "Récent"
            }
            for src, cnt, max_page, doc_id, date_up in docs
        ]
    }


@app.delete("/documents/{source_name}")
def delete_document(source_name: str, session_id: str = Query("default")):
    with SessionLocal() as session:
        deleted = (
            session.query(Document)
            .filter(Document.session_id == session_id, Document.source == source_name)
            .delete()
        )
        session.commit()
    # Invalidation du cache
    semantic_cache.clear(session_id)
    return {"message": f"{deleted} chunks supprimés pour '{source_name}'"}


@app.delete("/clear")
def clear_all_documents(session_id: str = Query("default")):
    with SessionLocal() as session:
        deleted = (
            session.query(Document)
            .filter(Document.session_id == session_id)
            .delete()
        )
        session.commit()
    # Invalidation du cache
    semantic_cache.clear(session_id)
    return {"message": f"Session réinitialisée : {deleted} chunks supprimés"}
