import streamlit as st
import tempfile
import os
import uuid
from datetime import datetime

from app.ingest import ingerer_pdf
from app.rag import poser_question_stream, recuperer_contexte, poser_question
from app.evaluator import evaluer_reponse_rag
from app.models import SessionLocal, Document
from app.cache import semantic_cache
from sqlalchemy import func

st.set_page_config(page_title="RAG Enterprise - Niveau 4 Expert", page_icon="🏆", layout="wide")

# Gestion d'un identifiant de session unique par utilisateur
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "user_" + str(uuid.uuid4())[:8]

session_id = st.session_state["session_id"]

st.title("🏆 RAG Enterprise - Niveau 4 (Production & Hybride RRF)")
st.caption(f"Session : `{session_id}` • Recherche Hybride (Dense + Sparse RRF) • Cache Sémantique • CRAG Anti-Hallucination")

# Récupérer les documents existants pour la session
existing_doc_names = []
try:
    with SessionLocal() as db_session:
        docs_db = (
            db_session.query(Document.source, func.count(Document.id), func.max(Document.page), func.max(Document.date_upload))
            .filter(Document.session_id == session_id)
            .group_by(Document.source)
            .all()
        )
        existing_doc_names = [d[0] for d in docs_db]
except Exception:
    docs_db = []

# Barre latérale : Gestion Multi-Documents & Export
with st.sidebar:
    st.header("1. Mes documents (Multi-Upload)")
    st.write(f"Session active : `{session_id}`")
    
    if st.button("🔄 Nouvel espace vierge"):
        st.session_state["session_id"] = "user_" + str(uuid.uuid4())[:8]
        st.session_state["messages"] = []
        semantic_cache.clear(session_id)
        st.rerun()

    uploaded_files = st.file_uploader(
        "Importer 1 ou plusieurs PDF simultanément",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("Ingérer tous les PDF sélectionnés"):
            total_added = 0
            with st.spinner("Ingestion et indexation hybride en cours..."):
                temp_dir = tempfile.gettempdir()
                for uf in uploaded_files:
                    chemin_temp = os.path.join(temp_dir, uf.name)
                    with open(chemin_temp, "wb") as f:
                        f.write(uf.getbuffer())
                    try:
                        res = ingerer_pdf(chemin_temp, session_id=session_id, source_name=uf.name)
                        total_added += res.get("chunks", 0)
                    finally:
                        if os.path.exists(chemin_temp):
                            os.remove(chemin_temp)
                semantic_cache.clear(session_id)
                st.success(f"✅ {len(uploaded_files)} document(s) ingéré(s) ({total_added} fragments sémantiques) !")
                st.rerun()

    st.markdown("---")
    st.subheader("Filtrage documentaire sélectif")
    
    filtre_selection = []
    if existing_doc_names:
        filtre_selection = st.multiselect(
            "Cibler la recherche sur :",
            options=existing_doc_names,
            default=existing_doc_names,
            help="Décochez des documents pour restreindre la recherche RAG à certains fichiers"
        )
        for src, cnt, max_p, d_up in docs_db:
            date_str = d_up.strftime("%d/%m %H:%M") if d_up else ""
            st.caption(f"📄 **{src}** ({cnt} chunks • {max_p or 1} p. • {date_str})")
    else:
        st.caption("Aucun document en base pour cette session.")

    # Export Rapport Markdown
    st.markdown("---")
    st.subheader("📥 Export & Auditabilité (Niveau 4)")
    if "messages" in st.session_state and st.session_state["messages"]:
        report_lines = [
            f"# Rapport de Synthèse Documentaire RAG Enterprise",
            f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            f"Session : `{session_id}`\n",
            "## Documents analysés :",
        ]
        for src in existing_doc_names:
            report_lines.append(f"- {src}")
        report_lines.append("\n## Échanges & Réponses sourcées :\n")
        for m in st.session_state["messages"]:
            role = "Question" if m["role"] == "user" else "Réponse"
            report_lines.append(f"### {role} :\n{m['content']}")
            if m.get("sources"):
                report_lines.append(f"**Sources :** {', '.join(m['sources'])}\n")
        report_content = "\n".join(report_lines)
        
        st.download_button(
            label="📄 Télécharger le Rapport (.md)",
            data=report_content,
            file_name=f"rapport_rag_{session_id[:8]}.md",
            mime="text/markdown"
        )

    if st.button("🗑️ Supprimer mes documents"):
        with SessionLocal() as db_session:
            db_session.query(Document).filter(Document.session_id == session_id).delete()
            db_session.commit()
        semantic_cache.clear(session_id)
        st.success("Session purgée !")
        st.session_state["messages"] = []
        st.rerun()

    st.markdown("---")
    st.subheader("📊 Évaluation Qualité RAG")
    eval_question = st.text_input("Question à auditer :", value="Fais une synthèse de mes documents")
    if st.button("Lancer l'audit de qualité"):
        with st.spinner("Audit en cours..."):
            contexte, sources, _, _, _ = recuperer_contexte(eval_question, session_id=session_id, filtres_documents=filtre_selection)
            if not contexte:
                st.warning("Aucun document correspondant aux filtres !")
            else:
                res = poser_question(eval_question, session_id=session_id, filtres_documents=filtre_selection)
                metrics = evaluer_reponse_rag(eval_question, res["reponse"], contexte)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Fidélité", f"{metrics['fidelite']}%")
                col2.metric("Pertinence", f"{metrics['pertinence_reponse']}%")
                col3.metric("Précision", f"{metrics['precision_contexte']}%")
                st.info(f"**Score Global : {metrics['score_global']}%** — {metrics['analyse']}")

    st.markdown("---")
    st.success("**Niveau 4 Actif :**\n- 🔍 Recherche Hybride (Dense + Sparse RRF)\n- ⚡ Cache Sémantique (< 10ms)\n- 🛡️ Corrective RAG (Anti-Hallucination)\n- 🎯 Score de Confiance\n- 📥 Export de rapport Markdown")

# Initialisation de l'historique du chat
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Affichage des messages passés
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            sources_display = [f"📄 {s}" for s in msg["sources"]]
            st.caption(f"📎 **Sources :** {' • '.join(sources_display)}")
        if msg.get("confidence_score"):
            badge_cached = " • ⚡ *Cache sémantique*" if msg.get("cached") else ""
            st.caption(f"🎯 **Indice de confiance :** {msg['confidence_score']}%{badge_cached}")

# Saisie utilisateur
if prompt := st.chat_input("Posez votre question (sur l'ensemble ou une sélection de documents)..."):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        last_sources = []
        last_meta = {}
        historique_contexte = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state["messages"][:-1]
        ]
        
        def stream_generator():
            nonlocal last_sources, last_meta
            for token, sources, meta in poser_question_stream(
                prompt,
                session_id=session_id,
                chat_history=historique_contexte,
                filtres_documents=filtre_selection
            ):
                last_sources = sources
                last_meta = meta
                yield token

        reponse_complete = st.write_stream(stream_generator())
        
        if last_sources:
            sources_display = [f"📄 {s}" for s in last_sources]
            st.caption(f"📎 **Sources vérifiées :** {' • '.join(sources_display)}")

        if last_meta and last_meta.get("confidence_score"):
            badge_cached = " • ⚡ *Cache sémantique (<10ms)*" if last_meta.get("cached") else ""
            st.caption(f"🎯 **Indice de confiance :** {last_meta['confidence_score']}%{badge_cached}")
        
        st.session_state["messages"].append({
            "role": "assistant",
            "content": reponse_complete,
            "sources": last_sources,
            "confidence_score": last_meta.get("confidence_score") if last_meta else None,
            "cached": last_meta.get("cached", False) if last_meta else False
        })
