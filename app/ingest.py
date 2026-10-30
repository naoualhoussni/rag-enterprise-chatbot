import os
import re
import time
import uuid
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings

from app.models import SessionLocal, Document
from app.config import MISTRAL_API_KEY


def nettoyer_texte_pdf(texte: str) -> str:
    """Nettoie le texte extrait d'un PDF pour un chunking sémantique de qualité."""
    if not texte:
        return ""
    texte = re.sub(r'(?<!\n)\n(?!\n)', ' ', texte)
    texte = re.sub(r'[ \t]+', ' ', texte)
    texte = re.sub(r'\n{3,}', '\n\n', texte)
    return texte.strip()


def ingerer_pdf(
    chemin_pdf: str,
    session_id: str = "default",
    source_name: str = None,
    document_id: str = None
) -> dict:
    """Ingère un PDF avec métadonnées complètes (document_id, date, page, session)."""
    if source_name is None:
        source_name = os.path.basename(chemin_pdf)
    if document_id is None:
        document_id = "doc_" + str(uuid.uuid4())[:8]

    # 1. Charger les pages du PDF
    loader = PyPDFLoader(chemin_pdf)
    pages = loader.load()

    for p in pages:
        p.page_content = nettoyer_texte_pdf(p.page_content)

    # 2. Chunking sémantique
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        chunk_size=900,
        chunk_overlap=120,
        keep_separator=True
    )
    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if len(c.page_content.strip()) > 30]

    if not chunks:
        print(f"Aucun texte exploitable extrait depuis {chemin_pdf}")
        return {"document_id": document_id, "source": source_name, "chunks": 0, "pages": len(pages)}

    # 3. Embeddings Mistral
    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        mistral_api_key=MISTRAL_API_KEY,
        timeout=120,
        max_retries=5
    )

    # 4. Traiter par petits lots avec retry résilient (anti-429 capacity overload)
    batch_size = 8
    all_vectors = []
    texts = [chunk.page_content for chunk in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        success = False
        last_err = None
        for attempt in range(6):
            try:
                vectors = embeddings.embed_documents(batch)
                all_vectors.extend(vectors)
                success = True
                break
            except Exception as e:
                last_err = e
                print(f"[Upload] Tentative {attempt + 1}/6 (lot {i // batch_size + 1}) - attente de {3 * (attempt + 1)}s : {e}")
                time.sleep(3 * (attempt + 1))
        
        if not success:
            raise RuntimeError(f"Échec de calcul des embeddings pour le lot {i // batch_size} : {last_err}")
        time.sleep(0.5)

    # 5. Stocker en base avec métadonnées Niveau 3 & 4
    date_now = datetime.utcnow()
    with SessionLocal() as session:
        for chunk, vecteur in zip(chunks, all_vectors):
            num_page = int(chunk.metadata.get("page", 0)) + 1
            doc = Document(
                session_id=session_id,
                document_id=document_id,
                contenu=chunk.page_content,
                source=source_name,
                page=num_page,
                date_upload=date_now,
                embedding=vecteur
            )
            session.add(doc)
        session.commit()

    return {
        "document_id": document_id,
        "source": source_name,
        "chunks": len(chunks),
        "pages": len(pages)
    }
