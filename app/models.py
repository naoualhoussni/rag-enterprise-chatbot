import ssl
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pgvector.sqlalchemy import Vector

from app.config import DATABASE_URL

db_url = DATABASE_URL
if db_url and db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+pg8000://", 1)

connect_args = {}
if db_url and ("supabase.co" in db_url or "pooler.supabase.com" in db_url):
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    connect_args["ssl_context"] = ctx

engine = create_engine(db_url, connect_args=connect_args)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=False, index=True)
    document_id = Column(String, nullable=True, index=True)   # Niveau 3 : ID unique par fichier
    contenu = Column(Text, nullable=False)
    source = Column(String, nullable=False)
    page = Column(Integer, nullable=True)                      # Numéro de page
    date_upload = Column(DateTime, default=datetime.utcnow)    # Niveau 3 : Horodatage
    embedding = Column(Vector(1024))                           # Mistral embeddings = 1024 dims


def init_db():
    """Crée l'extension pgvector et assure la présence de toutes les colonnes Niveau 3."""
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS session_id VARCHAR DEFAULT 'default'"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS document_id VARCHAR"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS page INTEGER DEFAULT 1"))
        conn.execute(text("ALTER TABLE documents ADD COLUMN IF NOT EXISTS date_upload TIMESTAMP DEFAULT CURRENT_TIMESTAMP"))
        conn.commit()
