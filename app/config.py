import os
from dotenv import load_dotenv

# Contournements Windows & Environnement
os.environ["DISABLE_SQLALCHEMY_CEXT"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY manquante dans .env")
