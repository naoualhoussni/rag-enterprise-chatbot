# 🤖 RAG Enterprise - Assistant d'Intelligence Documentaire Multi-PDF

> Système de **Retrieval-Augmented Generation (RAG)** de niveau Entreprise / Production, propulsé par **Mistral AI**, **PostgreSQL (pgvector)** et **FastAPI / Streamlit**.

---

## 🌟 Points Forts du Projet

* ⚡ **Double Interface Utilisateur** :
  * **Application Web Réactive (FastAPI + Tailwind CSS)** avec streaming Server-Sent Events (SSE).
  * **Dashboard Data (Streamlit)** avec audit de performance et téléchargement direct.
* 🧠 **Modèles Mistral AI** :
  * Embeddings : `mistral-embed` (1024 dimensions).
  * Génération : `open-mistral-nemo` avec invites de synthèse contextuelle.
* 🔍 **Recherche Hybride (Dense + Sparse RRF)** :
  * Fusion par rang réciproque (*Reciprocal Rank Fusion*) combinant recherche vectorielle cosinus et recherche lexicale plein texte.
* 🎯 **Two-Stage Retrieval (Reranking)** :
  * Présélection des 15 meilleurs chunks puis réordonnancement par un modèle Reranker (scores 0-100%).
* 🛡️ **Corrective RAG (CRAG) & Guardrails** :
  * Évaluation préalable de la pertinence des sources.
  * Déclenchement d'un coupe-circuit anti-hallucination si le contexte est insuffisant.
  * Calcul d'un indice de confiance certifié.
* ⚡ **Cache Sémantique (< 10 ms)** :
  * Mise en cache des requêtes similaires (> 95% similarité cosinus) pour réduire les coûts d'API et diviser la latence par 100.
* 📊 **Audit Qualité Ragas** :
  * Évaluation automatisée de la **Fidélité** (*Faithfulness*), de la **Pertinence** (*Answer Relevance*) et de la **Précision du Contexte** (*Context Precision*).
* 📑 **Auditabilité & Export de Rapports** :
  * Téléchargement de rapports d'analyse au format Markdown avec citations et pages certifiées.

---

## 🏗️ Architecture Technique

```
rag-chatbot/
│── app/
│   ├── config.py           # Configuration de l'environnement
│   ├── models.py           # Modèles SQLAlchemy & pgvector
│   ├── ingest.py           # Découpage sémantique & vectorisation
│   ├── hybrid_search.py    # Recherche hybride Dense + Sparse (RRF)
│   ├── reranker.py         # Two-Stage Reranking
│   ├── cache.py            # Cache sémantique en mémoire
│   ├── evaluator.py        # Évaluateur de métriques Ragas
│   ├── rag.py              # Logique RAG, reformulation et streaming
│   ├── main.py             # API FastAPI & WebSockets/SSE
│   └── templates/
│       └── index.html      # Interface Web interactive
│── streamlit_app.py        # Interface Streamlit
│── evaluer_rag.py          # Script d'audit qualité CLI
│── docker-compose.yml      # Base de données PostgreSQL + pgvector locale
│── requirements.txt        # Dépendances Python
│── .env.example            # Modèle de variables d'environnement
└── README.md
```

---

## 🚀 Installation & Démarrage

### 1. Prérequis
* Python 3.10+
* Une base PostgreSQL avec extension `vector` (ex: Supabase, Neon, ou via Docker Compose)
* Une clé d'API Mistral AI ([console.mistral.ai](https://console.mistral.ai/))

### 2. Cloner le dépôt et installer les dépendances
```bash
git clone https://github.com/VOTRE_PSEUDO/rag-chatbot.git
cd rag-chatbot

python -m venv venv
# Sous Windows :
.\venv\Scripts\activate
# Sous Linux/Mac :
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configuration des variables d'environnement
Copiez `.env.example` en `.env` :
```bash
cp .env.example .env
```
Renseignez vos identifiants dans `.env` :
```env
MISTRAL_API_KEY=votre_cle_mistral_ici
DATABASE_URL=postgresql://postgres.xxx:mot_de_passe@hote:5432/postgres
```

### 4. Lancer l'application

#### Option A : Interface Web FastAPI
```bash
python -m uvicorn app.main:app --reload
```
Accédez à l'interface sur : **http://127.0.0.1:8000**

#### Option B : Interface Streamlit
```bash
python -m streamlit run streamlit_app.py
```
Accédez à l'interface sur : **http://localhost:8501**

#### Option C : Lancer l'évaluation de qualité Ragas (CLI)
```bash
python evaluer_rag.py
```

---

## 📄 Licence
Ce projet est sous licence MIT - libre de modification et d'utilisation.
