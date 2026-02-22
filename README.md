# NLP & Semantic Search Examples

This repository contains a short tutorial and runnable example for Natural Language Processing (NLP) tasks and a simple semantic search/document retrieval demonstration using TF-IDF and embedding-based search (Sentence-Transformers).

Files:
- `tutorial.md` — concise NLP introduction, preprocessing, representations, and models.
- `requirements.txt` — Python dependencies.
- `semantic_search.py` — runnable script demonstrating TF-IDF and embedding search over sample documents.

Quick start

1. Create a virtual environment and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the semantic search demo:

```powershell
python semantic_search.py --query "natural language processing" --method tfidf --top_k 3
python semantic_search.py --query "disaster" --method embed --top_k 3
```

If you don't have a `docs/` folder, the script will use built-in sample documents.
