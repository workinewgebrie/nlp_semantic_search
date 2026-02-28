NLP & Semantic Search — Demo and Guide
===================================

This repository demonstrates a small, presentation-ready semantic document retrieval system using two search methods:

- **TF-IDF**: lightweight, explainable (sklearn TfidfVectorizer + cosine similarity)
- **Embedding (dense)**: semantic search using Sentence-Transformers embeddings + cosine similarity

**What’s included**
- **File**: [tutorial.md](tutorial.md) — short NLP concepts and preprocessing notes.
- **File**: [requirements.txt](requirements.txt) — Python dependencies for the demo.
- **File**: [semantic_search.py](semantic_search.py) — main runnable script with loaders, preprocessing, TF-IDF and embedding search.
- **Folder**: [docs](docs) — optional sample text documents (used if present).

High-level flow (stepwise)
--------------------------
1. Prepare a corpus: either place .txt files in the `docs/` folder or provide a CSV/JSONL and use the provided loader helpers.
2. Preprocess text: tokenization, stopword removal, optional lemmatization (NLTK is used if available).
3. Index the corpus:
   - TF-IDF matrix is built at startup (fast).
   - Embeddings are computed lazily when you request embedding search (can be cached).
4. Query: the user supplies a query string and the script returns top-k documents ranked by cosine similarity.

Setup (recommended)
-------------------
1. Create and activate a virtual environment (PowerShell):

```powershell
cd C:\Users\hp\Desktop\nlp_semantic_search
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

Notes:
- If you already have a suitable Python interpreter (e.g. 3.13), use it to create the venv.
- If pip warns about `--no-warn-script-location`, you can ignore it or add the scripts folder to your PATH.

Running the demo
----------------
- TF-IDF search (fast, local):

```powershell
python semantic_search.py --query "natural language processing" --method tfidf --top_k 3
```

- Embedding search (semantic):

```powershell
python semantic_search.py --query "disaster" --method embed --top_k 3
```

If `docs/` exists, the script will index its .txt files; otherwise it uses built-in sample documents.

Command-line options
--------------------
- `--query, -q`: query text (required)
- `--method, -m`: `tfidf` or `embed` (default: `tfidf`)
- `--top_k, -k`: number of results to return (default: 3)
- `--docs_folder`: folder with .txt files (default: `docs`)

Codebase overview (file-by-file, stepwise)
-----------------------------------------
- `semantic_search.py`: main script. Key components:
  - Loader helpers: `load_documents_from_folder`, `load_documents_from_csv`, `load_documents_from_jsonl` — return lists of `(id, text)` tuples.
  - `preprocess_text()` — lowercasing, tokenization (NLTK or regex fallback), stopword removal, optional lemmatization.
  - `SemanticSearch` class:
    - builds TF-IDF matrix at initialization.
    - `init_embedding_model()` loads Sentence-Transformers lazily and encodes documents.
    - `search_tfidf()` and `search_embedding()` return lists of `(id, score, raw_text)`.
  - `main()` — CLI parsing and orchestration.

- `requirements.txt`: lists libraries used (NLTK, scikit-learn, pandas, sentence-transformers, etc.).

- `tutorial.md`: conceptual notes and suggestions for improving or extending the demo.

- `docs/`: optional local text documents; each .txt becomes one document with filename as id.

Troubleshooting / common issues
-------------------------------
- If `ModuleNotFoundError: No module named 'nltk'` appears, ensure the venv is activated and packages installed via `pip install -r requirements.txt` inside the venv.
- If you get import errors from a system/global site-packages path (e.g. `C:\Users\hp\AppData\Roaming\...`), you are likely running the global Python rather than the venv. Activate the venv and verify:

```powershell
python -c "import sys; print(sys.executable)"
```

- If a corrupted global package raises `IndentationError` (rare), uninstall or ignore the global package and prefer the venv installation: run the global interpreter with `-m pip uninstall <pkg>` and then reinstall inside the venv.

Extensions & presentation notes
-------------------------------
- To scale embeddings search for many documents, consider using FAISS or another vector store instead of a dense numpy loop.
- Cache `embedding_matrix` to disk (NumPy `.npy`) after the first run to speed repeated startup for presentations.
- For a live demo: use `--method tfidf` as a quick fallback if the audience machine has no internet or GPU resources.

Example: caching embeddings
--------------------------
1. Run embedding once to create `embeddings.npy` (modify `semantic_search.py` or run an interactive snippet).
2. On subsequent runs, load the cached array to avoid re-encoding.

Next steps I can do for you
--------------------------
- add a short PowerShell script to automate venv creation and package installation
- implement embedding cache load/save in `semantic_search.py`
- prepare a one-slide summary and a short demo script for presentation

Tell me which of the extras above you'd like me to add next.
