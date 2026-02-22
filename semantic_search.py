"""
Simple semantic search / document retrieval demo.

Supports two methods:
- tfidf : TF-IDF vectorizer + cosine similarity
- embed : sentence-transformers embeddings + cosine similarity

Usage examples:
    python semantic_search.py --query "natural language processing" --method tfidf --top_k 3
    python semantic_search.py --query "disaster" --method embed --top_k 3

If a `docs/` directory exists with .txt files, the script will index them; otherwise it uses built-in samples.
"""

from __future__ import annotations
import argparse
import os
import glob
import math
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


def ensure_nltk_resources() -> None:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')


def load_documents_from_folder(folder: str) -> List[Tuple[str, str]]:
    pattern = os.path.join(folder, "*.txt")
    paths = sorted(glob.glob(pattern))
    docs = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            docs.append((os.path.basename(path), f.read()))
    return docs


def default_documents() -> List[Tuple[str, str]]:
    samples = [
        ("doc1", "Natural Language Processing enables computers to understand human language."),
        ("doc2", "Wildfire in California forced thousands to evacuate their homes."),
        ("doc3", "Scientists develop new models for language understanding using transformers."),
        ("doc4", "Local football team wins championship after dramatic penalty shootout."),
        ("doc5", "Floods and earthquakes cause major damage during the storm season."),
    ]
    return samples


def preprocess_text(text: str, lemmatizer: WordNetLemmatizer, stop_words_set: set) -> str:
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words_set]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


class SemanticSearch:
    def __init__(self, documents: List[Tuple[str, str]]):
        ensure_nltk_resources()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

        self.docs_meta = []  # list of (id, raw_text, preprocessed_text)
        for ident, text in documents:
            pre = preprocess_text(text, self.lemmatizer, self.stop_words)
            self.docs_meta.append((ident, text, pre))

        # TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer()
        corpus = [pre for (_id, _raw, pre) in self.docs_meta]
        if len(corpus) > 0:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        else:
            self.tfidf_matrix = None

        # Embedding model (lazy)
        self.embed_model = None
        self.embedding_matrix = None

    def init_embedding_model(self, model_name: str = 'all-MiniLM-L6-v2') -> None:
        if SentenceTransformer is None:
            raise RuntimeError('sentence-transformers not installed; install sentence-transformers in requirements')
        if self.embed_model is None:
            self.embed_model = SentenceTransformer(model_name)
            texts = [raw for (_id, raw, _pre) in self.docs_meta]
            self.embedding_matrix = np.array(self.embed_model.encode(texts, show_progress_bar=False))

    def search_tfidf(self, query: str, top_k: int = 5) -> List[Tuple[str, float, str]]:
        if self.tfidf_matrix is None:
            return []
        q_pre = preprocess_text(query, self.lemmatizer, self.stop_words)
        q_vec = self.tfidf_vectorizer.transform([q_pre])
        similarities = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
        ranks = np.argsort(-similarities)[:top_k]
        results = []
        for idx in ranks:
            ident, raw, pre = self.docs_meta[idx]
            results.append((ident, float(similarities[idx]), raw))
        return results

    def search_embedding(self, query: str, top_k: int = 5, model_name: str = 'all-MiniLM-L6-v2') -> List[Tuple[str, float, str]]:
        if self.embed_model is None:
            self.init_embedding_model(model_name)
        q_emb = np.array(self.embed_model.encode([query], show_progress_bar=False))[0]
        # cosine similarity
        dot = np.dot(self.embedding_matrix, q_emb)
        denom = np.linalg.norm(self.embedding_matrix, axis=1) * (np.linalg.norm(q_emb) + 1e-12)
        similarities = dot / denom
        ranks = np.argsort(-similarities)[:top_k]
        results = []
        for idx in ranks:
            ident, raw, pre = self.docs_meta[idx]
            results.append((ident, float(similarities[idx]), raw))
        return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', '-q', type=str, required=True)
    parser.add_argument('--method', '-m', choices=['tfidf', 'embed'], default='tfidf')
    parser.add_argument('--top_k', '-k', type=int, default=3)
    parser.add_argument('--docs_folder', type=str, default='docs')
    args = parser.parse_args()

    if os.path.isdir(args.docs_folder):
        documents = load_documents_from_folder(args.docs_folder)
        if len(documents) == 0:
            documents = default_documents()
    else:
        documents = default_documents()

    searcher = SemanticSearch(documents)

    if args.method == 'tfidf':
        results = searcher.search_tfidf(args.query, top_k=args.top_k)
    else:
        try:
            results = searcher.search_embedding(args.query, top_k=args.top_k)
        except RuntimeError as exc:
            print('Embedding search unavailable:', exc)
            print('Falling back to TF-IDF search')
            results = searcher.search_tfidf(args.query, top_k=args.top_k)

    print(f"Top {args.top_k} results for '{args.query}' using {args.method}:")
    for rank, (ident, score, text) in enumerate(results, start=1):
        print(f"{rank}. [{ident}] score={score:.4f} -- {text[:200]}")


if __name__ == '__main__':
    main()
