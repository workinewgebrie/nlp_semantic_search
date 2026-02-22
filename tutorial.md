## Introduction to Natural Language Processing (NLP)

Natural Language Processing (NLP) is the branch of AI that builds systems to understand, interpret and generate human language. Core difficulties include ambiguity, variability, and contextual meaning.

Key components and common tasks:
- Tokenization — split text into words/sentences.
- Normalization — lowercase, remove punctuation, stopwords, stemming, lemmatization.
- POS tagging — identify grammatical class of tokens.
- Named Entity Recognition (NER) — find people, places, organizations.
- Parsing & relationships — dependency parsing and relation extraction.
- Coreference resolution — link pronouns to entities.
- Text classification — sentiment, topic, spam detection.
- Language modelling & generation — GPT-style models.

Popular Python libraries:
- NLTK — academic, corpora, preprocessing, tagging, lemmatization.
- spaCy — production-grade NLP pipelines, fast tokenization, POS, NER, vectors.
- Gensim — topic models and Word2Vec.
- scikit-learn — feature extraction (BoW, TF-IDF) and ML models.
- sentence-transformers — easy-to-use pretrained sentence/document embeddings.

Text representation summary:
- One-hot / Bag-of-words — simple counts.
- TF-IDF — counts scaled by inverse document frequency.
- Word embeddings — Word2Vec, GloVe, fastText.
- Contextual embeddings — BERT and other transformers.

This repository includes `semantic_search.py` with two retrieval methods:
1. TF-IDF + cosine similarity
2. Embedding-based search using Sentence-Transformers

See `semantic_search.py` for code and usage examples.
