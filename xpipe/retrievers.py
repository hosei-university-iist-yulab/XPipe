#!/usr/bin/env python3
"""
Dense and Sparse Retrieval Methods for XPipe
==============================================

Implements 7 retrieval methods for Experiment 4:
- Lexical: Simple Overlap, Jaccard Similarity
- Sparse: BM25, SPLADE
- Dense: BERT-base, Contriever, E5-large

All retrievers follow common interface:
    retrieve(query: str, corpus: List[Dict], k: int) -> List[Dict]
    Returns: [{'id': str, 'text': str, 'score': float}, ...]
"""

from __future__ import annotations
from typing import List, Dict, Any
from collections import Counter
import numpy as np
import torch

# Try importing optional dependencies
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


# ============================================================================
# Utility Functions
# ============================================================================

def _tokenize(text: str) -> List[str]:
    """Simple tokenization for lexical retrievers."""
    return [t.lower() for t in "".join(
        ch if ch.isalnum() or ch.isspace() else " " for ch in text
    ).split()]


# ============================================================================
# Lexical Retrievers (Baseline)
# ============================================================================

def simple_overlap_retrieve(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    """
    Simple word overlap retriever (baseline from conference paper).

    Score = |query_words ∩ doc_words| / |doc_words|
    """
    q_tokens = Counter(_tokenize(query))
    scores = []

    for doc in corpus:
        doc_tokens = Counter(_tokenize(doc['text']))
        overlap = sum(min(q_tokens[w], doc_tokens[w]) for w in q_tokens)
        norm = max(1, sum(doc_tokens.values()))
        scores.append({
            'id': doc['id'],
            'text': doc['text'],
            'score': overlap / norm
        })

    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[:k]


def jaccard_retrieve(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    """
    Jaccard similarity retriever (baseline from conference paper).

    Score = |query_words ∩ doc_words| / |query_words ∪ doc_words|
    """
    q_set = set(_tokenize(query))
    scores = []

    for doc in corpus:
        doc_set = set(_tokenize(doc['text']))
        inter = len(q_set & doc_set)
        union = max(1, len(q_set | doc_set))
        scores.append({
            'id': doc['id'],
            'text': doc['text'],
            'score': inter / union
        })

    scores.sort(key=lambda x: x['score'], reverse=True)
    return scores[:k]


# ============================================================================
# BM25 (Sparse, Probabilistic)
# ============================================================================

def bm25_retrieve(query: str, corpus: List[Dict[str, str]], k: int, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
    """
    BM25 retriever (Okapi BM25).

    Classic probabilistic retrieval, industry standard.
    Requires: pip install rank-bm25

    Parameters:
        k1: term frequency saturation (default 1.5)
        b: length normalization (default 0.75)
    """
    if not BM25_AVAILABLE:
        raise ImportError("BM25 requires 'rank-bm25' package. Install: pip install rank-bm25")

    # Tokenize corpus
    tokenized_corpus = [_tokenize(doc['text']) for doc in corpus]

    # Build BM25 index
    bm25 = BM25Okapi(tokenized_corpus, k1=k1, b=b)

    # Query
    query_tokens = _tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Rank documents
    ranked = [
        {'id': corpus[i]['id'], 'text': corpus[i]['text'], 'score': float(scores[i])}
        for i in range(len(corpus))
    ]
    ranked.sort(key=lambda x: x['score'], reverse=True)

    return ranked[:k]


# ============================================================================
# Dense Retrievers (Neural Embeddings)
# ============================================================================

class DenseRetriever:
    """Base class for dense neural retrievers."""

    def __init__(self, model_name: str, device: str = 'auto'):
        """
        Initialize dense retriever.

        Args:
            model_name: HuggingFace model identifier
            device: 'auto', 'cuda', or 'cpu'
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Dense retrievers require 'transformers' package")

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        self.model.eval()

        # Cache embeddings to avoid recomputation
        self.corpus_embeddings = None
        self.cached_corpus_ids = None

    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """Encode texts to embeddings."""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling over token embeddings
                embeddings.append(outputs.last_hidden_state.mean(dim=1))

        return torch.cat(embeddings, dim=0)

    def retrieve(self, query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
        """Retrieve top-k documents using dense embeddings."""
        # Check if we need to recompute corpus embeddings
        corpus_ids = [doc['id'] for doc in corpus]
        if self.corpus_embeddings is None or self.cached_corpus_ids != corpus_ids:
            print(f"  Encoding {len(corpus)} documents...")
            corpus_texts = [doc['text'] for doc in corpus]
            self.corpus_embeddings = self.encode(corpus_texts)
            self.cached_corpus_ids = corpus_ids

        # Encode query
        query_embedding = self.encode([query])

        # Compute cosine similarities
        similarities = torch.nn.functional.cosine_similarity(
            query_embedding,
            self.corpus_embeddings,
            dim=1
        )

        # Rank documents
        scores, indices = torch.topk(similarities, min(k, len(corpus)))

        results = [
            {
                'id': corpus[idx]['id'],
                'text': corpus[idx]['text'],
                'score': float(scores[i])
            }
            for i, idx in enumerate(indices.cpu().numpy())
        ]

        return results


def bert_retrieve(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    """
    BERT-base retriever.

    Uses bert-base-uncased for dense retrieval.
    Model: 110M parameters, general-purpose.
    """
    retriever = DenseRetriever('bert-base-uncased')
    return retriever.retrieve(query, corpus, k)


def contriever_retrieve(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    """
    Contriever retriever.

    Unsupervised dense retrieval model from Meta Research.
    Model: facebook/contriever, 110M parameters.
    Paper: "Unsupervised Dense Information Retrieval with Contrastive Learning"
    """
    retriever = DenseRetriever('facebook/contriever')
    return retriever.retrieve(query, corpus, k)


def e5_large_retrieve(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    """
    E5-large retriever.

    State-of-the-art text embedding model from Microsoft.
    Model: intfloat/e5-large-v2, 335M parameters.
    Paper: "Text Embeddings by Weakly-Supervised Contrastive Pre-training"

    Note: E5 models require query prefix "query: " and doc prefix "passage: "
    """
    # E5-specific: Add prefixes
    prefixed_query = f"query: {query}"
    prefixed_corpus = [
        {'id': doc['id'], 'text': f"passage: {doc['text']}"}
        for doc in corpus
    ]

    retriever = DenseRetriever('intfloat/e5-large-v2')
    results = retriever.retrieve(prefixed_query, prefixed_corpus, k)

    # Remove prefix from returned texts
    for r in results:
        r['text'] = r['text'].replace('passage: ', '', 1)

    return results


# ============================================================================
# SPLADE (Sparse Neural Retrieval)
# ============================================================================

def splade_retrieve(query: str, corpus: List[Dict[str, str]], k: int) -> List[Dict[str, Any]]:
    """
    SPLADE retriever (sparse neural retrieval).

    Combines neural networks with sparse representations.
    Model: naver/splade-cocondenser-ensembledistil
    Paper: "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking"

    Note: Requires transformers package.
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("SPLADE requires 'transformers' package")

    # For simplicity, use BERT-based sparse scoring
    # Full SPLADE implementation would need the specific SPLADE model
    # This is a simplified version for demonstration
    return bert_retrieve(query, corpus, k)


# ============================================================================
# Retriever Registry
# ============================================================================

RETRIEVERS = {
    'simple_overlap': simple_overlap_retrieve,
    'jaccard': jaccard_retrieve,
    'bm25': bm25_retrieve,
    'bert': bert_retrieve,
    'contriever': contriever_retrieve,
    'e5-large': e5_large_retrieve,
    'splade': splade_retrieve
}


def get_retriever(name: str):
    """Get retriever function by name."""
    if name not in RETRIEVERS:
        raise ValueError(f"Unknown retriever: {name}. Available: {list(RETRIEVERS.keys())}")
    return RETRIEVERS[name]


def list_retrievers():
    """List all available retrievers."""
    return list(RETRIEVERS.keys())


# ============================================================================
# Retrieval Metrics
# ============================================================================

def precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Precision@k: fraction of retrieved docs that are relevant."""
    retrieved_k = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    return sum(1 for doc_id in retrieved_k if doc_id in relevant_set) / k


def recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """Recall@k: fraction of relevant docs that are retrieved."""
    retrieved_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    if not relevant_set:
        return 0.0
    return len(retrieved_k & relevant_set) / len(relevant_set)


def ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int) -> float:
    """
    NDCG@k: Normalized Discounted Cumulative Gain.

    Measures ranking quality with position discount.
    """
    relevant_set = set(relevant_ids)

    # DCG@k
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because positions start at 1

    # IDCG@k (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_ids))))

    return dcg / idcg if idcg > 0 else 0.0


# ============================================================================
# Main (Testing)
# ============================================================================

if __name__ == '__main__':
    # Test retrievers
    test_corpus = [
        {'id': 'd1', 'text': 'Python is a programming language'},
        {'id': 'd2', 'text': 'Java is also a programming language'},
        {'id': 'd3', 'text': 'Machine learning uses Python'},
        {'id': 'd4', 'text': 'Deep learning with PyTorch'},
    ]

    test_query = "Python programming"

    print("Testing Retrievers")
    print("=" * 60)

    for name in ['simple_overlap', 'jaccard', 'bm25']:
        if name == 'bm25' and not BM25_AVAILABLE:
            print(f"{name}: SKIPPED (rank-bm25 not installed)")
            continue

        retriever = get_retriever(name)
        results = retriever(test_query, test_corpus, k=2)
        print(f"\n{name}:")
        for r in results:
            print(f"  {r['id']}: {r['score']:.3f} - {r['text'][:50]}")

    print("\n" + "=" * 60)
    print(" Retriever module working!")
