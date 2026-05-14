"""
retrieval — hybrid legal case retrieval package
================================================

Entry point::

    from app.retrieval import retrieve

    response = retrieve(
        query="Motor accident compensation Section 166 MV Act",
        filters={"year_min": 2000, "categories": ["MotorAccident"]},
        top_k=20,
    )

The package provides two independent retrieval pipelines:

- Lexical (BM25 + deterministic signals) — always available, auto-built
  from judgments.parquet on first use and cached.

- Semantic (InLegalBERT + FAISS ANN) — optional, requires the offline
  index to be built first:
      python -m app.data.trends.retrieval_index

A fine-tuned sentence-transformer encoder (law-ai/InLegalBERT wrapped with
mean-pooling, trained with MultipleNegativesRankingLoss on citation pairs)
improves retrieval quality if present at:
      app/data/models/inlegalbert-retrieval/

Build it with:
      python -m app.data.trends.finetune_retrieval
"""
from .fusion import retrieve, DEFAULT_WEIGHTS
from .lexical import get_lexical_retriever
from .semantic import get_semantic_retriever

__all__ = ["retrieve", "DEFAULT_WEIGHTS", "get_lexical_retriever", "get_semantic_retriever"]
