"""
retrieval/semantic.py — InLegalBERT dense retrieval with FAISS
==============================================================

Wraps law-ai/InLegalBERT in a mean-pooling sentence encoder and
performs approximate nearest-neighbour (ANN) search via a FAISS
IndexFlatIP (inner-product on L2-normalised vectors = cosine similarity).

The FAISS index must be pre-built by running:
    python -m app.data.trends.retrieval_index

Until the index is built, ``SemanticRetriever.search()`` returns an empty
list and ``is_available()`` returns False.  The lexical pipeline continues
to work in this degraded mode.

Fine-tuned model
----------------
If ``app/data/models/inlegalbert-retrieval/`` exists (produced by
``finetune_retrieval.py``), it is used automatically in place of the
base law-ai/InLegalBERT weights.

Chunked embedding (long judgments)
-----------------------------------
``embed_chunked()`` splits text into overlapping 256-token chunks, embeds
each, and returns the mean-pooled aggregate.  Used by ``retrieval_index.py``
when building the index.  At query time, queries are typically short
enough to fit in a single 512-token pass.

Dependencies: transformers, torch, faiss-cpu
    pip install transformers torch faiss-cpu
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

import numpy as np
from pathlib import Path

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG       = Path(__file__).resolve().parent.parent
_DATA      = _PKG / "data" / "processed"
_RET_DIR   = _DATA / "retrieval"
_FAISS_IDX = _RET_DIR/"faiss.index"
_CASE_IDS  = _RET_DIR/"case_ids.json"
_FINETUNED = _PKG / "data" / "models" / "inlegalbert-retrieval"

_BASE_MODEL = "law-ai/InLegalBERT"
_MAX_TOKENS = 512
_CHUNK_SIZE = 256
_CHUNK_STRIDE = 128


def _best_model_name() -> str:
    return str(_FINETUNED) if _FINETUNED.exists() else _BASE_MODEL


def _preload_torch() -> None:
    """
    Load Torch before FAISS or large parquet-backed dataframes initialize
    native DLLs.  On Windows, importing Torch later can fail while loading
    c10.dll even when Torch is correctly installed.
    """
    try:
        import torch  # noqa: F401, PLC0415
    except ImportError as exc:
        raise ImportError(
            "transformers and torch are required for semantic retrieval.\n"
            "Run: pip install transformers torch"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            "torch failed to load its native DLLs. On Windows, make sure "
            "torch is imported before FAISS or large parquet-backed "
            "dataframes are loaded in this Python process."
        ) from exc


# ---------------------------------------------------------------------------
# Encoder (shared with retrieval_index.py)
# ---------------------------------------------------------------------------
class InLegalEncoder:
    """
    Mean-pooling sentence encoder backed by InLegalBERT.

    Lazily loads the model + tokenizer on first call.  Shared as a
    module-level singleton so the 400 MB weights are loaded at most once.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or _best_model_name()
        self._loaded = False
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None
        self._device = None

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            _preload_torch()
            try:
                from transformers import AutoModel, AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "transformers and torch are required for semantic retrieval.\n"
                    "Run: pip install transformers torch"
                ) from exc

            log.info("Loading encoder model: %s", self._model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModel.from_pretrained(self._model_name)
            self._model.eval()

            import torch as _torch
            self._device = _torch.device(
                "cuda" if _torch.cuda.is_available() else "cpu"
            )
            self._model.to(self._device)
            log.info("Encoder ready on %s", self._device)
            self._loaded = True

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts to L2-normalised embeddings [N, H]."""
        self._ensure_loaded()
        import torch

        all_embs: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start : start + batch_size]
            enc = self._tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=_MAX_TOKENS,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                out = self._model(**enc)

            mask = enc["attention_mask"].unsqueeze(-1).float()
            summed = (out.last_hidden_state * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-9)
            pooled = (summed / counts).cpu().numpy()

            norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-9)
            all_embs.append(pooled / norms)

        return np.vstack(all_embs)

    def encode_chunked(self, text: str, batch_size: int = 16) -> np.ndarray:
        """
        Embed a long text via overlapping chunks, return mean-pooled aggregate.

        Chunks: _CHUNK_SIZE tokens, stride _CHUNK_STRIDE tokens.
        Falls back to a single pass if the text fits in _MAX_TOKENS.
        Returns a single L2-normalised vector [H].
        """
        self._ensure_loaded()
        import torch

        enc = self._tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"][0]  # [total_tokens]

        if len(input_ids) <= _MAX_TOKENS - 2:
            return self.encode([text])[0]

        chunks: list[str] = []
        for start in range(0, len(input_ids), _CHUNK_STRIDE):
            end = start + _CHUNK_SIZE
            chunk_ids = input_ids[start:end]
            chunk_text = self._tokenizer.decode(
                chunk_ids, skip_special_tokens=True
            )
            chunks.append(chunk_text)
            if end >= len(input_ids):
                break

        chunk_embs = self.encode(chunks, batch_size=batch_size)    # [C, H]
        mean = chunk_embs.mean(axis=0)
        norm = np.linalg.norm(mean)
        return mean / max(norm, 1e-9)


# ---------------------------------------------------------------------------
# SemanticRetriever
# ---------------------------------------------------------------------------
class SemanticRetriever:
    """FAISS-backed dense retriever using InLegalBERT embeddings."""

    def __init__(self) -> None:
        self._lock   = threading.Lock()
        self._state  = "unloaded"   # "unloaded" | "unavailable" | "ready"
        self._index  = None
        self._case_ids: list[str] = []
        self._encoder: InLegalEncoder | None = None

    def is_available(self) -> bool:
        """Return True only if the FAISS index exists and has been loaded."""
        self._try_load()
        return self._state == "ready"

    def _try_load(self) -> None:
        if self._state != "unloaded":
            return
        with self._lock:
            if self._state != "unloaded":
                return
            if not _FAISS_IDX.exists() or not _CASE_IDS.exists():
                log.warning(
                    "FAISS index not found at %s.  "
                    "Run retrieval_index.py to build it.  "
                    "Semantic retrieval disabled.",
                    _FAISS_IDX,
                )
                self._state = "unavailable"
                return
            try:
                _preload_torch()
            except (ImportError, RuntimeError) as exc:
                log.warning("%s  Semantic retrieval disabled.", exc)
                self._state = "unavailable"
                return
            try:
                import faiss
            except ImportError:
                log.warning(
                    "faiss-cpu not installed.  "
                    "Run: pip install faiss-cpu.  "
                    "Semantic retrieval disabled."
                )
                self._state = "unavailable"
                return

            log.info("Loading FAISS index from %s…", _FAISS_IDX)
            self._index = faiss.read_index(str(_FAISS_IDX))
            with open(_CASE_IDS) as f:
                self._case_ids = json.load(f)
            self._encoder = _get_encoder()
            self._state = "ready"
            log.info("SemanticRetriever ready (%d vectors)", self._index.ntotal)

    # ------------------------------------------------------------------
    # Search methods
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 150) -> list[dict]:
        """Free-text semantic search.

        Returns list of {case_id, semantic_score, semantic_rank}.
        Returns [] if FAISS is unavailable.
        """
        if not self.is_available():
            return []

        qvec = self._encoder.encode([query])   # [1, H]
        return self._faiss_search(qvec, top_k)

    def search_by_case(
        self, case_id: str, top_k: int = 150, exclude_self: bool = True
    ) -> list[dict]:
        """Find judgments semantically similar to a given case.

        Uses the stored FAISS vector for that case (no re-encoding needed).
        Returns [] if the case_id is not in the index.
        """
        if not self.is_available():
            return []

        if case_id not in self._case_ids:
            log.debug("case_id %s not in FAISS index", case_id)
            return []

        idx = self._case_ids.index(case_id)
        vec = self._index.reconstruct(idx).reshape(1, -1)  # [1, H]
        results = self._faiss_search(vec, top_k + (1 if exclude_self else 0))
        if exclude_self:
            results = [r for r in results if r["case_id"] != case_id]
        return results[:top_k]

    def embed_query(self, text: str) -> np.ndarray:
        """Return L2-normalised embedding for arbitrary text (for case-mode)."""
        if not self.is_available():
            return np.array([])
        return self._encoder.encode([text])[0]

    def _faiss_search(self, qvec: np.ndarray, top_k: int) -> list[dict]:
        scores, indices = self._index.search(
            qvec.astype(np.float32), top_k
        )
        results: list[dict] = []
        for rank, (score, idx) in enumerate(
            zip(scores[0], indices[0]), start=1
        ):
            if idx < 0:
                continue
            case_name = self._case_ids.get(str(idx), "").split("_")
            final_name = ""
            for i in range(len(case_name)-1):
                final_name +=  case_name[i] + " "
            results.append({
                "case_id":        final_name,
                "semantic_score": float(score),
                "semantic_rank":  rank,
            })
        return results


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_encoder_instance: InLegalEncoder | None = None
_encoder_lock = threading.Lock()

_sem_retriever: SemanticRetriever | None = None
_sem_lock = threading.Lock()


def _get_encoder() -> InLegalEncoder:
    global _encoder_instance
    if _encoder_instance is None:
        with _encoder_lock:
            if _encoder_instance is None:
                _encoder_instance = InLegalEncoder()
    return _encoder_instance


def get_semantic_retriever() -> SemanticRetriever:
    global _sem_retriever
    if _sem_retriever is None:
        with _sem_lock:
            if _sem_retriever is None:
                _sem_retriever = SemanticRetriever()
    return _sem_retriever


def get_encoder() -> InLegalEncoder:
    """Public accessor for the shared encoder (used by retrieval_index.py)."""
    return _get_encoder()
