"""
retrieval/lexical.py — BM25 keyword retrieval + deterministic signal tables
===========================================================================

Provides two capabilities:

1. BM25 keyword search over all 26,688 judgments.
   Indexes: title × 3  +  headnote  +  act_text × 2  +  sections_cited
   Tokenisation: lowercase, split on non-alphanumeric, drop tokens ≤ 1 char,
   drop generic stopwords.  Domain terms (IPC, crpc, writ, etc.) are kept.

2. Deterministic similarity signals for a set of candidate case_ids:
   • shared_sections   — overlapping IPC/CrPC sections (from ipc_crpc_long)
   • shared_statutes   — overlapping statute names (from act_text)
   • shared_articles   — overlapping constitutional article numbers
   • citation_overlap  — number of cases both cite (from citations_resolved)
   • direct_citation   — one case directly cites the other
   • category_match    — same case category

The BM25 index is built lazily on first use and saved to
``app/data/processed/retrieval/bm25_corpus.pkl`` for reuse across restarts.
If the pickle is stale (judgments.parquet is newer), it is rebuilt.

Dependencies: rank_bm25  (pip install rank-bm25)
"""
from __future__ import annotations

import logging
import pickle
import re
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .ner import extract_statutes_from_text, extract_articles_from_sections

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PKG      = Path(__file__).resolve().parent.parent          # app/
_DATA     = _PKG / "data" / "processed"
_RET_DIR  = _DATA / "retrieval"
_JUDGMENTS = _DATA / "judgments.parquet"
_BM25_PKL  = _RET_DIR / "bm25_corpus.pkl"

_TRENDS   = _DATA / "trends"
_IPC_LONG = _TRENDS / "ipc_crpc_long.parquet"
_CITS     = _TRENDS / "citations_resolved.parquet"
_CATS     = _TRENDS / "case_categories.parquet"

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------
_STOPWORDS = frozenset("""
a an the of in on at by to for with and or is are was were be been being
has have had do does did will would could should may might shall
this that these those it its he she they we their them their
his her our your my its from into through during after before
above below between through during including until against
among throughout despite towards upon concerning of about into
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[a-z0-9])*")


def _tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) > 1 and t not in _STOPWORDS]


def _build_doc(row: dict) -> list[str]:
    """Combine fields into a single token list, weighting title and act_text."""
    title  = str(row.get("title",   "") or "")
    head   = str(row.get("headnote","") or "")
    acts   = str(row.get("act_text","") or "")
    secs   = row.get("sections_cited")
    secs_text = ""
    if secs is not None:
        try:
            secs_text = " ".join(str(s) for s in secs if isinstance(s, str))
        except TypeError:
            pass
    # Title weight=3, act_text weight=2, headnote and sections weight=1
    combined = (
        f"{title} {title} {title} "
        f"{acts} {acts} "
        f"{head} "
        f"{secs_text}"
    )
    return _tokenize(combined)


# ---------------------------------------------------------------------------
# LexicalRetriever
# ---------------------------------------------------------------------------
class LexicalRetriever:
    """BM25 retriever with deterministic signal tables, lazily loaded."""

    def __init__(self) -> None:
        self._lock    = threading.Lock()
        self._ready   = False

        # BM25 corpus
        self._bm25    = None
        self._case_ids: list[str] = []

        # Metadata lookup (case_id → dict)
        self._meta: dict[str, dict] = {}

        # Signal lookups
        self._sections:  dict[str, frozenset] = {}   # case_id → frozenset{"IPC:302", …}
        self._statutes:  dict[str, frozenset] = {}   # case_id → frozenset{"Indian Penal Code", …}
        self._articles:  dict[str, frozenset] = {}   # case_id → frozenset{"21", "32", …}
        self._cit_out:   dict[str, frozenset] = {}   # case_id → frozenset of cited case_ids
        self._cit_in:    dict[str, frozenset] = {}   # case_id → frozenset of citing case_ids
        self._categories: dict[str, str] = {}        # case_id → category string

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            self._build()

    def _build(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for lexical retrieval.  "
                "Run: pip install rank-bm25"
            ) from exc

        log.info("Loading judgments for LexicalRetriever…")
        df = pd.read_parquet(_JUDGMENTS)

        # ── Metadata ──────────────────────────────────────────────────
        self._meta = {}
        for row in df[["case_id","title","year","headnote","bench_size",
                        "act_text","sections_cited","indiankanoon_url"]].to_dict("records"):
            cid = row["case_id"]
            hn  = str(row.get("headnote") or "")
            self._meta[cid] = {
                "case_id":        cid,
                "title":          str(row.get("title") or ""),
                "year":           int(row.get("year") or 0),
                "bench_size":     int(row.get("bench_size") or 0),
                "headnote_preview": hn[:280] + ("…" if len(hn) > 280 else ""),
                "indiankanoon_url": str(row.get("indiankanoon_url") or ""),
            }

        # ── Signal: statutes per case (from act_text) ─────────────────
        self._statutes = {
            row["case_id"]: frozenset(extract_statutes_from_text(
                str(row.get("act_text") or "")))
            for row in df[["case_id","act_text"]].to_dict("records")
        }

        # ── Signal: constitutional articles per case ──────────────────
        self._articles = {
            row["case_id"]: frozenset(extract_articles_from_sections(
                row.get("sections_cited")))
            for row in df[["case_id","sections_cited"]].to_dict("records")
        }

        # ── Signal: IPC/CrPC sections per case ───────────────────────
        if _IPC_LONG.exists():
            ipc_df = pd.read_parquet(_IPC_LONG)
            secs: dict[str, set] = {}
            for row in ipc_df[["case_id","statute","section"]].to_dict("records"):
                key = f"{row['statute']}:{row['section']}"
                secs.setdefault(row["case_id"], set()).add(key)
            self._sections = {k: frozenset(v) for k, v in secs.items()}

        # ── Signal: citation graph ────────────────────────────────────
        if _CITS.exists():
            cit_df = pd.read_parquet(_CITS)
            out: dict[str, set] = {}
            inn: dict[str, set] = {}
            for row in cit_df[["source_id","target_id"]].to_dict("records"):
                s, t = row["source_id"], row["target_id"]
                out.setdefault(s, set()).add(t)
                inn.setdefault(t, set()).add(s)
            self._cit_out = {k: frozenset(v) for k, v in out.items()}
            self._cit_in  = {k: frozenset(v) for k, v in inn.items()}

        # ── Signal: categories ─────────────────────────────────────────
        if _CATS.exists():
            cats_df = pd.read_parquet(_CATS)
            self._categories = dict(
                zip(cats_df["case_id"], cats_df["case_category"])
            )
            # Merge category into meta
            for cid, cat in self._categories.items():
                if cid in self._meta:
                    self._meta[cid]["category"] = cat

        # Fill missing category
        for meta in self._meta.values():
            meta.setdefault("category", "Other")

        # ── BM25 corpus ───────────────────────────────────────────────
        bm25_is_fresh = (
            _BM25_PKL.exists()
            and _BM25_PKL.stat().st_mtime > _JUDGMENTS.stat().st_mtime
        )

        if bm25_is_fresh:
            log.info("Loading cached BM25 index from %s", _BM25_PKL)
            with open(_BM25_PKL, "rb") as f:
                saved = pickle.load(f)
            self._bm25     = saved["bm25"]
            self._case_ids = saved["case_ids"]
        else:
            log.info("Building BM25 index for %d judgments…", len(df))
            corpus: list[list[str]] = []
            self._case_ids = []
            for row in df[["case_id","title","headnote","act_text",
                            "sections_cited"]].to_dict("records"):
                self._case_ids.append(row["case_id"])
                corpus.append(_build_doc(row))
            self._bm25 = BM25Okapi(corpus)
            _RET_DIR.mkdir(parents=True, exist_ok=True)
            with open(_BM25_PKL, "wb") as f:
                pickle.dump({"bm25": self._bm25, "case_ids": self._case_ids}, f)
            log.info("BM25 index cached at %s", _BM25_PKL)

        self._ready = True
        log.info("LexicalRetriever ready (%d docs)", len(self._case_ids))

    # ------------------------------------------------------------------
    # BM25 search
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 150) -> list[dict]:
        """Return up to top_k results sorted by BM25 score (descending).

        Each result: {case_id, bm25_score, bm25_rank}
        """
        self._ensure_loaded()
        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1]

        results: list[dict] = []
        rank = 1
        for idx in top_indices:
            s = float(scores[idx])
            if s <= 0:
                break
            if rank > top_k:
                break
            results.append({
                "case_id":    self._case_ids[idx],
                "bm25_score": s,
                "bm25_rank":  rank,
            })
            rank += 1

        return results

    # ------------------------------------------------------------------
    # Deterministic signal computation
    # ------------------------------------------------------------------
    def get_signals(
        self,
        candidate_ids: list[str],
        query_signals: dict[str, set],
        query_case_id: str | None = None,
    ) -> dict[str, dict]:
        """Compute per-candidate similarity signals.

        query_signals has keys: statutes, articles, sections (all sets of str).
        query_case_id is set when doing "find similar to case" queries.

        Returns {case_id: {shared_sections, shared_statutes, shared_articles,
                           citation_overlap, direct_citation, category_match,
                           pagerank_boost}}
        """
        self._ensure_loaded()

        q_secs     = query_signals.get("sections", set())
        q_statutes = query_signals.get("statutes",  set())
        q_arts     = query_signals.get("articles",  set())
        q_cat      = self._categories.get(query_case_id, "") if query_case_id else ""

        # For citation overlap: cases the query case cites / is cited by
        q_cit_out = self._cit_out.get(query_case_id, frozenset()) if query_case_id else frozenset()
        q_cit_in  = self._cit_in.get(query_case_id, frozenset())  if query_case_id else frozenset()
        q_cit_all = q_cit_out | q_cit_in

        out: dict[str, dict] = {}
        for cid in candidate_ids:
            c_secs     = self._sections.get(cid,  frozenset())
            c_statutes = self._statutes.get(cid,  frozenset())
            c_arts     = self._articles.get(cid,  frozenset())
            c_cit_out  = self._cit_out.get(cid,   frozenset())
            c_cit_in_  = self._cit_in.get(cid,    frozenset())
            c_cit_all  = c_cit_out | c_cit_in_
            c_cat      = self._categories.get(cid, "Other")

            shared_secs     = sorted(q_secs & c_secs)           if q_secs else []
            shared_statutes = sorted(q_statutes & c_statutes)    if q_statutes else []
            shared_arts     = sorted(q_arts & c_arts)            if q_arts else []

            direct_cit = (
                (query_case_id is not None)
                and (query_case_id in c_cit_out or cid in q_cit_out)
            )

            cit_overlap = int(len(q_cit_all & c_cit_all)) if q_cit_all else 0

            out[cid] = {
                "shared_sections":   shared_secs,
                "shared_statutes":   shared_statutes,
                "shared_articles":   shared_arts,
                "direct_citation":   direct_cit,
                "citation_overlap":  cit_overlap,
                "category_match":    (c_cat == q_cat) if q_cat else False,
                "candidate_category": c_cat,
            }
        return out

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def get_meta(self, case_id: str) -> dict:
        self._ensure_loaded()
        return self._meta.get(case_id, {"case_id": case_id, "title": case_id,
                                         "year": 0, "bench_size": 0,
                                         "category": "Other", "headnote_preview": "","indiankanoon_url": "",})

    def get_all_case_ids(self) -> list[str]:
        self._ensure_loaded()
        return list(self._case_ids)

    def search_titles(self, query: str, top_k: int = 20) -> list[dict]:
        """Search case titles for autocomplete / case selector."""
        self._ensure_loaded()
        q = query.lower()
        results = []
        for cid, meta in self._meta.items():
            if q in meta["title"].lower():
                results.append({"case_id": cid, "title": meta["title"],
                                  "year": meta["year"], "category": meta["category"]})
                if len(results) >= top_k * 5:
                    break
        results.sort(key=lambda x: x["title"].lower().index(q) if q in x["title"].lower() else 999)
        return results[:top_k]


# Module-level singleton
_retriever: LexicalRetriever | None = None
_singleton_lock = threading.Lock()


def get_lexical_retriever() -> LexicalRetriever:
    """Return the shared LexicalRetriever instance."""
    global _retriever
    if _retriever is None:
        with _singleton_lock:
            if _retriever is None:
                _retriever = LexicalRetriever()
    return _retriever
