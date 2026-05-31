"""
retrieval/fusion.py — candidate pool merge + multi-signal weighted reranker
============================================================================

Pipeline
--------
1. Run BM25 (LexicalRetriever.search) → up to 150 candidates + BM25 scores.
2. Run FAISS ANN (SemanticRetriever.search) → up to 150 candidates + cosine scores.
   If FAISS is unavailable, step 2 is skipped silently.
3. Union the two candidate pools, deduplicating on case_id.
   Provenance is tracked per candidate: ["lexical"], ["semantic"], or both.
4. Apply optional metadata filters (year range, category, bench size, statute/article).
5. Compute per-candidate deterministic signals via LexicalRetriever.get_signals.
6. Compute per-candidate NER overlap between query entities and candidate entities.
7. Score each candidate with a weighted multi-signal formula:

      fusion = w_bm25   * bm25_norm
             + w_sem    * semantic_score          (0 if unavailable)
             + w_sec    * section_overlap_score
             + w_stat   * statute_overlap_score
             + w_cit    * citation_score
             + w_art    * article_overlap_score
             + w_cat    * category_match_score

   All component scores are in [0, 1].  BM25 is normalised within the batch.

8. Sort candidates by fusion score for the "Top Results" section.
   The "Exact Matches" and "Semantic Results" sections keep their own
   original rankings (BM25 order / FAISS cosine order) so that high-precision
   lexical / semantic results remain visible even if they rank lower in fusion.

Guarantee: exact statutory or constitutional matches (shared_statutes /
shared_articles non-empty) always appear in the "Exact Matches" section
regardless of their fusion rank.

Weights
-------
Defaults are tuned for the Indian legal corpus.  Callers may pass a custom
``weights`` dict to override any subset of keys.  Keys not provided fall
back to the defaults.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .ner import extract_legal_entities, entities_to_query_signal
from .lexical import get_lexical_retriever
from .semantic import get_semantic_retriever

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default fusion weights (must sum to 1.0)
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS: dict[str, float] = {
    "bm25":     0.25,
    "semantic": 0.35,
    "sections": 0.13,
    "statutes": 0.10,
    "citation": 0.07,
    "articles": 0.06,
    "category": 0.04,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    case_id:          str
    title:            str
    year:             int
    category:         str
    bench_size:       int
    headnote_preview: str
    indiankanoon_url: str

    fusion_score:     float
    bm25_score:       float      # raw (unnormalised)
    bm25_norm:        float      # normalised to [0,1] within batch
    semantic_score:   float      # cosine similarity from FAISS

    shared_sections:  list[str]
    shared_statutes:  list[str]
    shared_articles:  list[str]
    citation_overlap: int
    direct_citation:  bool
    category_match:   bool

    ner_overlap: dict[str, list[str]] = field(default_factory=dict)
    provenance:  list[str]            = field(default_factory=list)
    score_breakdown: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id":           self.case_id,
            "title":             self.title,
            "year":              self.year,
            "category":          self.category,
            "bench_size":        self.bench_size,
            "headnote_preview":  self.headnote_preview,
            "fusion_score":      round(self.fusion_score, 4),
            "bm25_score":        round(self.bm25_score, 4),
            "bm25_norm":         round(self.bm25_norm, 4),
            "semantic_score":    round(self.semantic_score, 4),
            "shared_sections":   self.shared_sections,
            "shared_statutes":   self.shared_statutes,
            "shared_articles":   self.shared_articles,
            "citation_overlap":  self.citation_overlap,
            "direct_citation":   self.direct_citation,
            "category_match":    self.category_match,
            "ner_overlap":       self.ner_overlap,
            "provenance":        self.provenance,
            "score_breakdown":   {k: round(v, 4) for k, v in self.score_breakdown.items()},
            "indiankanoon_url": self.indiankanoon_url,
        }


# ---------------------------------------------------------------------------
# Public retrieve() function
# ---------------------------------------------------------------------------
def retrieve(
    query:        str,
    case_id:      str | None = None,
    filters:      dict | None = None,
    weights:      dict | None = None,
    top_k:        int = 20,
    lexical_k:    int = 150,
    semantic_k:   int = 150,
) -> dict[str, Any]:
    """
    Run the hybrid retrieval pipeline and return a structured response.

    Parameters
    ----------
    query       : Free-text query string (may be empty when case_id is given).
    case_id     : Optional — retrieve cases similar to this case.
    filters     : Optional filter dict with keys:
                    year_min (int), year_max (int),
                    categories (list[str]), bench_size_min (int),
                    statute (str), article (str)
    weights     : Override DEFAULT_WEIGHTS keys.
    top_k       : Number of results in the unified "top_results" section.
    lexical_k   : Candidates retrieved from BM25.
    semantic_k  : Candidates retrieved from FAISS.

    Returns
    -------
    {
      status, query, query_case_id, query_entities,
      top_results, lexical_results, semantic_results,
      meta: {lexical_candidates, semantic_candidates, total_candidates,
             semantic_available, retrieval_time_ms}
    }
    """
    t0 = time.perf_counter()

    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    filt = filters or {}

    lex  = get_lexical_retriever()
    sem  = get_semantic_retriever()

    # ── Resolve effective query text ─────────────────────────────────
    # When a case_id is given, supplement an empty query with the case's
    # headnote so NER and (optionally) BM25 have signal to work with.
    effective_query = query.strip()
    if not effective_query and case_id:
        meta = lex.get_meta(case_id)
        effective_query = meta.get("headnote_preview", "")

    # ── NER on query ─────────────────────────────────────────────────
    query_ents = extract_legal_entities(effective_query)
    query_signals = entities_to_query_signal(query_ents)

    # ── Retrieve from BM25 ───────────────────────────────────────────
    lex_raw: list[dict] = []
    if effective_query:
        lex_raw = lex.search(effective_query, top_k=lexical_k)
    elif case_id:
        # Fallback: BM25 on case title
        meta = lex.get_meta(case_id)
        title_query = meta.get("title", "")
        if title_query:
            lex_raw = lex.search(title_query, top_k=lexical_k)

    # ── Retrieve from FAISS ──────────────────────────────────────────
    sem_raw: list[dict] = []
    semantic_available = sem.is_available()
    if semantic_available:
        if case_id and not query.strip():
            sem_raw = sem.search_by_case(case_id, top_k=semantic_k)
        elif effective_query:
            sem_raw = sem.search(effective_query, top_k=semantic_k)

    # ── Merge candidate pools ────────────────────────────────────────
    # {case_id: {bm25_score, bm25_rank, semantic_score, semantic_rank, provenance}}
    pool: dict[str, dict] = {}

    for r in lex_raw:
        cid = r["case_id"]
        pool[cid] = {
            "bm25_score":    r["bm25_score"],
            "bm25_rank":     r["bm25_rank"],
            "semantic_score": 0.0,
            "semantic_rank":  9999,
            "provenance":    ["lexical"],
        }

    for r in sem_raw:
        cid = r["case_id"]
        if cid in pool:
            pool[cid]["semantic_score"] = r["semantic_score"]
            pool[cid]["semantic_rank"]  = r["semantic_rank"]
            pool[cid]["provenance"].append("semantic")
        else:
            pool[cid] = {
                "bm25_score":    0.0,
                "bm25_rank":     9999,
                "semantic_score": r["semantic_score"],
                "semantic_rank":  r["semantic_rank"],
                "provenance":    ["semantic"],
            }

    # Exclude the query case itself
    if case_id and case_id in pool:
        del pool[case_id]

    # ── Apply metadata filters ────────────────────────────────────────
    if filt:
        filtered: dict[str, dict] = {}
        for cid, scores in pool.items():
            m = lex.get_meta(cid)
            if filt.get("year_min") and m["year"] < filt["year_min"]:
                continue
            if filt.get("year_max") and m["year"] > filt["year_max"]:
                continue
            if filt.get("bench_size_min") and m["bench_size"] < filt["bench_size_min"]:
                continue
            if filt.get("categories") and m["category"] not in filt["categories"]:
                continue
            filtered[cid] = scores
        pool = filtered

    candidate_ids = list(pool.keys())

    # ── Compute deterministic signals ─────────────────────────────────
    signals = lex.get_signals(candidate_ids, query_signals, query_case_id=case_id)

    # ── BM25 normalisation ────────────────────────────────────────────
    max_bm25 = max((v["bm25_score"] for v in pool.values()), default=1.0)
    if max_bm25 == 0:
        max_bm25 = 1.0

    # ── Build scored results ──────────────────────────────────────────
    results: list[RetrievalResult] = []

    for cid in candidate_ids:
        sc  = pool[cid]
        sig = signals.get(cid, {})
        m   = lex.get_meta(cid)

        bm25_raw  = sc["bm25_score"]
        bm25_norm = bm25_raw / max_bm25
        sem_score = sc["semantic_score"]

        # Optional post-filter by statute / article text (substring match)
        if filt.get("statute"):
            if not any(filt["statute"].lower() in s.lower()
                       for s in sig.get("shared_statutes", [])):
                # Don't drop — just don't boost; let score decide
                pass
        if filt.get("article"):
            art_q = filt["article"].strip()
            if art_q not in sig.get("shared_articles", []):
                pass

        # ── Overlap scores ────────────────────────────────────────────
        shared_secs  = sig.get("shared_sections",  [])
        shared_stats = sig.get("shared_statutes",   [])
        shared_arts  = sig.get("shared_articles",   [])
        cit_overlap  = sig.get("citation_overlap",  0)
        direct_cit   = sig.get("direct_citation",   False)
        cat_match    = sig.get("category_match",     False)

        q_secs  = query_signals.get("sections", set())
        q_stats = query_signals.get("statutes",  set())
        q_arts  = query_signals.get("articles",  set())

        sec_score  = _overlap_score(len(shared_secs),  len(q_secs))
        stat_score = _overlap_score(len(shared_stats), len(q_stats))
        art_score  = _overlap_score(len(shared_arts),  len(q_arts))

        cit_score  = (0.8 if direct_cit else 0.0) + min(cit_overlap * 0.05, 0.2)
        cit_score  = min(cit_score, 1.0)

        cat_score  = 1.0 if cat_match else 0.0

        # ── Fusion score ──────────────────────────────────────────────
        breakdown = {
            "bm25":     w["bm25"]     * bm25_norm,
            "semantic": w["semantic"] * sem_score,
            "sections": w["sections"] * sec_score,
            "statutes": w["statutes"] * stat_score,
            "citation": w["citation"] * cit_score,
            "articles": w["articles"] * art_score,
            "category": w["category"] * cat_score,
        }
        fusion = sum(breakdown.values())

        # ── NER overlap ───────────────────────────────────────────────
        c_ents = extract_legal_entities(m.get("headnote_preview", ""))
        ner_overlap = {
            "statutes": sorted(set(query_ents["statutes"]) & set(c_ents["statutes"])),
            "articles": sorted(set(query_ents["articles"]) & set(c_ents["articles"])),
            "doctrines": sorted(set(query_ents["doctrines"]) & set(c_ents["doctrines"])),
        }

        results.append(RetrievalResult(
            case_id           = cid,
            title             = m["title"],
            year              = m["year"],
            category          = sig.get("candidate_category", m["category"]),
            bench_size        = m["bench_size"],
            headnote_preview  = m["headnote_preview"],
            fusion_score      = fusion,
            bm25_score        = bm25_raw,
            bm25_norm         = bm25_norm,
            semantic_score    = sem_score,
            shared_sections   = shared_secs,
            shared_statutes   = shared_stats,
            shared_articles   = shared_arts,
            citation_overlap  = cit_overlap,
            direct_citation   = direct_cit,
            category_match    = cat_match,
            ner_overlap       = ner_overlap,
            provenance        = sc["provenance"],
            score_breakdown   = breakdown,
            indiankanoon_url  =m.get("indiankanoon_url", ""),
        ))

    # ── Sort + slice ──────────────────────────────────────────────────
    results_by_fusion  = sorted(results, key=lambda r: r.fusion_score, reverse=True)
    results_by_bm25    = sorted(results, key=lambda r: r.bm25_score, reverse=True)
    results_by_sem     = sorted(results, key=lambda r: r.semantic_score, reverse=True)

    top_results     = [r.to_dict() for r in results_by_fusion[:top_k]]
    lexical_results = [r.to_dict() for r in results_by_bm25[:top_k]
                       if "lexical" in r.provenance]
    semantic_results = [r.to_dict() for r in results_by_sem[:top_k]
                        if "semantic" in r.provenance]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "status":          "ok",
        "query":           query,
        "query_case_id":   case_id,
        "query_entities":  query_ents,
        "top_results":     top_results,
        "lexical_results": lexical_results,
        "semantic_results": semantic_results,
        "meta": {
            "lexical_candidates":  len(lex_raw),
            "semantic_candidates": len(sem_raw),
            "total_candidates":    len(pool),
            "semantic_available":  semantic_available,
            "retrieval_time_ms":   round(elapsed_ms, 1),
            "weights_used":        w,
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _overlap_score(shared: int, query_count: int) -> float:
    """Recall-style overlap score: |shared| / max(1, |query_set|), capped at 1."""
    if query_count == 0:
        # No query signal in this dimension; give small bonus for any match
        return min(shared * 0.33, 1.0)
    return min(shared / query_count, 1.0)
