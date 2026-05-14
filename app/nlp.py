"""spaCy + Blackstone NLP pipeline loader and analysis helpers."""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import spacy
from spacy.language import Language

log = logging.getLogger(__name__)

# Regex patterns tuned for Indian legal text
INDIAN_CITATION_PATTERNS = [
    # AIR 1973 SC 1461
    re.compile(r"\bAIR\s+\d{4}\s+[A-Z]{2,4}\s+\d+\b"),
    # (1973) 4 SCC 225
    re.compile(r"\(\d{4}\)\s+\d+\s+SCC\s+\d+\b"),
    # 2019 SCC OnLine SC 1234
    re.compile(r"\b\d{4}\s+SCC\s+OnLine\s+[A-Z]{2,4}\s+\d+\b"),
    # ILR 1950 Mad 100
    re.compile(r"\bILR\s+\d{4}\s+[A-Za-z]+\s+\d+\b"),
]

INDIAN_STATUTE_PATTERNS = [
    # Section 302 of IPC / Section 302 IPC / Sec. 420 of the Indian Penal Code
    re.compile(
        r"\b(?:Section|Sec\.?|S\.)\s*\d+[A-Z]?"
        r"(?:\s*(?:of\s+(?:the\s+)?)?"
        r"(?:IPC|CrPC|CPC|Cr\.P\.C\.|C\.P\.C\.|"
        r"Indian\s+Penal\s+Code|"
        r"Code\s+of\s+Criminal\s+Procedure|"
        r"Code\s+of\s+Civil\s+Procedure|"
        r"Evidence\s+Act|Constitution(?:\s+of\s+India)?))?",
        re.IGNORECASE,
    ),
    # Article 14 of the Constitution
    re.compile(r"\bArticle\s+\d+[A-Z]?(?:\s*of\s+the\s+Constitution)?\b", re.IGNORECASE),
]


@lru_cache(maxsize=2)
def get_nlp(model_name: str, fallback_model: str = "en_core_web_sm") -> Language:
    """Load spaCy model; prefer Blackstone, fall back to a general model."""
    for name in (model_name, fallback_model):
        try:
            log.info("Loading spaCy model: %s", name)
            return spacy.load(name)
        except OSError as exc:
            log.warning("Could not load '%s': %s", name, exc)

    log.warning("No spaCy model available; using blank English pipeline.")
    return spacy.blank("en")


def extract_citations(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for pattern in INDIAN_CITATION_PATTERNS:
        for m in pattern.finditer(text):
            hits.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    return hits


def extract_statutes(text: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    for pattern in INDIAN_STATUTE_PATTERNS:
        for m in pattern.finditer(text):
            value = m.group(0).strip()
            if value:
                hits.append({"text": value, "start": m.start(), "end": m.end()})
    # de-duplicate overlaps (keep longest)
    hits.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    dedup: list[dict[str, Any]] = []
    last_end = -1
    for h in hits:
        if h["start"] >= last_end:
            dedup.append(h)
            last_end = h["end"]
    return dedup


def analyze_text(nlp: Language, text: str) -> dict[str, Any]:
    """Run the full NLP analysis pipeline on the provided text."""
    doc = nlp(text)

    entities = [
        {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char,
        }
        for ent in doc.ents
    ]

    sentences = [s.text.strip() for s in doc.sents] if doc.has_annotation("SENT_START") else []

    # Token-level POS summary (limited for payload size)
    pos_counts: dict[str, int] = {}
    for tok in doc:
        if tok.is_alpha:
            pos_counts[tok.pos_] = pos_counts.get(tok.pos_, 0) + 1

    return {
        "model": nlp.meta.get("name", "unknown"),
        "lang": nlp.meta.get("lang", "en"),
        "pipeline": nlp.pipe_names,
        "num_tokens": len(doc),
        "num_sentences": len(sentences),
        "entities": entities,
        "citations": extract_citations(text),
        "statutes": extract_statutes(text),
        "pos_counts": pos_counts,
        "sentences_preview": sentences[:10],
    }
