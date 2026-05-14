"""
retrieval/ner.py — legal entity extraction for retrieval signals
================================================================

Extracts structured legal entities from query text and judgment fields
for use as similarity signals in the hybrid retrieval pipeline.

Entity types
------------
statutes          Statute names  (Indian Penal Code, Motor Vehicles Act …)
articles          Constitutional Article numbers  (14, 21, 32, 226 …)
ipc_sections      IPC section numbers  (302, 498A, 420 …)
crpc_sections     CrPC section numbers  (482, 438, 144 …)
doctrines         Legal doctrines  (res judicata, promissory estoppel …)
procedural        Procedural writs & terms  (certiorari, habeas corpus …)
judges            Judge names mentioned in text
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Statute vocabulary — ordered longest-first so greedy matching picks the
# most-specific name (e.g., "Motor Vehicles Act" before "Vehicles Act").
# ---------------------------------------------------------------------------
_STATUTE_VOCAB: list[tuple[re.Pattern, str]] = [
    (re.compile(r"Indian\s+Penal\s+Code", re.I),             "Indian Penal Code"),
    (re.compile(r"\bIPC\b"),                                  "Indian Penal Code"),
    (re.compile(r"Code\s+of\s+Criminal\s+Procedure", re.I),  "Code of Criminal Procedure"),
    (re.compile(r"\bCr\.?P\.?C\.?\b"),                       "Code of Criminal Procedure"),
    (re.compile(r"Code\s+of\s+Civil\s+Procedure", re.I),     "Code of Civil Procedure"),
    (re.compile(r"\bCPC\b"),                                  "Code of Civil Procedure"),
    (re.compile(r"Constitution(?:\s+of\s+India)?", re.I),    "Constitution of India"),
    (re.compile(r"Income[-\s]?Tax\s+Act", re.I),             "Income Tax Act"),
    (re.compile(r"Customs\s+Act", re.I),                     "Customs Act"),
    (re.compile(r"Central\s+Excise\s+Act", re.I),            "Central Excise Act"),
    (re.compile(r"\bGST\b"),                                  "Goods and Services Tax"),
    (re.compile(r"Wealth[-\s]?Tax\s+Act", re.I),             "Wealth Tax Act"),
    (re.compile(r"Arbitration\s+and\s+Conciliation\s+Act", re.I), "Arbitration and Conciliation Act"),
    (re.compile(r"Arbitration\s+Act", re.I),                 "Arbitration Act"),
    (re.compile(r"Motor\s+Vehicles?\s+Act", re.I),           "Motor Vehicles Act"),
    (re.compile(r"Consumer\s+Protection\s+Act", re.I),       "Consumer Protection Act"),
    (re.compile(r"Representation\s+of\s+the\s+People\s+Act", re.I), "Representation of the People Act"),
    (re.compile(r"Industrial\s+Disputes\s+Act", re.I),       "Industrial Disputes Act"),
    (re.compile(r"Factories\s+Act", re.I),                   "Factories Act"),
    (re.compile(r"Minimum\s+Wages\s+Act", re.I),             "Minimum Wages Act"),
    (re.compile(r"Payment\s+of\s+Gratuity\s+Act", re.I),    "Payment of Gratuity Act"),
    (re.compile(r"Hindu\s+Marriage\s+Act", re.I),            "Hindu Marriage Act"),
    (re.compile(r"Special\s+Marriage\s+Act", re.I),          "Special Marriage Act"),
    (re.compile(r"Hindu\s+Succession\s+Act", re.I),          "Hindu Succession Act"),
    (re.compile(r"Contempt\s+of\s+Courts?\s+Act", re.I),     "Contempt of Courts Act"),
    (re.compile(r"Evidence\s+Act", re.I),                    "Evidence Act"),
    (re.compile(r"Transfer\s+of\s+Property\s+Act", re.I),   "Transfer of Property Act"),
    (re.compile(r"Specific\s+Relief\s+Act", re.I),           "Specific Relief Act"),
    (re.compile(r"Limitation\s+Act", re.I),                  "Limitation Act"),
    (re.compile(r"Contract\s+Act", re.I),                    "Contract Act"),
    (re.compile(r"Companies\s+Act", re.I),                   "Companies Act"),
    (re.compile(r"POCSO\s+Act|Protection\s+of\s+Children.*Sexual\s+Offences", re.I), "POCSO Act"),
    (re.compile(r"Prevention\s+of\s+Corruption\s+Act", re.I), "Prevention of Corruption Act"),
    (re.compile(r"Arms\s+Act", re.I),                        "Arms Act"),
    (re.compile(r"Narcotic\s+Drugs.*Psychotropic\s+Substances", re.I), "NDPS Act"),
    (re.compile(r"\bNDPS\s+Act\b", re.I),                   "NDPS Act"),
    (re.compile(r"Dowry\s+Prohibition\s+Act", re.I),         "Dowry Prohibition Act"),
]

# ---------------------------------------------------------------------------
# Section patterns
# ---------------------------------------------------------------------------
_IPC_SECTION_RE = re.compile(
    r"[Ss]ect?(?:ion)?\.?\s*(\d+[-A-Z]?)"
    r"(?:\s*(?:of\s+(?:the\s+)?)?"
    r"(?:Indian\s+Penal\s+Code|IPC))?",
    re.I,
)
_CRPC_SECTION_RE = re.compile(
    r"[Ss]ect?(?:ion)?\.?\s*(\d+[-A-Z]?)"
    r"(?:\s*(?:of\s+(?:the\s+)?)?"
    r"(?:Code\s+of\s+Criminal\s+Procedure|Cr\.?P\.?C\.?))?",
    re.I,
)
# Standalone IPC / CrPC section markers in sections_cited field
_IPC_TAG_RE   = re.compile(r"(?:IPC|Indian Penal Code)[^;,\n]*?(\d+[A-Z]?)", re.I)
_CRPC_TAG_RE  = re.compile(r"(?:CrPC|Cr\.?P\.?C\.|Criminal Procedure)[^;,\n]*?(\d+[A-Z]?)", re.I)

_ARTICLE_RE = re.compile(
    r"[Aa]rt(?:icle)?\.?\s*(\d+[A-Z]?(?:\s*\(\s*\d+\s*\))?)",
)

# ---------------------------------------------------------------------------
# Doctrine / procedural vocabulary (lowercase match)
# ---------------------------------------------------------------------------
_DOCTRINES = [
    "res judicata", "promissory estoppel", "natural justice",
    "ultra vires", "legitimate expectation", "audi alteram partem",
    "nemo judex in causa sua", "doctrine of necessity", "waiver",
    "estoppel", "constructive res judicata", "lis pendens",
    "in pari delicto", "ex parte", "obiter dicta", "ratio decidendi",
]
_PROCEDURAL = [
    "certiorari", "mandamus", "habeas corpus", "prohibition",
    "quo warranto", "suo motu", "interlocutory", "ex parte",
    "ad interim", "prima facie", "locus standi", "amicus curiae",
    "in camera", "remand", "acquittal", "conviction", "bail",
    "anticipatory bail", "stay", "injunction",
]
_JUDGE_TITLE_RE = re.compile(
    r"(?:Justice|J\.)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]*\.?){0,3})",
)


def extract_legal_entities(text: str) -> dict[str, list[str]]:
    """Extract structured legal entities from raw text.

    Returns a dict with keys: statutes, articles, ipc_sections,
    crpc_sections, doctrines, procedural, judges.
    All values are deduplicated lists of strings.
    """
    if not text:
        return _empty()

    text_lower = text.lower()

    statutes: list[str] = []
    seen_statutes: set[str] = set()
    for rx, name in _STATUTE_VOCAB:
        if rx.search(text) and name not in seen_statutes:
            statutes.append(name)
            seen_statutes.add(name)

    articles = sorted(
        {m.group(1).strip().replace(" ", "").replace("(", "").replace(")", "")
         for m in _ARTICLE_RE.finditer(text)},
        key=lambda x: int(re.search(r"\d+", x).group()) if re.search(r"\d+", x) else 999,
    )

    ipc_sections = sorted(
        {m.group(1).upper() for m in _IPC_TAG_RE.finditer(text)},
    )
    crpc_sections = sorted(
        {m.group(1).upper() for m in _CRPC_TAG_RE.finditer(text)},
    )

    doctrines = [d for d in _DOCTRINES if d in text_lower]
    procedural = [p for p in _PROCEDURAL if p in text_lower]

    judges = list(dict.fromkeys(
        m.group(1).strip() for m in _JUDGE_TITLE_RE.finditer(text)
    ))

    return {
        "statutes":     statutes,
        "articles":     articles,
        "ipc_sections": ipc_sections,
        "crpc_sections": crpc_sections,
        "doctrines":    doctrines,
        "procedural":   procedural,
        "judges":       judges,
    }


def extract_articles_from_sections(sections_array) -> set[str]:
    """Parse constitutional article numbers from a sections_cited numpy array."""
    arts: set[str] = set()
    if sections_array is None:
        return arts
    try:
        items = list(sections_array)
    except TypeError:
        return arts
    for item in items:
        for m in _ARTICLE_RE.finditer(str(item)):
            raw = m.group(1).strip()
            num = re.search(r"\d+", raw)
            if num:
                arts.add(num.group())
    return arts


def extract_statutes_from_text(act_text: str) -> set[str]:
    """Parse statute names from the act_text field."""
    if not act_text:
        return set()
    found: set[str] = set()
    for rx, name in _STATUTE_VOCAB:
        if rx.search(act_text):
            found.add(name)
    return found


def _empty() -> dict[str, list[str]]:
    return {k: [] for k in ("statutes", "articles", "ipc_sections",
                             "crpc_sections", "doctrines", "procedural", "judges")}


def entities_to_query_signal(ents: dict[str, Any]) -> dict[str, set[str]]:
    """Convert NER output to signal sets used for overlap scoring."""
    return {
        "statutes":  set(ents.get("statutes", [])),
        "articles":  set(ents.get("articles", [])),
        "sections":  set(ents.get("ipc_sections", [])) | set(ents.get("crpc_sections", [])),
        "doctrines": set(ents.get("doctrines", [])),
    }
