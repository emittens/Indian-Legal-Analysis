# Indian Legal Analysis

A Flask (Python 3.11) web application that performs NLP analysis on Indian
legal text — judgments, statute extracts, and case notes — using
[spaCy](https://spacy.io) and the [Blackstone](https://github.com/ICLRandD/Blackstone)
legal NLP model, augmented with regex patterns tuned for Indian citations
(AIR, SCC, SCC OnLine, ILR) and statutes (IPC, CrPC, CPC, Evidence Act,
Constitution Articles).

## Features

- Web UI to paste and analyze legal text.
- JSON API (`POST /api/analyze`) for programmatic use.
- Offline trend-analysis pipeline over 75 years (1950–2025) of Supreme Court judgments.
- Extraction of:
  - Named entities (via Blackstone: `CASE`, `COURT`, `JUDGE`, `PROVISION`, etc.; or general NER as fallback).
  - Indian case citations (AIR, SCC, SCC OnLine, ILR).
  - Statutory references and Constitution Articles.
  - Sentence segmentation and POS summary.

## Project layout

```
Indian_Legal_Analysis/
├── app/
│   ├── __init__.py                    # Flask application factory
│   ├── nlp.py                         # spaCy + Blackstone loading & analysis
│   ├── routes.py                      # UI and /api routes
│   ├── static/style.css
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   └── result.html
│   └── data/
│       ├── convert_pdfs_to_text.py    # Step 1: PDFs → per-year combined_txt_file.txt
│       ├── parse_combined_txt.py      # Step 2: combined text → judgments.parquet + citations.parquet
│       ├── supreme_court_judgments/
│       │   └── script.py             # PDF filename sanitizer/renamer (utility)
│       ├── supreme_court_judgments_text/
│       │   └── <1950..2025>/combined_txt_file.txt   # 76 yearly text corpora
│       ├── processed/
│       │   ├── judgments.parquet       # 26,688 rows — one per judgment
│       │   ├── citations.parquet      # 19,211 rows — citator long table
│       │   ├── judgments_preview.csv   # 50-row CSV preview
│       │   ├── trends/                # 27 aggregate parquet tables
│       │   └── charts/                # 26 PNG charts + dashboard.html
│       └── trends/
│           ├── common.py              # Shared loaders, paths, chart styling
│           ├── eda_overview.py        # Phase 1: coverage & volume EDA
│           ├── categorize.py          # Phase 1.5: rule-based case classifier
│           ├── volume_trends.py       # Phase 2a
│           ├── length_trends.py       # Phase 2b
│           ├── bench_trends.py        # Phase 2c
│           ├── judge_trends.py        # Phase 3
│           ├── statute_trends.py      # Phase 4a
│           ├── section_hotspots.py    # Phase 4b
│           ├── ipc_crpc_evolution.py  # Phase 4.5
│           ├── citation_network.py    # Phase 5
│           ├── keyword_trends.py      # Phase 6
│           ├── build_report.py        # Phase 7: rollup summary + dashboard
│           ├── amendments.yaml        # Landmark amendment overlays
│           └── bns_bnss_mapping.yaml  # IPC/CrPC → BNS/BNSS successor map
├── run.py                             # dev server entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

## Setup (conda — Python 3.11)

```powershell
# 1. Activate the conda environment with Python 3.11
conda activate py311

# 2. Install all dependencies (web app + ingestion + trend pipeline)
pip install -r requirements.txt

# 3. Install the spaCy English model (required)
python -m spacy download en_core_web_sm

# 4. (Optional) Install the Blackstone legal NLP model
#    This may fail on newer spaCy — the app falls back to en_core_web_sm automatically
pip install https://blackstone-model.s3-eu-west-1.amazonaws.com/en_blackstone_proto-0.0.1.tar.gz

# 5. Configure environment
copy .env.example .env

# 6. Run
python run.py
```

Open http://127.0.0.1:5000.

### Notes on Blackstone + Python 3.11

Blackstone's published wheel was trained on an older spaCy. If the direct
install above fails on spaCy 3.7, the app will automatically fall back to
`en_core_web_sm` (general English NER) — the Indian-specific regex
extractors for citations and statutes will still run. To force a specific
model set `SPACY_MODEL` in `.env`.

## API

```bash
curl -X POST http://127.0.0.1:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "In AIR 1973 SC 1461 (Kesavananda Bharati), the Court examined Article 368 of the Constitution and Section 3 of the Evidence Act."}'
```

Response includes `entities`, `citations`, `statutes`, `pos_counts`,
`sentences_preview`, and pipeline metadata.

## Health check

```
GET /api/health
```

## Data ingestion pipeline

The corpus ingestion is a three-step process. Steps 1–2 convert raw PDFs
into analyst-friendly Parquet tables; the trend scripts (Step 3) aggregate
those into charts and summary tables.

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 0 (utility) | [script.py](app/data/supreme_court_judgments/script.py) | Raw PDFs with messy filenames | Sanitized PDF filenames |
| 1 | [convert_pdfs_to_text.py](app/data/convert_pdfs_to_text.py) | PDFs under `supreme_court_judgments/<year>/` | Per-year `combined_txt_file.txt` |
| 2 | [parse_combined_txt.py](app/data/parse_combined_txt.py) | Per-year `combined_txt_file.txt` | `judgments.parquet` (26,688 rows) + `citations.parquet` (19,211 rows) |
| 3 | [trends/](app/data/trends/) (12 scripts) | Parquet tables from Step 2 | 27 trend parquets + 26 PNGs + `dashboard.html` |

### Regenerate parquets from text

```powershell
python app\data\parse_combined_txt.py
# Or for specific years:
python app\data\parse_combined_txt.py --years 2024 2025
```

## Trend analysis pipeline

Offline analytical layer over the 75-year Supreme Court corpus
(26,688 judgments, 1950–2025). The trend scripts under
[app/data/trends/](app/data/trends/) produce 27 parquet tables and
27 charts covering volume, length, bench composition, judge authorship,
statute & section citations, IPC/CrPC evolution (with BNS/BNSS bridging),
citation network, and TF-IDF keywords.

### Regenerate everything

All dependencies are included in `requirements.txt`. Run each phase in order:

```powershell
python app\data\trends\eda_overview.py          # Phase 1
python app\data\trends\categorize.py            # Phase 1.5
python app\data\trends\volume_trends.py         # Phase 2a
python app\data\trends\length_trends.py         # Phase 2b
python app\data\trends\bench_trends.py          # Phase 2c
python app\data\trends\judge_trends.py          # Phase 3
python app\data\trends\statute_trends.py        # Phase 4a
python app\data\trends\section_hotspots.py      # Phase 4b
python app\data\trends\ipc_crpc_evolution.py    # Phase 4.5
python app\data\trends\citation_network.py      # Phase 5
python app\data\trends\keyword_trends.py        # Phase 6
python app\data\trends\build_report.py          # Phase 7 (rollup)
```

Outputs land under:

- [app/data/processed/trends/](app/data/processed/trends/) — parquet tables
- [app/data/processed/charts/](app/data/processed/charts/) — PNG charts + `dashboard.html`

### Key artefacts

| File | Contents |
|------|----------|
| [trends_summary.parquet](app/data/processed/trends/trends_summary.parquet) | Per-year headline metrics (volume, length, bench, marquee sections) |
| [ipc_crpc_summary.parquet](app/data/processed/trends/ipc_crpc_summary.parquet) | Per-section totals, OLS slope, constitution-bench share, BNS/BNSS successor |
| [citations_pagerank.parquet](app/data/processed/trends/citations_pagerank.parquet) | Top-50 most-influential judgments by corpus PageRank |
| [dashboard.html](app/data/processed/charts/dashboard.html) | Interactive Plotly rollup (volume, length, bench, marquee sections) |

### Data quality notes

- Indian Kanoon caps downloads at ~400 judgments/year from ~1970 onwards;
  absolute volumes reflect the scraped sample, not actual court output.
- Headnote coverage is dense pre-2000 and sparse after; keyword TF-IDF
  therefore limits to the 9,530 judgments with a substantive headnote.
- Bench-size values above 15 are parser errors (upstream "Bench:" line ate
  trailing text); [bench_trends.py](app/data/trends/bench_trends.py) clips
  to ≤15 (largest legitimate SC bench is 13: Kesavananda Bharati, 1973).
- IPC/CrPC disambiguation prefers the curated
  [bns_bnss_mapping.yaml](app/data/trends/bns_bnss_mapping.yaml); sections
  not in the mapping fall back to act-text presence (credited to both when
  both IPC and CrPC are named).

### Reference YAMLs

- [amendments.yaml](app/data/trends/amendments.yaml) — landmark amendments
  (498A insertion, CrPC 1973 commencement, Navtej, Nirbhaya, BNS/BNSS
  commencement, 44th Amendment) used as vertical-line overlays in
  ``ipc_marquee_timelines.png`` / ``crpc_marquee_timelines.png``.
- [bns_bnss_mapping.yaml](app/data/trends/bns_bnss_mapping.yaml) —
  curated ~60 IPC and ~40 CrPC section → BNS/BNSS successor mappings.
