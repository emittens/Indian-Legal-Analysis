"""Flask application factory for the Indian Legal Analysis app."""
from __future__ import annotations

import logging
import os

from flask import Flask
from dotenv import load_dotenv


def create_app() -> Flask:
    load_dotenv()

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-secret"),
        SPACY_MODEL=os.getenv("SPACY_MODEL", "en_blackstone_proto"),
        FALLBACK_SPACY_MODEL=os.getenv("FALLBACK_SPACY_MODEL", "en_core_web_sm"),
        MAX_TEXT_LENGTH=int(os.getenv("MAX_TEXT_LENGTH", "100000")),
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Pre-load the NLP pipeline once at startup
    from .nlp import get_nlp
    with app.app_context():
        get_nlp(app.config["SPACY_MODEL"], app.config["FALLBACK_SPACY_MODEL"])

    from .routes import main_bp, api_bp
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    # Warm up the lexical retriever in a background thread so the first
    # request to /retrieve doesn't pay the full BM25 load latency.
    import threading
    def _warmup():
        try:
            from .retrieval.lexical import get_lexical_retriever
            get_lexical_retriever()._ensure_loaded()
        except Exception as exc:
            logging.getLogger(__name__).warning("Retrieval warmup failed: %s", exc)

    threading.Thread(target=_warmup, daemon=True, name="retrieval-warmup").start()

    return app
