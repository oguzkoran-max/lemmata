"""Central configuration for Lemmata.

All application constants, defaults, and thresholds live here.
No other module should hard-code values that belong in config.
"""

from __future__ import annotations

# ── Application ────────────────────────────────────────────────────────────────

APP_VERSION: str = "0.3.0"
APP_TITLE: str = "Lemmata — Multilingual Topic Modeling"
APP_PAGE_ICON: str = "📊"

# ── Supported Languages ───────────────────────────────────────────────────────
# Maps ISO 639-1 code → spaCy small-model name.
# No Portuguese, no Turkish (future work).

SUPPORTED_LANGUAGES: dict[str, str] = {
    "it": "it_core_news_sm",
    "en": "en_core_web_sm",
    "de": "de_core_news_sm",
    "fr": "fr_core_news_sm",
    "es": "es_core_news_sm",
}

# ── Chunking ──────────────────────────────────────────────────────────────────
# Word-count target; sentence boundaries respected (decision 32).

CHUNK_SIZE_MIN: int = 300
CHUNK_SIZE_MAX: int = 3000
CHUNK_SIZE_DEFAULT: int = 1000

# ── Topic Modeling ────────────────────────────────────────────────────────────
# Two inference methods, both deterministic for a given seed (P045).
# "gibbs" (default): collapsed Gibbs sampling via tomotopy, as in MALLET.
# "variational": scikit-learn batch variational Bayes, the only method
# before v0.3.0; kept so earlier results can be reproduced.

NUM_TOPICS_MIN: int = 2
NUM_TOPICS_MAX: int = 30
NUM_TOPICS_DEFAULT: int = 5

WORDS_PER_TOPIC_MIN: int = 5
WORDS_PER_TOPIC_MAX: int = 30
WORDS_PER_TOPIC_DEFAULT: int = 10

RANDOM_SEED: int = 42

LDA_METHODS: dict[str, str] = {
    "gibbs": "Gibbs sampling (MALLET-style)",
    "variational": "Variational Bayes (fast, scikit-learn)",
}
LDA_METHOD_DEFAULT: str = "gibbs"

# Variational: perplexity is checked every LDA_EVALUATE_EVERY iterations and
# fitting stops once it improves by less than LDA_PERP_TOL (scikit-learn's
# own criterion). Without the check the model always ran to max_iter.
LDA_LEARNING_METHOD: str = "batch"
LDA_MAX_ITER: int = 50
LDA_MAX_ITER_MIN: int = 10
LDA_MAX_ITER_MAX: int = 500
LDA_EVALUATE_EVERY: int = 10
LDA_PERP_TOL: float = 0.1

# Gibbs: defaults are MALLET 2.0.8's (alpha sum 5, beta 0.01, 1000
# iterations, burn-in 200) with hyperparameter optimisation every 10
# iterations, as in little-mallet-wrapper. They were fixed before any
# validation run (P045). One worker keeps runs reproducible.
GIBBS_ALPHA_SUM: float = 5.0
GIBBS_BETA: float = 0.01
GIBBS_ITERATIONS: int = 1000
GIBBS_ITERATIONS_MIN: int = 100
GIBBS_ITERATIONS_MAX: int = 5000
GIBBS_BURN_IN: int = 200
GIBBS_OPTIMIZE_INTERVAL: int = 10
GIBBS_CHECK_EVERY: int = 10
# Converged when the mean log-likelihood per word of the last
# GIBBS_CONVERGENCE_WINDOW checks differs from the window before by less
# than GIBBS_CONVERGENCE_TOL (relative).
GIBBS_CONVERGENCE_WINDOW: int = 5
GIBBS_CONVERGENCE_TOL: float = 0.001

# ── POS Filtering ─────────────────────────────────────────────────────────────
# Defaults (decisions 12, 95): chosen by the developer on the advice of a
# corpus linguist (Hakan Cangır), given before development.

DEFAULT_POS_TAGS: list[str] = ["NOUN", "PROPN", "ADJ"]

POS_PRESETS: dict[str, list[str]] = {
    "Content words": ["NOUN", "PROPN", "ADJ"],
    "Content + verbs": ["NOUN", "PROPN", "ADJ", "VERB"],
    "All open class": ["NOUN", "PROPN", "ADJ", "VERB", "ADV"],
    "Custom": [],
}

# ── File Upload ───────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS: list[str] = ["txt", "pdf", "odt", "docx", "epub"]
MAX_FILE_SIZE_MB: int = 50
MAX_TOTAL_SIZE_MB: int = 100

# ── Document-Term Matrix Auto Thresholds ──────────────────────────────────────
# min_df / max_df adjusted by corpus size (decision 25).
# Keys are upper bounds of document count; values are (min_df, max_df).

DF_THRESHOLDS: dict[int, tuple[int | float, float]] = {
    10: (1, 0.95),
    50: (2, 0.90),
    100: (3, 0.85),
    500: (5, 0.80),
    10_000: (10, 0.75),
}


def get_df_auto(n_docs: int) -> tuple[int | float, float]:
    """Return (min_df, max_df) for the given corpus size."""
    for upper, thresholds in DF_THRESHOLDS.items():
        if n_docs <= upper:
            return thresholds
    # Very large corpus: use the most aggressive filtering.
    return 10, 0.75


# ── Visual / Theme ────────────────────────────────────────────────────────────
# Decisions 66, 131-134, 144.

COLOR_PRIMARY: str = "#0F6E56"
COLOR_PALETTE: str = "tableau10"
COLOR_HEATMAP: str = "viridis"

BG_MAIN: str = "#F8F9FA"
BG_SIDEBAR: str = "#F0F2F6"

WORDCLOUD_WIDTH: int = 800
WORDCLOUD_HEIGHT: int = 400
CHART_DPI: int = 300

# ── Export ─────────────────────────────────────────────────────────────────────
# Decision 130.

ZIP_NAME_TEMPLATE: str = "lemmata_{lang}_{n}topics_{date}_{time}.zip"

# ── Future-Ready Constants ────────────────────────────────────────────────────
# Locked to current values; change when features are implemented.

NGRAM_RANGE: tuple[int, int] = (1, 1)
MODEL_SIZE: str = "sm"
VECTORIZER_TYPE: str = "count"

# ── Text Hygiene ──────────────────────────────────────────────────────────────

ENCODING_FALLBACK_CHAIN: list[str] = ["utf-8", "latin-1"]
LOW_TOKEN_RATIO_THRESHOLD: float = 0.10

# ── Logging ───────────────────────────────────────────────────────────────────

LOG_LEVEL: str = "INFO"
MIN_UNIQUE_LEMMAS_WARNING: int = 50
