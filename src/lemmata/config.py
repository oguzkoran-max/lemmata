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
# No number of topics is right for every corpus. The default follows the two
# most widely used tools, MALLET (--num-topics) and scikit-learn
# (LatentDirichletAllocation n_components), which both default to 10 (P046).
# It was 5 up to v0.3.0 (decision 13).
NUM_TOPICS_DEFAULT: int = 10

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
# Log-likelihood per word is recorded every GIBBS_CHECK_EVERY iterations.
# Gibbs runs get no converged/not-converged verdict: with MALLET's defaults
# the log-likelihood still rises slowly at 1000 iterations, so any fixed
# threshold either warns on nearly every run or misses the rise. The app
# reports how much it changed in the second half of the run instead.
GIBBS_CHECK_EVERY: int = 10

# Topic stability (P046), as defined by Agrawal, Fu and Menzies (2018,
# Information and Software Technology 98: 74–88). The model is fitted
# STABILITY_RUNS times with different seeds. A topic of the first run is
# stable at size n in a run when some topic of that run shares at least n of
# its top STABILITY_TOP_WORDS words; its score is the share of runs in which
# it is stable, and R_n is the median score over the topics. Agrawal et al.
# use 10 runs and at most 9 words (7 ± 2 items, from the cognitive science
# literature they cite); their worked example and first finding use n = 5.
# Rieger et al. (2024, PeerJ Computer Science 10: e2279) use 100 runs by
# default; 10 runs, as in Agrawal et al., take a tenth of the time. Runs
# differ only in the seed, as Rieger et al.'s replications differ only in
# the random initialisation; Agrawal et al. also shuffle the document order.
# The seeds follow the user's seed (seed, seed + 1, ...) and are fixed in
# advance, so that none is picked after seeing the results.
STABILITY_RUNS: int = 10
STABILITY_TOP_WORDS: int = 9
STABILITY_DISPLAY_N: int = 5

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

# ── Document-Term Matrix: vocabulary pruning ──────────────────────────────────
# Terms in fewer than 0.5% or in more than 99% of the documents are removed.
# Maier, Niekler, Wiedemann and Stoltenberg (2020, Computational Communication
# Research 2(2): 139–152, p. 141) report these as the disciplinary standard:
# fewer than 0.5% (at most 1%) of all documents for infrequent words (Grimmer
# 2010; Quinn et al. 2010; Denny and Spirling 2018) and more than 99% for
# frequent words (Hopkins and King 2010; Grimmer and Stewart 2013). Both are
# shares of the documents, so they follow the corpus size. Approved by the
# authors in P047; they replace the unsourced tiers of decision 25 (min_df 1
# to 10 documents, max_df 0.95 to 0.75 by corpus size). Passed to scikit-learn's
# CountVectorizer as proportions.
MIN_DF_DEFAULT: float = 0.005
MAX_DF_DEFAULT: float = 0.99


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
