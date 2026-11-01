"""LDA topic modeling for Lemmata.

Builds the document-term matrix, fits LDA (Gibbs sampling via tomotopy or
scikit-learn variational Bayes), extracts topics ordered by corpus
prevalence, and computes coherence / fit metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gensim.corpora import Dictionary as GensimDictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy.sparse import csr_matrix, spmatrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from lemmata.config import (
    GIBBS_ALPHA_SUM,
    GIBBS_BETA,
    GIBBS_BURN_IN,
    GIBBS_CHECK_EVERY,
    GIBBS_CONVERGENCE_TOL,
    GIBBS_CONVERGENCE_WINDOW,
    GIBBS_ITERATIONS,
    GIBBS_OPTIMIZE_INTERVAL,
    LDA_EVALUATE_EVERY,
    LDA_LEARNING_METHOD,
    LDA_MAX_ITER,
    LDA_METHOD_DEFAULT,
    LDA_METHODS,
    LDA_PERP_TOL,
    NGRAM_RANGE,
    NUM_TOPICS_DEFAULT,
    RANDOM_SEED,
    VECTORIZER_TYPE,
    WORDS_PER_TOPIC_DEFAULT,
    get_df_auto,
)

logger = logging.getLogger(__name__)


# ── Document-Term Matrix ──────────────────────────────────────────────────────


def build_dtm(
    processed_texts: list[str],
    min_df: int | float | None = None,
    max_df: float | None = None,
) -> tuple[spmatrix, CountVectorizer, dict[str, Any]]:
    """Build a document-term matrix from pre-processed texts.

    Parameters
    ----------
    processed_texts:
        Space-joined lemma strings, one per document.
    min_df:
        Minimum document frequency.  *None* → auto from corpus size.
    max_df:
        Maximum document frequency.  *None* → auto from corpus size.

    Returns
    -------
    tuple[spmatrix, CountVectorizer, dict]
        - Sparse DTM (docs × terms).
        - Fitted vectorizer.
        - DTM info dict with transparency fields (decision 103).
    """
    n_docs = len(processed_texts)

    if min_df is None or max_df is None:
        auto_min, auto_max = get_df_auto(n_docs)
        if min_df is None:
            min_df = auto_min
        if max_df is None:
            max_df = auto_max

    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=NGRAM_RANGE,
    )
    dtm = vectorizer.fit_transform(processed_texts)

    # Build an unfiltered vectorizer to measure removed terms.
    vec_all = CountVectorizer(ngram_range=NGRAM_RANGE)
    vec_all.fit(processed_texts)
    vocab_all = len(vec_all.vocabulary_)
    vocab_kept = len(vectorizer.vocabulary_)

    dtm_info: dict[str, Any] = {
        "n_docs": n_docs,
        "vocabulary_total": vocab_all,
        "vocabulary_kept": vocab_kept,
        "terms_removed": vocab_all - vocab_kept,
        "dtm_shape": dtm.shape,
        "min_df": min_df,
        "max_df": max_df,
        "vectorizer_type": VECTORIZER_TYPE,
    }

    logger.info(
        "DTM built: %d docs × %d terms (removed %d)",
        n_docs,
        vocab_kept,
        vocab_all - vocab_kept,
    )

    return dtm, vectorizer, dtm_info


# ── LDA ───────────────────────────────────────────────────────────────────────


class GibbsLDA:
    """Collapsed Gibbs sampling LDA (tomotopy) for the Lemmata pipeline.

    Exposes the parts of scikit-learn's ``LatentDirichletAllocation`` that
    the rest of Lemmata reads (``components_``, ``n_components``,
    ``n_iter_``, ``transform``, ``perplexity``, ``score``), so visualisation
    and export code work with either method.

    Parameters
    ----------
    n_components:
        Number of topics.
    random_state:
        Sampler seed. With one worker, the same seed gives the same model.
    max_iter:
        Number of Gibbs iterations.
    alpha_sum:
        Sum of the symmetric document-topic prior (MALLET ``--alpha``).
    beta:
        Topic-word prior (MALLET ``--beta``).
    burn_in:
        Iterations before the first hyperparameter optimisation.
    optimize_interval:
        Iterations between hyperparameter optimisations (0 disables it).
        tomotopy optimises the document-topic prior only.
    check_every:
        Iterations between log-likelihood records (for convergence).
    """

    def __init__(
        self,
        n_components: int,
        random_state: int = RANDOM_SEED,
        max_iter: int = GIBBS_ITERATIONS,
        alpha_sum: float = GIBBS_ALPHA_SUM,
        beta: float = GIBBS_BETA,
        burn_in: int = GIBBS_BURN_IN,
        optimize_interval: int = GIBBS_OPTIMIZE_INTERVAL,
        check_every: int = GIBBS_CHECK_EVERY,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.alpha_sum = alpha_sum
        self.beta = beta
        self.burn_in = burn_in
        self.optimize_interval = optimize_interval
        self.check_every = check_every

        self.components_: np.ndarray | None = None
        self.alpha_: np.ndarray | None = None
        self.n_iter_: int = 0
        self.ll_trace_: list[float] = []
        self._mdl: Any = None
        self._doc_topic: np.ndarray | None = None
        self._train_signature: tuple[tuple[int, int], int] | None = None
        self._feature_names: np.ndarray | None = None

    def fit_transform(self, dtm: spmatrix, feature_names: np.ndarray) -> np.ndarray:
        """Fit on a document-term matrix and return the doc-topic matrix."""
        try:
            import tomotopy as tp
        except ImportError as exc:  # pragma: no cover - dependency check
            raise ImportError(
                "Gibbs sampling needs the 'tomotopy' package. Install it or "
                "choose the variational method."
            ) from exc

        dtm = csr_matrix(dtm)
        n_docs, n_terms = dtm.shape
        feature_names = np.asarray(feature_names)

        mdl = tp.LDAModel(
            k=self.n_components,
            alpha=self.alpha_sum / self.n_components,
            eta=self.beta,
            seed=self.random_state,
        )
        mdl.burn_in = self.burn_in
        mdl.optim_interval = self.optimize_interval

        # tomotopy takes token lists; LDA is bag-of-words, so counts suffice.
        rows_in_model = self._add_rows(mdl, dtm, feature_names)
        if not rows_in_model:
            raise ValueError("All documents are empty after filtering; nothing to model.")

        self.ll_trace_ = []
        while mdl.global_step < self.max_iter:
            step = min(self.check_every, self.max_iter - mdl.global_step)
            mdl.train(step, workers=1, parallel=tp.ParallelScheme.NONE)
            self.ll_trace_.append(float(mdl.ll_per_word))
        self.n_iter_ = int(mdl.global_step)
        self.alpha_ = np.asarray(mdl.alpha, dtype=float)

        # Topic-word pseudo-counts aligned to the vectorizer vocabulary.
        column = {word: i for i, word in enumerate(feature_names)}
        cols = np.array([column[word] for word in mdl.used_vocabs])
        counts = np.asarray(mdl.get_count_by_topics(), dtype=float)
        components = np.zeros((self.n_components, n_terms))
        for t in range(self.n_components):
            components[t, cols] = np.asarray(mdl.get_topic_word_dist(t)) * max(counts[t], 1.0)
        self.components_ = components

        # Empty documents get the prior mean. Iterate over mdl.docs:
        # tomotopy 0.14.0 raises IndexError for mdl.docs[j] with j > 0.
        doc_topic = np.tile(self.alpha_ / self.alpha_.sum(), (n_docs, 1))
        for row, doc in zip(rows_in_model, mdl.docs):
            doc_topic[row] = doc.get_topic_dist()

        self._mdl = mdl
        self._doc_topic = doc_topic
        self._train_signature = (dtm.shape, dtm.nnz)
        self._feature_names = feature_names
        return doc_topic.copy()

    def transform(self, dtm: spmatrix) -> np.ndarray:
        """Doc-topic matrix; the stored one for the training matrix."""
        import tomotopy as tp

        dtm = csr_matrix(dtm)
        if (dtm.shape, dtm.nnz) == self._train_signature:
            return self._doc_topic.copy()

        out = np.tile(self.alpha_ / self.alpha_.sum(), (dtm.shape[0], 1))
        docs, rows = [], []
        for row in range(dtm.shape[0]):
            words = self._row_words(dtm, row, self._feature_names)
            if words:
                docs.append(self._mdl.make_doc(words))
                rows.append(row)
        if docs:
            dists, _ = self._mdl.infer(
                docs, iterations=100, workers=1, parallel=tp.ParallelScheme.NONE
            )
            for row, dist in zip(rows, dists):
                out[row] = dist
        return out

    def perplexity(self, dtm: spmatrix | None = None) -> float:
        """Training perplexity, exp(-log-likelihood per word)."""
        return float(self._mdl.perplexity)

    def score(self, dtm: spmatrix | None = None) -> float:
        """Training log-likelihood (per-word value × number of words)."""
        return float(self._mdl.ll_per_word * self._mdl.num_words)

    @staticmethod
    def _row_words(dtm: csr_matrix, row: int, feature_names: np.ndarray) -> list[str]:
        start, end = dtm.indptr[row], dtm.indptr[row + 1]
        idx, counts = dtm.indices[start:end], dtm.data[start:end].astype(int)
        return np.repeat(feature_names[idx], counts).tolist()

    def _add_rows(self, mdl: Any, dtm: csr_matrix, feature_names: np.ndarray) -> list[int]:
        rows = []
        for row in range(dtm.shape[0]):
            words = self._row_words(dtm, row, feature_names)
            if words:
                mdl.add_doc(words)
                rows.append(row)
        return rows


def _fit_model(
    dtm: spmatrix,
    feature_names: np.ndarray,
    n_topics: int,
    method: str,
    random_seed: int,
    max_iter: int,
) -> tuple[Any, np.ndarray]:
    """Fit LDA with the chosen method; return the model and raw doc-topic matrix."""
    if method == "gibbs":
        model = GibbsLDA(n_components=n_topics, random_state=random_seed, max_iter=max_iter)
        return model, model.fit_transform(dtm, feature_names)
    if method == "variational":
        model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_seed,
            learning_method=LDA_LEARNING_METHOD,
            max_iter=max_iter,
            evaluate_every=LDA_EVALUATE_EVERY,
            perp_tol=LDA_PERP_TOL,
        )
        return model, model.fit_transform(dtm)
    raise ValueError(f"Unknown LDA method {method!r}; use one of {sorted(LDA_METHODS)}.")


def _gibbs_converged(
    ll_trace: list[float],
    window: int = GIBBS_CONVERGENCE_WINDOW,
    tol: float = GIBBS_CONVERGENCE_TOL,
) -> bool:
    """True if mean log-likelihood per word has stopped changing."""
    if len(ll_trace) < 2 * window:
        return False
    recent = float(np.mean(ll_trace[-window:]))
    before = float(np.mean(ll_trace[-2 * window:-window]))
    return abs(recent - before) <= tol * abs(before)


def _check_convergence(model: Any, method: str, max_iter: int) -> tuple[bool, str]:
    """Return (converged, warning text) for decision 124, per method."""
    if method == "gibbs":
        if _gibbs_converged(model.ll_trace_):
            return True, ""
        span = 2 * GIBBS_CONVERGENCE_WINDOW * GIBBS_CHECK_EVERY
        return False, (
            f"Log-likelihood was still changing after {max_iter} Gibbs "
            f"iterations (by more than {GIBBS_CONVERGENCE_TOL:.1%} over the "
            f"last {span} iterations). Consider increasing iterations in "
            "Advanced settings."
        )
    if model.n_iter_ < max_iter:
        return True, ""
    return False, (
        f"Model did not converge within {max_iter} iterations (perplexity "
        f"still improved by more than {LDA_PERP_TOL} between checks). "
        "Consider increasing max iterations in Advanced settings."
    )


def _iterations_run(model: Any, method: str, max_iter: int) -> int:
    """Iterations actually run.

    scikit-learn leaves its loop before counting the iteration whose
    perplexity check stopped the fit, so ``n_iter_`` is one short whenever
    a variational fit stops early.
    """
    if method == "variational" and model.n_iter_ < max_iter:
        return int(model.n_iter_) + 1
    return int(model.n_iter_)


def _hyperparameters(model: Any, method: str) -> dict[str, Any]:
    """Priors and inference settings actually used, for the export."""
    if method == "gibbs":
        return {
            "alpha_sum_initial": model.alpha_sum,
            "alpha_final": [float(a) for a in model.alpha_],
            "beta": model.beta,
            "burn_in": model.burn_in,
            "optimize_interval": model.optimize_interval,
            "workers": 1,
        }
    return {
        "doc_topic_prior": float(model.doc_topic_prior_),
        "topic_word_prior": float(model.topic_word_prior_),
        "learning_method": LDA_LEARNING_METHOD,
        "evaluate_every": LDA_EVALUATE_EVERY,
        "perp_tol": LDA_PERP_TOL,
    }


def run_lda(
    dtm: spmatrix,
    vectorizer: CountVectorizer,
    n_topics: int = NUM_TOPICS_DEFAULT,
    n_words: int = WORDS_PER_TOPIC_DEFAULT,
    random_seed: int = RANDOM_SEED,
    max_iter: int | None = None,
    method: str = LDA_METHOD_DEFAULT,
) -> tuple[Any, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
    """Fit LDA and return topics ordered by corpus prevalence.

    Parameters
    ----------
    dtm:
        Sparse document-term matrix from :func:`build_dtm`.
    vectorizer:
        Fitted ``CountVectorizer``.
    n_topics:
        Number of topics to extract.
    n_words:
        Words per topic for the summary.
    random_seed:
        Deterministic seed (decision 121).
    max_iter:
        Iterations (Gibbs) or maximum EM iterations (variational).
        *None* → the method's default.
    method:
        ``"gibbs"`` or ``"variational"`` (see ``config.LDA_METHODS``).

    Returns
    -------
    tuple[model, ndarray, list[dict], dict]
        - Fitted model (:class:`GibbsLDA` or ``LatentDirichletAllocation``).
        - Document-topic matrix (docs × topics), **reordered** by prevalence.
        - Topic summaries (reordered).
        - Model info dict (perplexity, log-likelihood, convergence warning).
    """
    if max_iter is None:
        max_iter = GIBBS_ITERATIONS if method == "gibbs" else LDA_MAX_ITER

    feature_names = vectorizer.get_feature_names_out()
    model, doc_topic_raw = _fit_model(
        dtm, feature_names, n_topics, method, random_seed, max_iter
    )  # (n_docs, n_topics)

    # ── Convergence check (decision 124) ──────────────────────────────────
    converged, convergence_warning = _check_convergence(model, method, max_iter)
    if convergence_warning:
        logger.warning(convergence_warning)

    # ── Metrics ───────────────────────────────────────────────────────────
    perplexity = model.perplexity(dtm)
    log_likelihood = model.score(dtm)

    # ── Topic summaries (unordered) ───────────────────────────────────────
    summaries_raw = _extract_topic_summaries(model, feature_names, n_words)

    # ── Reorder by corpus prevalence (decision 122) ───────────────────────
    avg_weights = doc_topic_raw.mean(axis=0)  # (n_topics,)
    prevalence_order = np.argsort(avg_weights)[::-1]  # descending

    doc_topic = doc_topic_raw[:, prevalence_order]
    summaries = [summaries_raw[i] for i in prevalence_order]

    # Relabel topics 1..N in prevalence order.
    for rank, summary in enumerate(summaries, start=1):
        top_words = ", ".join(summary["words"][:3])
        summary["topic_id"] = rank
        summary["label"] = f"Topic {rank} ({top_words})"
        summary["avg_weight"] = float(avg_weights[prevalence_order[rank - 1]])

    # ── Dominant topic per document (decision 114) ────────────────────────
    dominant_topics = np.argmax(doc_topic, axis=1) + 1  # 1-indexed

    model_info: dict[str, Any] = {
        "method": method,
        "n_topics": n_topics,
        "n_words": n_words,
        "random_seed": random_seed,
        "max_iter": max_iter,
        "iterations_used": _iterations_run(model, method, max_iter),
        "converged": converged,
        "convergence_warning": convergence_warning,
        "perplexity": float(perplexity),
        "log_likelihood": float(log_likelihood),
        "hyperparameters": _hyperparameters(model, method),
        "dominant_topics": dominant_topics.tolist(),
        "prevalence_order": prevalence_order.tolist(),
    }

    return model, doc_topic, summaries, model_info


# ── Coherence ─────────────────────────────────────────────────────────────────


def compute_coherence(
    topics: list[dict[str, Any]],
    processed_texts: list[str],
) -> dict[str, Any]:
    """Compute C_v coherence using Gensim (decision 18).

    Parameters
    ----------
    topics:
        Topic summaries from :func:`run_lda`, each with a ``"words"`` list.
    processed_texts:
        Space-joined lemma strings (same as fed to :func:`build_dtm`).

    Returns
    -------
    dict
        ``{"c_v": float, "per_topic": list[float]}``.
    """
    tokenized = [text.split() for text in processed_texts]
    dictionary = GensimDictionary(tokenized)

    topic_word_lists = [t["words"] for t in topics]

    cm = CoherenceModel(
        topics=topic_word_lists,
        texts=tokenized,
        dictionary=dictionary,
        coherence="c_v",
    )
    c_v = cm.get_coherence()
    per_topic = cm.get_coherence_per_topic()

    return {
        "c_v": float(c_v),
        "per_topic": [float(v) for v in per_topic],
    }


# ── Topic Summary (standalone) ────────────────────────────────────────────────


def get_topic_summary(
    model: Any,
    vectorizer: CountVectorizer,
    n_words: int = WORDS_PER_TOPIC_DEFAULT,
) -> list[dict[str, Any]]:
    """Extract topic words and weights without reordering.

    Useful for inspection outside the main pipeline.

    Parameters
    ----------
    model:
        Fitted LDA model.
    vectorizer:
        Fitted ``CountVectorizer``.
    n_words:
        Number of top words per topic.

    Returns
    -------
    list[dict]
        One dict per topic with ``words``, ``weights``, ``topic_id``.
    """
    feature_names = vectorizer.get_feature_names_out()
    return _extract_topic_summaries(model, feature_names, n_words)


# ── Sweep (Future — decision 6) ──────────────────────────────────────────────


def sweep_coherence(
    processed_texts: list[str],
    dtm: spmatrix,
    vectorizer: CountVectorizer,
    topic_range: tuple[int, int] = (2, 15),
    random_seed: int = RANDOM_SEED,
    max_iter: int | None = None,
    n_words: int = WORDS_PER_TOPIC_DEFAULT,
    progress_callback: Any | None = None,
    method: str = LDA_METHOD_DEFAULT,
) -> dict[str, Any]:
    """Run LDA across a range of topic counts and return coherence scores.

    Implements decision 6 — optimal topic finder.

    Parameters
    ----------
    processed_texts:
        Space-joined lemma strings.
    dtm:
        Sparse document-term matrix.
    vectorizer:
        Fitted ``CountVectorizer``.
    topic_range:
        (min_topics, max_topics) inclusive.
    random_seed:
        Deterministic seed.
    max_iter:
        Iterations per fit. *None* → the method's default.
    n_words:
        Words per topic for coherence computation.
    progress_callback:
        Optional callable ``(current_step, total_steps) -> None``
        for UI progress updates.
    method:
        ``"gibbs"`` or ``"variational"``, as in :func:`run_lda`.

    Returns
    -------
    dict
        ``{"k_values": list[int], "coherence_scores": list[float],
        "best_k": int}``
    """
    if max_iter is None:
        max_iter = GIBBS_ITERATIONS if method == "gibbs" else LDA_MAX_ITER

    k_min, k_max = topic_range
    k_values: list[int] = list(range(k_min, k_max + 1))
    coherence_scores: list[float] = []
    total = len(k_values)

    feature_names = vectorizer.get_feature_names_out()

    # Pre-tokenize texts once for Gensim coherence.
    tokenized = [text.split() for text in processed_texts]
    dictionary = GensimDictionary(tokenized)

    for i, k in enumerate(k_values):
        if progress_callback is not None:
            progress_callback(i, total)

        # Fit LDA with same parameters as run_lda().
        model, _ = _fit_model(dtm, feature_names, k, method, random_seed, max_iter)

        # Extract topic word lists.
        topic_word_lists: list[list[str]] = []
        for component in model.components_:
            top_indices = component.argsort()[::-1][:n_words]
            words = [str(feature_names[idx]) for idx in top_indices]
            topic_word_lists.append(words)

        # Compute C_v coherence.
        cm = CoherenceModel(
            topics=topic_word_lists,
            texts=tokenized,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_scores.append(float(cm.get_coherence()))

        logger.info("Sweep k=%d: C_v=%.4f", k, coherence_scores[-1])

    # Final progress tick.
    if progress_callback is not None:
        progress_callback(total, total)

    best_idx = int(np.argmax(coherence_scores))
    best_k = k_values[best_idx]

    return {
        "k_values": k_values,
        "coherence_scores": coherence_scores,
        "best_k": best_k,
    }


# ── Internal Helpers ──────────────────────────────────────────────────────────


def _extract_topic_summaries(
    model: Any,
    feature_names: np.ndarray,
    n_words: int,
) -> list[dict[str, Any]]:
    """Extract top-N words and normalised weights per topic."""
    summaries: list[dict[str, Any]] = []

    for idx, component in enumerate(model.components_):
        top_indices = component.argsort()[::-1][:n_words]
        words = [str(feature_names[i]) for i in top_indices]
        raw_weights = component[top_indices]
        # Normalise weights to sum to 1 for interpretability.
        total = raw_weights.sum()
        weights = (raw_weights / total).tolist() if total > 0 else raw_weights.tolist()

        summaries.append(
            {
                "topic_id": idx + 1,
                "label": "",
                "words": words,
                "weights": weights,
                "avg_weight": 0.0,
            }
        )

    return summaries
