"""LDA topic modeling for Lemmata.

Builds the document-term matrix, fits LDA (Gibbs sampling via tomotopy or
scikit-learn variational Bayes), extracts topics ordered by corpus
prevalence, and computes coherence / fit metrics.
"""

from __future__ import annotations

import logging
import math
from numbers import Integral
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
    GIBBS_ITERATIONS,
    GIBBS_OPTIMIZE_INTERVAL,
    LDA_EVALUATE_EVERY,
    LDA_LEARNING_METHOD,
    LDA_MAX_ITER,
    LDA_METHOD_DEFAULT,
    LDA_METHODS,
    LDA_PERP_TOL,
    MAX_DF_DEFAULT,
    MIN_DF_DEFAULT,
    NGRAM_RANGE,
    NUM_TOPICS_DEFAULT,
    RANDOM_SEED,
    STABILITY_RUNS,
    STABILITY_TOP_WORDS,
    VECTORIZER_TYPE,
    WORDS_PER_TOPIC_DEFAULT,
)

logger = logging.getLogger(__name__)


# ── Document-Term Matrix ──────────────────────────────────────────────────────


def build_dtm(
    processed_texts: list[str],
    min_df: int | float | None = None,
    max_df: int | float | None = None,
) -> tuple[spmatrix, CountVectorizer, dict[str, Any]]:
    """Build a document-term matrix from pre-processed texts.

    Parameters
    ----------
    processed_texts:
        Space-joined lemma strings, one per document.
    min_df:
        Terms in fewer documents are removed: a share of the documents
        (float) or a number of documents (int), as in scikit-learn.
        *None* → ``MIN_DF_DEFAULT`` (0.5% of the documents).
    max_df:
        Terms in more documents are removed, in the same units.
        *None* → ``MAX_DF_DEFAULT`` (99% of the documents).

    Returns
    -------
    tuple[spmatrix, CountVectorizer, dict]
        - Sparse DTM (docs × terms).
        - Fitted vectorizer.
        - DTM info dict with transparency fields (decision 103), including
          what the thresholds removed (P047).
    """
    n_docs = len(processed_texts)
    if min_df is None:
        min_df = MIN_DF_DEFAULT
    if max_df is None:
        max_df = MAX_DF_DEFAULT

    vectorizer = CountVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=NGRAM_RANGE,
    )
    dtm = vectorizer.fit_transform(processed_texts)

    # Unfiltered counts, to report what the thresholds removed.
    vec_all = CountVectorizer(ngram_range=NGRAM_RANGE)
    counts_all = vec_all.fit_transform(processed_texts)
    vocab_all = len(vec_all.vocabulary_)
    vocab_kept = len(vectorizer.vocabulary_)

    # Document counts as scikit-learn computes them: a share is multiplied
    # by the number of documents, an integer is used as it is.
    low = min_df if isinstance(min_df, Integral) else min_df * n_docs
    high = max_df if isinstance(max_df, Integral) else max_df * n_docs
    doc_freq = np.asarray((counts_all > 0).sum(axis=0)).ravel()
    term_freq = np.asarray(counts_all.sum(axis=0)).ravel()
    rare = doc_freq < low
    common = doc_freq > high
    tokens_total = int(term_freq.sum())
    tokens_removed = int(term_freq[rare | common].sum())

    dtm_info: dict[str, Any] = {
        "n_docs": n_docs,
        "vocabulary_total": vocab_all,
        "vocabulary_kept": vocab_kept,
        "terms_removed": vocab_all - vocab_kept,
        "terms_removed_rare": int(rare.sum()),
        "terms_removed_common": int(common.sum()),
        "terms_removed_share": (vocab_all - vocab_kept) / vocab_all if vocab_all else 0.0,
        # Terms in this many documents or fewer, and in this many or more,
        # were removed.
        "rare_cutoff_docs": max(math.ceil(low) - 1, 0),
        "common_cutoff_docs": math.floor(high) + 1,
        "tokens_total": tokens_total,
        "tokens_removed": tokens_removed,
        "tokens_removed_share": tokens_removed / tokens_total if tokens_total else 0.0,
        "docs_emptied": int((np.asarray(dtm.sum(axis=1)).ravel() == 0).sum()),
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
        Iterations between log-likelihood records (for the change report).
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
        self.ll_iters_: list[int] = []
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

        self.ll_trace_, self.ll_iters_ = [], []
        while mdl.global_step < self.max_iter:
            step = min(self.check_every, self.max_iter - mdl.global_step)
            mdl.train(step, workers=1, parallel=tp.ParallelScheme.NONE)
            self.ll_trace_.append(float(mdl.ll_per_word))
            self.ll_iters_.append(int(mdl.global_step))
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


def _ll_change(
    ll_trace: list[float], ll_iters: list[int]
) -> tuple[float, int, int] | None:
    """Change in log-likelihood per word over the second half of a run.

    A straight line is fitted to all recorded values in the second half, so
    the result depends less on the sampler's fluctuation than a comparison
    of two single values would (it still reflects some of it). Returns the
    change as a percentage of the mean value (positive = the model fits the
    texts better) and the first and last iteration of that half, or *None*
    if too few values were recorded.
    """
    half = np.asarray(ll_trace[len(ll_trace) // 2:], dtype=float)
    if len(half) < 3:
        return None
    x = np.arange(len(half))
    slope = np.polyfit(x, half, 1)[0]
    pct = float(slope * (len(half) - 1) / abs(half.mean()) * 100)
    return pct, ll_iters[len(ll_iters) // 2], ll_iters[-1]


def _check_convergence(model: Any, method: str, max_iter: int) -> tuple[bool | None, str]:
    """Return (converged, warning text) for decision 124, per method.

    Gibbs runs get no verdict (None); see ``config.GIBBS_CHECK_EVERY``.
    """
    if method == "gibbs":
        return None, ""
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

    ll_change = _ll_change(model.ll_trace_, model.ll_iters_) if method == "gibbs" else None

    model_info: dict[str, Any] = {
        "method": method,
        "n_topics": n_topics,
        "n_words": n_words,
        "random_seed": random_seed,
        "max_iter": max_iter,
        "iterations_used": _iterations_run(model, method, max_iter),
        "converged": converged,
        "convergence_warning": convergence_warning,
        # Gibbs only: (% change, from iteration, to iteration), see _ll_change.
        "ll_change_second_half": ll_change,
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

    Implements decision 6. ``best_k`` is the count with the highest C_v; since
    P046 the app reports it as such and not as the optimal number of topics.

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


# ── Topic Stability ───────────────────────────────────────────────────────────


def _top_word_sets(model: Any, top_n: int) -> list[set[int]]:
    """Vocabulary indices of each topic's ``top_n`` most probable words."""
    return [set(np.argsort(c)[::-1][:top_n].tolist()) for c in model.components_]


def stability_scores(runs: list[list[set[Any]]], max_n: int = STABILITY_TOP_WORDS) -> dict[str, Any]:
    """Score the topics of ``runs[0]`` against all runs (Agrawal et al. 2018).

    ``runs`` holds one set of top words per topic for each run. For a topic of
    the first run, its overlap with a run is the largest number of words it
    shares with any topic of that run. The first run is counted too, as in the
    paper's worked example, so every topic scores at least ``1 / len(runs)``.
    The topic's score at size n is the share of runs with an overlap of at
    least n words; R_n is the median of these scores over the topics.

    Returns
    -------
    dict
        ``{"overlaps": list[list[int]], "per_topic": {n: list[float]},
        "r_n": {n: float}}`` for n = 1 … ``max_n``.
    """
    overlaps = [[max(len(topic & other) for other in run) for run in runs] for topic in runs[0]]
    per_topic = {
        n: [sum(o >= n for o in row) / len(runs) for row in overlaps] for n in range(1, max_n + 1)
    }
    r_n = {n: float(np.median(scores)) for n, scores in per_topic.items()}
    return {"overlaps": overlaps, "per_topic": per_topic, "r_n": r_n}


def topic_stability(
    dtm: spmatrix,
    vectorizer: CountVectorizer,
    n_topics: int,
    random_seed: int = RANDOM_SEED,
    max_iter: int | None = None,
    method: str = LDA_METHOD_DEFAULT,
    n_runs: int = STABILITY_RUNS,
    top_n: int = STABILITY_TOP_WORDS,
    reference_model: Any | None = None,
    progress_callback: Any | None = None,
) -> dict[str, Any]:
    """Refit the model with other seeds and measure how stable its topics are.

    The first run is the user's model (seed ``random_seed``); the others use
    seeds ``random_seed + 1`` … ``random_seed + n_runs - 1`` with the same
    settings (see ``config.STABILITY_RUNS``). Topics keep the model's own
    order; ``run_lda``'s ``prevalence_order`` maps them to the order shown in
    the app.

    Parameters
    ----------
    reference_model:
        The fitted model from :func:`run_lda`, to avoid fitting it again.
        *None* → fitted here with ``random_seed``.
    progress_callback:
        Optional callable ``(current_step, total_steps) -> None``.

    Returns
    -------
    dict
        :func:`stability_scores`' result plus ``seeds``, ``n_runs``,
        ``top_n`` and ``method``.
    """
    if max_iter is None:
        max_iter = GIBBS_ITERATIONS if method == "gibbs" else LDA_MAX_ITER

    feature_names = vectorizer.get_feature_names_out()
    seeds = [random_seed + i for i in range(n_runs)]
    if reference_model is None:
        reference_model, _ = _fit_model(dtm, feature_names, n_topics, method, seeds[0], max_iter)

    runs = [_top_word_sets(reference_model, top_n)]
    for i, seed in enumerate(seeds[1:], start=1):
        if progress_callback is not None:
            progress_callback(i, n_runs)
        model, _ = _fit_model(dtm, feature_names, n_topics, method, seed, max_iter)
        runs.append(_top_word_sets(model, top_n))
    if progress_callback is not None:
        progress_callback(n_runs, n_runs)

    result = stability_scores(runs, top_n)
    result.update({"seeds": seeds, "n_runs": n_runs, "top_n": top_n, "method": method})
    return result


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
