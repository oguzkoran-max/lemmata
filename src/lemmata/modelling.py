"""LDA topic modeling for Lemmata.

Builds the document-term matrix, fits scikit-learn LDA, extracts topics
ordered by corpus prevalence, and computes coherence / fit metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from gensim.corpora import Dictionary as GensimDictionary
from gensim.models.coherencemodel import CoherenceModel
from scipy.sparse import spmatrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from lemmata.config import (
    LDA_LEARNING_METHOD,
    LDA_MAX_ITER,
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


def run_lda(
    dtm: spmatrix,
    vectorizer: CountVectorizer,
    n_topics: int = NUM_TOPICS_DEFAULT,
    n_words: int = WORDS_PER_TOPIC_DEFAULT,
    random_seed: int = RANDOM_SEED,
    max_iter: int = LDA_MAX_ITER,
) -> tuple[LatentDirichletAllocation, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
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
        Maximum EM iterations.

    Returns
    -------
    tuple[LDA, ndarray, list[dict], dict]
        - Fitted LDA model.
        - Document-topic matrix (docs × topics), **reordered** by prevalence.
        - Topic summaries (reordered).
        - Model info dict (perplexity, log-likelihood, convergence warning).
    """
    model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_seed,
        learning_method=LDA_LEARNING_METHOD,
        max_iter=max_iter,
    )

    doc_topic_raw = model.fit_transform(dtm)  # (n_docs, n_topics)

    # ── Convergence check (decision 124) ──────────────────────────────────
    converged = model.n_iter_ < max_iter
    convergence_warning = ""
    if not converged:
        convergence_warning = (
            f"Model used all {max_iter} iterations without converging. "
            "Consider increasing max iterations in Advanced settings."
        )
        logger.warning(convergence_warning)

    # ── Metrics ───────────────────────────────────────────────────────────
    perplexity = model.perplexity(dtm)
    log_likelihood = model.score(dtm)

    # ── Topic summaries (unordered) ───────────────────────────────────────
    feature_names = vectorizer.get_feature_names_out()
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
        "n_topics": n_topics,
        "n_words": n_words,
        "random_seed": random_seed,
        "max_iter": max_iter,
        "iterations_used": model.n_iter_,
        "converged": converged,
        "convergence_warning": convergence_warning,
        "perplexity": float(perplexity),
        "log_likelihood": float(log_likelihood),
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
    model: LatentDirichletAllocation,
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
    max_iter: int = LDA_MAX_ITER,
    n_words: int = WORDS_PER_TOPIC_DEFAULT,
    progress_callback: Any | None = None,
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
        Maximum EM iterations per fit.
    n_words:
        Words per topic for coherence computation.
    progress_callback:
        Optional callable ``(current_step, total_steps) -> None``
        for UI progress updates.

    Returns
    -------
    dict
        ``{"k_values": list[int], "coherence_scores": list[float],
        "best_k": int}``
    """
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
        model = LatentDirichletAllocation(
            n_components=k,
            random_state=random_seed,
            learning_method=LDA_LEARNING_METHOD,
            max_iter=max_iter,
        )
        model.fit(dtm)

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
    model: LatentDirichletAllocation,
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
