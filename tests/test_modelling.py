"""Tests for lemmata.modelling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse

from lemmata.modelling import (
    build_dtm,
    compute_coherence,
    run_lda,
    sweep_coherence,
)


# ── build_dtm ─────────────────────────────────────────────────────────────────


class TestBuildDtm:
    """Verify document-term matrix construction."""

    def test_returns_sparse_matrix(self, processed_texts):
        dtm, vec, info = build_dtm(processed_texts)
        assert issparse(dtm)

    def test_returns_vectorizer(self, processed_texts):
        from sklearn.feature_extraction.text import CountVectorizer

        dtm, vec, info = build_dtm(processed_texts)
        assert isinstance(vec, CountVectorizer)

    def test_dtm_info_keys(self, processed_texts):
        _, _, info = build_dtm(processed_texts)
        required = {
            "n_docs", "vocabulary_total", "vocabulary_kept",
            "terms_removed", "dtm_shape", "min_df", "max_df",
            "vectorizer_type",
        }
        assert required.issubset(set(info.keys()))

    def test_dtm_shape_matches_docs(self, processed_texts):
        dtm, _, info = build_dtm(processed_texts)
        assert dtm.shape[0] == len(processed_texts)
        assert info["n_docs"] == len(processed_texts)
        assert info["dtm_shape"] == dtm.shape

    def test_custom_min_max_df(self, processed_texts):
        _, _, info = build_dtm(processed_texts, min_df=1, max_df=1.0)
        assert info["min_df"] == 1
        assert info["max_df"] == 1.0

    def test_vocabulary_kept_le_total(self, processed_texts):
        _, _, info = build_dtm(processed_texts)
        assert info["vocabulary_kept"] <= info["vocabulary_total"]


# ── run_lda ───────────────────────────────────────────────────────────────────


class TestRunLda:
    """Verify LDA fitting and topic extraction."""

    def test_return_types(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        model, doc_topic, topics, model_info = run_lda(
            dtm, vec, n_topics=2, n_words=5,
        )
        assert doc_topic.shape == (len(processed_texts), 2)
        assert isinstance(topics, list)
        assert len(topics) == 2
        assert isinstance(model_info, dict)

    def test_model_info_keys(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, _, model_info = run_lda(dtm, vec, n_topics=2)
        required = {
            "n_topics", "n_words", "random_seed", "max_iter",
            "iterations_used", "converged", "convergence_warning",
            "perplexity", "log_likelihood", "dominant_topics",
            "prevalence_order",
        }
        assert required.issubset(set(model_info.keys()))

    def test_topic_summary_structure(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=2, n_words=5)
        for t in topics:
            assert "topic_id" in t
            assert "label" in t
            assert "words" in t
            assert "weights" in t
            assert len(t["words"]) == 5
            assert len(t["weights"]) == 5

    def test_topics_ordered_by_prevalence(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=2, n_words=5)
        # avg_weight should be descending.
        weights = [t["avg_weight"] for t in topics]
        assert weights == sorted(weights, reverse=True)

    def test_topic_ids_sequential(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=2)
        ids = [t["topic_id"] for t in topics]
        assert ids == [1, 2]

    def test_dominant_topics_valid(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, _, model_info = run_lda(dtm, vec, n_topics=2)
        for dt in model_info["dominant_topics"]:
            assert 1 <= dt <= 2

    def test_doc_topic_rows_sum_to_one(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, doc_topic, _, _ = run_lda(dtm, vec, n_topics=2)
        row_sums = doc_topic.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

    def test_perplexity_is_numeric(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, _, model_info = run_lda(dtm, vec, n_topics=2)
        assert isinstance(model_info["perplexity"], float)
        assert isinstance(model_info["log_likelihood"], float)


class TestDeterminism:
    """CRITICAL: LDA must be deterministic with same seed (decision 50)."""

    def test_same_seed_identical_results(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)

        _, doc_topic_1, topics_1, _ = run_lda(
            dtm, vec, n_topics=2, n_words=5, random_seed=42,
        )
        _, doc_topic_2, topics_2, _ = run_lda(
            dtm, vec, n_topics=2, n_words=5, random_seed=42,
        )

        np.testing.assert_array_equal(doc_topic_1, doc_topic_2)

        for t1, t2 in zip(topics_1, topics_2):
            assert t1["words"] == t2["words"]
            np.testing.assert_allclose(t1["weights"], t2["weights"])

    def test_different_seed_different_results(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)

        _, doc_topic_1, _, _ = run_lda(
            dtm, vec, n_topics=2, random_seed=42,
        )
        _, doc_topic_2, _, _ = run_lda(
            dtm, vec, n_topics=2, random_seed=99,
        )

        # They should (very likely) differ.
        assert not np.array_equal(doc_topic_1, doc_topic_2)


# ── compute_coherence ─────────────────────────────────────────────────────────


class TestComputeCoherence:
    """Verify coherence scoring."""

    def test_returns_c_v(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=2, n_words=5)

        result = compute_coherence(topics, processed_texts)
        assert "c_v" in result
        assert isinstance(result["c_v"], float)

    def test_returns_per_topic(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=2, n_words=5)

        result = compute_coherence(topics, processed_texts)
        assert "per_topic" in result
        assert len(result["per_topic"]) == 2


# ── sweep_coherence ───────────────────────────────────────────────────────────


class TestSweepCoherence:
    """Verify sweep stub raises NotImplementedError."""

    def test_raises_not_implemented(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        with pytest.raises(NotImplementedError):
            sweep_coherence(processed_texts, dtm, vec)
