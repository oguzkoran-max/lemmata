"""Tests for lemmata.modelling."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import issparse
from sklearn.decomposition import LatentDirichletAllocation

from lemmata.config import (
    GIBBS_ALPHA_SUM,
    GIBBS_BETA,
    GIBBS_BURN_IN,
    GIBBS_ITERATIONS,
    GIBBS_OPTIMIZE_INTERVAL,
    LDA_EVALUATE_EVERY,
    LDA_MAX_ITER,
    MAX_DF_DEFAULT,
    MIN_DF_DEFAULT,
    RANDOM_SEED,
    STABILITY_RUNS,
    STABILITY_TOP_WORDS,
)
from lemmata.modelling import (
    GibbsLDA,
    _ll_change,
    build_dtm,
    compute_coherence,
    run_lda,
    stability_scores,
    sweep_coherence,
    topic_stability,
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


# ── vocabulary pruning (P047) ─────────────────────────────────────────────────


def _pruning_corpus() -> list[str]:
    """400 documents: 0.5% is 2 documents and 99% is 396."""
    docs = []
    for i in range(400):
        words = ["ovunque", "base" if i % 2 == 0 else "altro"]
        if i < 396:
            words.append("quasi")
        if i < 2:
            words.append("bordo")
        if i == 0:
            words.append("raro")
        docs.append(" ".join(words))
    return docs


class TestPruning:
    """Rare and common terms are removed by share of documents (P047)."""

    def test_defaults_are_the_standard_shares(self, processed_texts):
        _, _, info = build_dtm(processed_texts)
        assert info["min_df"] == MIN_DF_DEFAULT == 0.005
        assert info["max_df"] == MAX_DF_DEFAULT == 0.99

    def test_strictly_fewer_and_strictly_more(self):
        # raro: 1 of 400 documents (< 0.5%) and ovunque: 400 of 400
        # (> 99%) go; bordo (exactly 0.5%) and quasi (exactly 99%) stay.
        _, vec, info = build_dtm(_pruning_corpus())
        assert set(vec.get_feature_names_out()) == {"altro", "base", "bordo", "quasi"}
        assert info["terms_removed_rare"] == 1
        assert info["terms_removed_common"] == 1
        assert info["rare_cutoff_docs"] == 1
        assert info["common_cutoff_docs"] == 397

    def test_shares_removed(self):
        _, _, info = build_dtm(_pruning_corpus())
        assert info["terms_removed_share"] == pytest.approx(2 / 6)
        # 1199 word occurrences; raro once and ovunque 400 times.
        assert info["tokens_total"] == 1199
        assert info["tokens_removed"] == 401
        assert info["tokens_removed_share"] == pytest.approx(401 / 1199)

    def test_counts_add_up(self, processed_texts):
        _, _, info = build_dtm(processed_texts)
        assert (
            info["terms_removed_rare"] + info["terms_removed_common"]
            == info["terms_removed"]
            == info["vocabulary_total"] - info["vocabulary_kept"]
        )

    def test_document_counts_and_emptied_documents(self):
        # Integers are numbers of documents, as in scikit-learn.
        _, vec, info = build_dtm(["aa bb", "aa cc", "dd"], min_df=2, max_df=1.0)
        assert list(vec.get_feature_names_out()) == ["aa"]
        assert info["terms_removed_rare"] == 3
        assert info["rare_cutoff_docs"] == 1
        assert info["common_cutoff_docs"] > info["n_docs"]
        assert info["docs_emptied"] == 1

    def test_small_corpus_keeps_rare_terms(self):
        # 70 documents: 0.5% is 0.35 documents, so no term is rare.
        docs = [f"comune parola{i}" for i in range(70)]
        _, _, info = build_dtm(docs, max_df=1.0)
        assert info["terms_removed_rare"] == 0
        assert info["rare_cutoff_docs"] == 0


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

    def test_variational_same_seed_identical_results(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        runs = [
            run_lda(dtm, vec, n_topics=2, random_seed=42, method="variational")
            for _ in range(2)
        ]
        np.testing.assert_array_equal(runs[0][1], runs[1][1])


# ── Methods (P045) ────────────────────────────────────────────────────────────


@pytest.fixture()
def themed_texts() -> list[str]:
    """Synthetic corpus with four clear themes, large enough to converge."""
    import random

    rng = random.Random(0)
    themes = [
        ["mare", "nave", "isola", "vento", "porto"],
        ["denaro", "mercante", "fiorino", "prezzo", "vendita"],
        ["amore", "donna", "cuore", "bacio", "lettera"],
        ["chiesa", "frate", "peccato", "dio", "messa"],
    ]
    texts = []
    for i in range(120):
        main, other = themes[i % 4], themes[(i + 1) % 4]
        words = [rng.choice(main) for _ in range(40)]
        words += [rng.choice(other) for _ in range(10)]
        texts.append(" ".join(words))
    return texts


class TestMethods:
    """Gibbs sampling (default) and variational Bayes (P045)."""

    def test_default_method_is_gibbs(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        model, _, _, model_info = run_lda(dtm, vec, n_topics=2)
        assert isinstance(model, GibbsLDA)
        assert model_info["method"] == "gibbs"
        assert model_info["max_iter"] == GIBBS_ITERATIONS

    def test_variational_method(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        model, doc_topic, _, model_info = run_lda(
            dtm, vec, n_topics=2, method="variational",
        )
        assert isinstance(model, LatentDirichletAllocation)
        assert model_info["method"] == "variational"
        assert model_info["max_iter"] == LDA_MAX_ITER
        np.testing.assert_allclose(doc_topic.sum(axis=1), 1.0, atol=1e-5)

    def test_unknown_method_raises(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        with pytest.raises(ValueError, match="Unknown LDA method"):
            run_lda(dtm, vec, n_topics=2, method="nmf")

    def test_gibbs_defaults_are_mallet_defaults(self):
        # MALLET 2.0.8 train-topics: --alpha 5.0, --beta 0.01,
        # --num-iterations 1000, --optimize-burn-in 200;
        # little-mallet-wrapper: --optimize-interval 10.
        assert GIBBS_ALPHA_SUM == 5.0
        assert GIBBS_BETA == 0.01
        assert GIBBS_ITERATIONS == 1000
        assert GIBBS_BURN_IN == 200
        assert GIBBS_OPTIMIZE_INTERVAL == 10

    def test_gibbs_hyperparameters_recorded(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, _, model_info = run_lda(dtm, vec, n_topics=2, max_iter=100)
        hp = model_info["hyperparameters"]
        assert hp["alpha_sum_initial"] == GIBBS_ALPHA_SUM
        assert hp["beta"] == GIBBS_BETA
        assert hp["burn_in"] == GIBBS_BURN_IN
        assert len(hp["alpha_final"]) == 2

    def test_gibbs_empty_document_gets_prior(self):
        texts = ["mare nave mare", "mare nave nave", "zzz"]
        dtm, vec, _ = build_dtm(texts, min_df=2, max_df=1.0)
        assert dtm[2].nnz == 0
        model, doc_topic, _, _ = run_lda(dtm, vec, n_topics=2, max_iter=100)
        np.testing.assert_allclose(doc_topic.sum(axis=1), 1.0, atol=1e-5)
        prior = model.alpha_ / model.alpha_.sum()
        np.testing.assert_allclose(np.sort(doc_topic[2]), np.sort(prior), atol=1e-6)

    def test_gibbs_transform_returns_training_matrix(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        model, _, _, model_info = run_lda(dtm, vec, n_topics=4, max_iter=200)
        raw = model.transform(dtm)
        order = model_info["prevalence_order"]
        np.testing.assert_allclose(raw.sum(axis=1), 1.0, atol=1e-5)
        assert raw.shape == (len(themed_texts), 4)
        assert len(order) == 4

    def test_gibbs_recovers_planted_themes(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=4, n_words=5)
        found = {frozenset(t["words"]) for t in topics}
        assert frozenset(["mare", "nave", "isola", "vento", "porto"]) in found
        assert frozenset(["chiesa", "frate", "peccato", "dio", "messa"]) in found


# ── Convergence indicator (decision 124, P045) ────────────────────────────────


class TestConvergence:
    """The warning must say whether the model converged, not whether it
    reached the iteration limit (D2 check K3: the old warning fired on
    every run, converged or not)."""

    def test_variational_converged_model_does_not_warn(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, _, info = run_lda(
            dtm, vec, n_topics=4, method="variational", max_iter=500,
        )
        assert info["iterations_used"] < 500
        assert info["converged"] is True
        assert info["convergence_warning"] == ""

    def test_variational_early_stop_counts_last_iteration(self, themed_texts):
        # The fit can only stop at a perplexity check, i.e. after a multiple
        # of LDA_EVALUATE_EVERY iterations; scikit-learn's n_iter_ is one short.
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        model, _, _, info = run_lda(
            dtm, vec, n_topics=4, method="variational", max_iter=500,
        )
        assert info["iterations_used"] == model.n_iter_ + 1
        assert info["iterations_used"] % LDA_EVALUATE_EVERY == 0

    def test_variational_full_run_counts_all_iterations(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, _, info = run_lda(
            dtm, vec, n_topics=4, method="variational", max_iter=10,
        )
        assert info["iterations_used"] == 10

    def test_variational_unconverged_model_warns(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, _, info = run_lda(
            dtm, vec, n_topics=4, method="variational", max_iter=10,
        )
        assert info["converged"] is False
        assert "did not converge" in info["convergence_warning"]

    def test_variational_check_does_not_change_fit(self, themed_texts):
        # Perplexity checks are read-only: a fit stopped at the limit is the
        # same as before v0.3.0, so earlier results stay reproducible.
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        model, _, _, _ = run_lda(
            dtm, vec, n_topics=4, method="variational", max_iter=10,
        )
        old = LatentDirichletAllocation(
            n_components=4, random_state=RANDOM_SEED,
            learning_method="batch", max_iter=10,
        ).fit(dtm)
        np.testing.assert_allclose(model.components_, old.components_)

    def test_gibbs_reports_change_instead_of_verdict(self, themed_texts):
        # Gibbs runs get no converged/not-converged verdict and no warning;
        # the change over the second half of the run is reported instead.
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, _, info = run_lda(dtm, vec, n_topics=4)
        assert info["converged"] is None
        assert info["convergence_warning"] == ""
        pct, first, last = info["ll_change_second_half"]
        assert isinstance(pct, float)
        assert (first, last) == (510, GIBBS_ITERATIONS)

    def test_gibbs_short_run_reports_its_own_second_half(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, _, info = run_lda(dtm, vec, n_topics=4, max_iter=100)
        _, first, last = info["ll_change_second_half"]
        assert (first, last) == (60, 100)

    def test_variational_has_no_ll_change(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        _, _, _, info = run_lda(dtm, vec, n_topics=4, method="variational")
        assert info["ll_change_second_half"] is None

    def test_ll_change(self):
        iters = [10 * (i + 1) for i in range(10)]
        flat = [-2.0] * 10
        rising = [-3.0 + 0.01 * i for i in range(10)]
        pct, first, last = _ll_change(flat, iters)
        assert pct == pytest.approx(0.0, abs=1e-9)
        assert (first, last) == (60, 100)
        pct, _, _ = _ll_change(rising, iters)
        # Second half -2.95 ... -2.91: change 0.04 on a mean of 2.93.
        assert pct == pytest.approx(0.04 / 2.93 * 100)
        assert _ll_change(flat[:4], iters[:4]) is None  # 2 values in the half


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

    def test_uses_every_word_shown(self, processed_texts, monkeypatch):
        # gensim keeps only 20 words per topic unless told otherwise (P048).
        import lemmata.modelling as modelling

        seen = []
        real = modelling.CoherenceModel

        def spy(*args, **kwargs):
            cm = real(*args, **kwargs)
            seen.append(len(cm.topics[0]))
            return cm

        monkeypatch.setattr(modelling, "CoherenceModel", spy)
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        _, _, topics, _ = run_lda(dtm, vec, n_topics=2, n_words=25)
        compute_coherence(topics, processed_texts)
        sweep_coherence(processed_texts, dtm, vec, topic_range=(2, 2), n_words=25)
        assert seen == [25, 25]


# ── sweep_coherence ───────────────────────────────────────────────────────────


class TestSweepCoherence:
    """Verify coherence sweep (decision 6)."""

    def test_returns_required_keys(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        result = sweep_coherence(
            processed_texts, dtm, vec, topic_range=(2, 3),
        )
        assert "k_values" in result
        assert "coherence_scores" in result
        assert "best_k" in result

    def test_best_k_matches_max_coherence(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        result = sweep_coherence(
            processed_texts, dtm, vec, topic_range=(2, 3),
        )
        scores = result["coherence_scores"]
        k_vals = result["k_values"]
        expected_best = k_vals[scores.index(max(scores))]
        assert result["best_k"] == expected_best

    def test_k_values_match_range(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        result = sweep_coherence(
            processed_texts, dtm, vec, topic_range=(2, 3),
        )
        assert result["k_values"] == [2, 3]
        assert len(result["coherence_scores"]) == 2

    def test_progress_callback_called(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        calls = []
        result = sweep_coherence(
            processed_texts, dtm, vec, topic_range=(2, 3),
            progress_callback=lambda c, t: calls.append((c, t)),
        )
        # Should be called for each k plus final tick.
        assert len(calls) >= 2

    @pytest.mark.slow
    def test_full_range(self, processed_texts):
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        result = sweep_coherence(
            processed_texts, dtm, vec, topic_range=(2, 5),
        )
        assert result["k_values"] == [2, 3, 4, 5]
        assert len(result["coherence_scores"]) == 4
        assert result["best_k"] in [2, 3, 4, 5]


# ── topic stability (P046) ────────────────────────────────────────────────────


def _words(start: int, n: int = 9) -> set[int]:
    return set(range(start, start + n))


class TestStabilityScores:
    """The measure of Agrawal, Fu and Menzies (2018), checked on its own example."""

    def test_worked_example(self):
        # Four runs, four topics of nine words. As in the paper's example, at
        # n = 5 the first run's topics recur in 2, 3, 1 and 4 of the 4 runs.
        t0, t1, t2, t3 = _words(0), _words(100), _words(200), _words(300)
        f1, f2, f3 = _words(900), _words(1000), _words(1100)
        runs = [
            [t0, t1, t2, t3],
            [t0, t1, t3, f1],
            [t1, t3, f1, f2],
            [t3, f1, f2, f3],
        ]
        result = stability_scores(runs)
        assert result["per_topic"][5] == [0.5, 0.75, 0.25, 1.0]
        assert result["r_n"][5] == pytest.approx(0.625)

    def test_partial_overlap_counts_up_to_shared_words(self):
        reference = _words(0)
        five_shared = set(range(5)) | _words(500, 4)
        result = stability_scores([[reference], [five_shared]])
        assert result["overlaps"] == [[9, 5]]
        assert result["per_topic"][5] == [1.0]
        assert result["per_topic"][6] == [0.5]

    def test_scores_for_every_size(self):
        result = stability_scores([[_words(0)], [_words(0)]])
        assert sorted(result["r_n"]) == list(range(1, STABILITY_TOP_WORDS + 1))


class TestTopicStability:

    def test_defaults_follow_config(self):
        import inspect

        params = inspect.signature(topic_stability).parameters
        assert params["n_runs"].default == STABILITY_RUNS == 10
        assert params["top_n"].default == STABILITY_TOP_WORDS == 9

    def test_runs_seeds_and_shapes(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        result = topic_stability(dtm, vec, n_topics=4, random_seed=42, n_runs=3)
        assert result["seeds"] == [42, 43, 44]
        assert result["n_runs"] == 3
        for n in range(1, STABILITY_TOP_WORDS + 1):
            assert len(result["per_topic"][n]) == 4
            assert all(1 / 3 <= s <= 1.0 for s in result["per_topic"][n])

    def test_reference_model_is_the_users_model(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        model, _, _, _ = run_lda(dtm, vec, n_topics=4, random_seed=7)
        given = topic_stability(dtm, vec, n_topics=4, random_seed=7, n_runs=2,
                                reference_model=model)
        fitted = topic_stability(dtm, vec, n_topics=4, random_seed=7, n_runs=2)
        assert given["overlaps"] == fitted["overlaps"]

    def test_progress_callback_called(self, themed_texts):
        dtm, vec, _ = build_dtm(themed_texts, min_df=1, max_df=1.0)
        calls = []
        topic_stability(dtm, vec, n_topics=4, n_runs=3,
                        progress_callback=lambda c, t: calls.append((c, t)))
        assert calls == [(1, 3), (2, 3), (3, 3)]
