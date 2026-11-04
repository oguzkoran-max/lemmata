"""Tests for lemmata.visualisation."""

from __future__ import annotations

import re
from typing import Any

import altair as alt
import numpy as np
import pytest
from matplotlib.figure import Figure

from lemmata.visualisation import (
    configure_altair_theme,
    create_coherence_sweep_chart,
    create_diachronic_chart,
    create_distribution_chart,
    create_heatmap,
    create_top_lemmas_chart,
    create_topic_bars,
    create_topic_map,
    create_wordcloud,
    COHERENCE_NOTE,
    get_topic_color,
    get_topic_color_scale,
)
from lemmata.visualisation import _topic_map_fallback


# ── get_topic_color ───────────────────────────────────────────────────────────


class TestGetTopicColor:
    """Verify topic colour assignment."""

    def test_returns_hex_string(self):
        color = get_topic_color(1)
        assert isinstance(color, str)
        assert re.match(r"^#[0-9a-fA-F]{6}$", color)

    def test_different_ids_different_colors(self):
        colors = {get_topic_color(i) for i in range(1, 11)}
        assert len(colors) == 10

    def test_wraps_after_10(self):
        assert get_topic_color(1) == get_topic_color(11)

    def test_consistent(self):
        # Same ID always returns same colour.
        assert get_topic_color(3) == get_topic_color(3)


# ── get_topic_color_scale ─────────────────────────────────────────────────────


class TestGetTopicColorScale:
    """Verify Altair colour scale construction."""

    def test_returns_scale(self):
        scale = get_topic_color_scale(5)
        assert isinstance(scale, alt.Scale)


# ── configure_altair_theme ────────────────────────────────────────────────────


class TestConfigureAltairTheme:
    """Verify theme registration."""

    def test_runs_without_error(self):
        configure_altair_theme()

    def test_theme_registered(self):
        configure_altair_theme()
        # Support both new (>=5.5) and old (<5.5) Altair theme API.
        if hasattr(alt, "theme") and hasattr(alt.theme, "names"):
            assert "lemmata" in alt.theme.names()
        else:
            assert "lemmata" in alt.themes._registered


# ── create_topic_bars ─────────────────────────────────────────────────────────


class TestCreateTopicBars:
    """Verify topic bar chart creation."""

    def test_returns_altair_chart(self, sample_topics):
        chart = create_topic_bars(sample_topics[0])
        assert isinstance(chart, alt.Chart)

    def test_accepts_custom_color(self, sample_topics):
        chart = create_topic_bars(sample_topics[0], color="#ff0000")
        assert isinstance(chart, alt.Chart)


# ── create_wordcloud ──────────────────────────────────────────────────────────


class TestCreateWordcloud:
    """Verify wordcloud creation."""

    def test_returns_matplotlib_figure(self, sample_topics):
        fig = create_wordcloud(sample_topics[0])
        assert isinstance(fig, Figure)

    def test_custom_dimensions(self, sample_topics):
        fig = create_wordcloud(sample_topics[0], width=400, height=200)
        assert isinstance(fig, Figure)


# ── create_heatmap ────────────────────────────────────────────────────────────


class TestCreateHeatmap:
    """Verify heatmap creation."""

    def test_returns_altair_chart(self, sample_doc_topic_matrix):
        chart = create_heatmap(
            sample_doc_topic_matrix,
            ["doc1", "doc2", "doc3"],
            ["Topic 1", "Topic 2"],
        )
        assert isinstance(chart, alt.Chart)


# ── create_distribution_chart ─────────────────────────────────────────────────


class TestCreateDistributionChart:
    """Verify distribution chart creation."""

    def test_returns_matplotlib_figure(self, sample_doc_topic_matrix):
        chart = create_distribution_chart(
            sample_doc_topic_matrix,
            ["doc1", "doc2", "doc3"],
            ["Topic 1", "Topic 2"],
        )
        from matplotlib.figure import Figure
        assert isinstance(chart, Figure)


# ── create_diachronic_chart ───────────────────────────────────────────────────


class TestCreateDiachronicChart:
    """Verify diachronic chart creation."""

    def test_returns_figure_without_boundaries(self, sample_doc_topic_matrix):
        from matplotlib.figure import Figure
        chart = create_diachronic_chart(
            sample_doc_topic_matrix,
            ["doc1", "doc2", "doc3"],
            ["Topic 1", "Topic 2"],
        )
        assert isinstance(chart, Figure)

    def test_returns_figure_with_boundaries(self, sample_doc_topic_matrix):
        from matplotlib.figure import Figure
        chart = create_diachronic_chart(
            sample_doc_topic_matrix,
            ["doc1", "doc2", "doc3"],
            ["Topic 1", "Topic 2"],
            file_boundaries=[1],
        )
        assert isinstance(chart, Figure)


# ── create_top_lemmas_chart ───────────────────────────────────────────────────


class TestCreateTopLemmasChart:
    """Verify top lemmas bar chart."""

    def test_returns_altair_chart(self):
        counts = {"word1": 50, "word2": 30, "word3": 20}
        chart = create_top_lemmas_chart(counts)
        assert isinstance(chart, alt.Chart)

    def test_respects_n_param(self):
        counts = {f"w{i}": 100 - i for i in range(50)}
        chart = create_top_lemmas_chart(counts, n=10)
        assert isinstance(chart, alt.Chart)


# ── coherence note (decision 41, bands removed in P046) ───────────────────────


class TestCoherenceNote:
    """C_v is shown with its source and without quality bands."""

    def test_cites_the_measures_definition(self):
        assert "Röder et al. (2015)" in COHERENCE_NOTE

    def test_has_no_quality_labels(self):
        for label in ("Good", "Moderate", "Low"):
            assert label not in COHERENCE_NOTE

    def test_bands_function_is_gone(self):
        import lemmata.visualisation as vis

        assert not hasattr(vis, "get_coherence_display")


# ── create_coherence_sweep_chart ─────────────────────────────────────────────


class TestCreateCoherenceSweepChart:
    """Verify coherence sweep chart creation."""

    def test_returns_matplotlib_figure(self):
        fig = create_coherence_sweep_chart(
            [2, 3, 4, 5], [0.3, 0.45, 0.5, 0.42],
        )
        assert isinstance(fig, Figure)

    def test_with_current_k(self):
        fig = create_coherence_sweep_chart(
            [2, 3, 4, 5], [0.3, 0.45, 0.5, 0.42], current_k=3,
        )
        assert isinstance(fig, Figure)

    def test_without_current_k(self):
        fig = create_coherence_sweep_chart(
            [2, 3, 4], [0.4, 0.5, 0.35],
        )
        assert isinstance(fig, Figure)


# ── create_topic_map (P045) ───────────────────────────────────────────────────


class TestCreateTopicMap:
    """pyLDAvis map for both methods; topics in display order."""

    @pytest.mark.parametrize("method", ["gibbs", "variational"])
    def test_pyldavis_html_for_both_methods(self, processed_texts, method):
        pytest.importorskip("pyLDAvis")
        from lemmata.modelling import build_dtm, run_lda

        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)
        model, _, _, info = run_lda(dtm, vec, n_topics=2, method=method)
        result = create_topic_map(
            model, dtm, vec, topic_order=info["prevalence_order"],
        )
        # Before v0.3.0 the map always fell back to the PCA chart.
        assert isinstance(result, str)
        assert "ldavis" in result.lower()

    def test_fallback_follows_display_order(self):
        components = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        doc_topic = np.array([[0.1, 0.7, 0.2], [0.1, 0.5, 0.4]])
        order = list(np.argsort(doc_topic.mean(axis=0))[::-1])
        chart = _topic_map_fallback(components[order], doc_topic[:, order])
        prevalence = list(chart.data["Prevalence"])
        assert prevalence == sorted(prevalence, reverse=True)
