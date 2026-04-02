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
    create_wordcloud,
    get_coherence_display,
    get_topic_color,
    get_topic_color_scale,
)


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


# ── get_coherence_display ─────────────────────────────────────────────────────


class TestGetCoherenceDisplay:
    """Verify coherence colour coding (decision 41)."""

    def test_returns_required_keys(self):
        result = get_coherence_display(0.55)
        required = {"color", "background", "label", "suggestion"}
        assert required == set(result.keys())

    def test_good_coherence(self):
        result = get_coherence_display(0.65)
        assert result["label"] == "Good"
        assert "d4edda" in result["background"]  # green

    def test_moderate_coherence(self):
        result = get_coherence_display(0.4)
        assert result["label"] == "Moderate"
        assert "fff3cd" in result["background"]  # yellow

    def test_low_coherence(self):
        result = get_coherence_display(0.2)
        assert result["label"] == "Low"
        assert "f8d7da" in result["background"]  # red

    def test_boundary_05(self):
        result = get_coherence_display(0.5)
        assert result["label"] == "Good"

    def test_boundary_03(self):
        result = get_coherence_display(0.3)
        assert result["label"] == "Moderate"

    def test_zero(self):
        result = get_coherence_display(0.0)
        assert result["label"] == "Low"


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
