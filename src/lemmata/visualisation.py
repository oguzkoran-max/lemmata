"""Visualisation module for Lemmata.

All chart-building lives here.  Every function returns a figure or data
object — **no** ``st.*`` calls.  The UI layer (app.py) is responsible for
rendering.
"""

from __future__ import annotations

import logging
from typing import Any

import altair as alt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from lemmata.config import (
    COLOR_HEATMAP,
    COLOR_PALETTE,
    COLOR_PRIMARY,
    WORDCLOUD_HEIGHT,
    WORDCLOUD_WIDTH,
)

logger = logging.getLogger(__name__)

# ── Tableau10 colour list (fixed-order, colour-blind friendly) ────────────────
# Altair's "tableau10" scheme as explicit hex so we can map topic_id → colour
# consistently across every chart (decision 125).

_TABLEAU10: list[str] = [
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ac",
]


def get_topic_color(topic_id: int) -> str:
    """Return a deterministic colour for *topic_id* (1-indexed).

    Wraps around if there are more than 10 topics (decision 125).

    Parameters
    ----------
    topic_id:
        1-based topic number.

    Returns
    -------
    str
        Hex colour string.
    """
    return _TABLEAU10[(topic_id - 1) % len(_TABLEAU10)]


def get_topic_color_scale(n_topics: int) -> alt.Scale:
    """Build an Altair ordinal colour scale for *n_topics*.

    Parameters
    ----------
    n_topics:
        Total number of topics.

    Returns
    -------
    alt.Scale
        Ready to use in ``alt.Color(..., scale=...)``.
    """
    domain = [f"Topic {i}" for i in range(1, n_topics + 1)]
    colors = [get_topic_color(i) for i in range(1, n_topics + 1)]
    return alt.Scale(domain=domain, range=colors)


# ── Altair Theme ──────────────────────────────────────────────────────────────


def configure_altair_theme() -> None:
    """Register and enable the Lemmata Altair theme.

    Call once at app startup.  Implements decisions 34 and 144:
    white background, minimal axes, light grid, sans-serif.
    """

    def _lemmata_theme() -> dict[str, Any]:
        return {
            "config": {
                "background": "white",
                "font": "sans-serif",
                "axis": {
                    "grid": True,
                    "gridColor": "#e8e8e8",
                    "gridOpacity": 0.5,
                    "labelFont": "sans-serif",
                    "labelFontSize": 11,
                    "titleFont": "sans-serif",
                    "titleFontSize": 12,
                    "tickColor": "#cccccc",
                    "domainColor": "#cccccc",
                },
                "header": {
                    "labelFont": "sans-serif",
                    "titleFont": "sans-serif",
                },
                "legend": {
                    "labelFont": "sans-serif",
                    "titleFont": "sans-serif",
                },
                "title": {
                    "font": "sans-serif",
                    "fontSize": 14,
                },
                "view": {
                    "stroke": "transparent",
                },
            }
        }

    # Support both new (>=5.5) and old (<5.5) Altair theme API.
    if hasattr(alt, "theme") and hasattr(alt.theme, "register"):
        @alt.theme.register("lemmata", enable=True)
        def _lemmata_theme_new() -> alt.theme.ThemeConfig:
            return alt.theme.ThemeConfig(_lemmata_theme())
    else:
        alt.themes.register("lemmata", _lemmata_theme)
        alt.themes.enable("lemmata")


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════


def create_topic_bars(
    topic: dict[str, Any],
    color: str | None = None,
) -> alt.Chart:
    """Create an interactive horizontal bar chart for a single topic.

    Implements decisions 15, 51, 144.

    Parameters
    ----------
    topic:
        Topic summary dict with ``words``, ``weights``, ``topic_id``.
    color:
        Bar colour.  *None* → auto from ``get_topic_color``.

    Returns
    -------
    alt.Chart
        Altair chart (600 px wide).
    """
    if color is None:
        color = get_topic_color(topic["topic_id"])

    df = pd.DataFrame(
        {"word": topic["words"], "weight": topic["weights"]}
    )
    # Preserve rank order (top word at the top).
    df["rank"] = range(len(df))

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("weight:Q", title="Weight"),
            y=alt.Y("word:N", sort=alt.EncodingSortField("rank"), title=None),
            color=alt.value(color),
            tooltip=[
                alt.Tooltip("word:N", title="Word"),
                alt.Tooltip("weight:Q", title="Weight", format=".4f"),
            ],
        )
        .properties(width=600, title=topic.get("label", f"Topic {topic['topic_id']}"))
    )

    return chart


# ═══════════════════════════════════════════════════════════════════════════════
# WORDCLOUD
# ═══════════════════════════════════════════════════════════════════════════════


def create_wordcloud(
    topic: dict[str, Any],
    width: int = WORDCLOUD_WIDTH,
    height: int = WORDCLOUD_HEIGHT,
    color: str | None = None,
) -> Figure:
    """Create a matplotlib wordcloud for a single topic.

    Implements decisions 15, 111, 148.

    Parameters
    ----------
    topic:
        Topic summary dict with ``words`` and ``weights``.
    width:
        Image width in pixels.
    height:
        Image height in pixels.
    color:
        Dominant colour for the cloud.  *None* → auto.

    Returns
    -------
    matplotlib.figure.Figure
        Static figure — export with :func:`file_io.export_figure_png`.
    """
    from wordcloud import WordCloud

    if color is None:
        color = get_topic_color(topic.get("topic_id", 1))

    freq = dict(zip(topic["words"], topic["weights"]))

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        color_func=lambda *_a, **_kw: color,
        prefer_horizontal=0.7,
        max_words=len(freq),
    ).generate_from_frequencies(freq)

    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_axis_off()
    fig.tight_layout(pad=0)

    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════


def create_heatmap(
    doc_topic_matrix: np.ndarray,
    doc_labels: list[str],
    topic_labels: list[str],
) -> alt.Chart:
    """Create a document-topic heatmap.

    Implements decisions 145 and 150.

    Parameters
    ----------
    doc_topic_matrix:
        Array of shape ``(n_docs, n_topics)``.
    doc_labels:
        Document names.
    topic_labels:
        Topic display labels.

    Returns
    -------
    alt.Chart
        Altair heatmap with hover tooltips, viridis colourmap.
    """
    n_docs = len(doc_labels)
    n_topics = len(topic_labels)

    rows: list[dict[str, Any]] = []
    for i, doc in enumerate(doc_labels):
        for j, topic in enumerate(topic_labels):
            rows.append(
                {"Document": doc, "Topic": topic, "Weight": float(doc_topic_matrix[i, j])}
            )

    df = pd.DataFrame(rows)

    chart_width = max(300, n_topics * 60)
    chart_height = max(200, n_docs * 25)

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Topic:N", sort=topic_labels, title=None),
            y=alt.Y("Document:N", sort=doc_labels, title=None),
            color=alt.Color(
                "Weight:Q",
                scale=alt.Scale(scheme=COLOR_HEATMAP),
                legend=alt.Legend(title="Weight"),
            ),
            tooltip=[
                alt.Tooltip("Document:N"),
                alt.Tooltip("Topic:N"),
                alt.Tooltip("Weight:Q", format=".4f"),
            ],
        )
        .properties(width=chart_width, height=chart_height)
    )

    return chart


# ═══════════════════════════════════════════════════════════════════════════════
# DISTRIBUTION CHART
# ═══════════════════════════════════════════════════════════════════════════════


def create_distribution_chart(
    doc_topic_matrix: np.ndarray,
    doc_labels: list[str],
    topic_labels: list[str],
) -> Figure:
    """Create a horizontal stacked bar chart of topic distribution per document.

    Implements decision 114.  Uses matplotlib for reliable rendering on
    Streamlit Cloud (Altair/Vega-Lite renderer is unreliable there).

    Parameters
    ----------
    doc_topic_matrix:
        Array ``(n_docs, n_topics)``.
    doc_labels:
        Document names.
    topic_labels:
        Topic display labels.

    Returns
    -------
    Figure
        Matplotlib figure with horizontal stacked bars.
    """
    import matplotlib.pyplot as plt

    n_docs, n_topics = doc_topic_matrix.shape
    # Normalize rows to sum to 1.
    row_sums = doc_topic_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    normed = doc_topic_matrix / row_sums

    fig_height = max(3, n_docs * 0.28)
    fig, ax = plt.subplots(figsize=(8, fig_height))

    y_pos = np.arange(n_docs)
    left = np.zeros(n_docs)

    for j in range(n_topics):
        color = _TABLEAU10[j % len(_TABLEAU10)]
        label = topic_labels[j] if j < len(topic_labels) else f"Topic {j + 1}"
        ax.barh(y_pos, normed[:, j], left=left, height=0.7, color=color, label=label)
        left += normed[:, j]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(doc_labels, fontsize=7)
    ax.set_xlabel("Topic proportion")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# DIACHRONIC VIEW
# ═══════════════════════════════════════════════════════════════════════════════


def create_diachronic_chart(
    doc_topic_matrix: np.ndarray,
    doc_labels: list[str],
    topic_labels: list[str],
    file_boundaries: list[int] | None = None,
) -> Figure:
    """Create a topic-weight trend line chart over document order.

    Implements decision 115.  Uses matplotlib for reliable rendering on
    Streamlit Cloud.

    Parameters
    ----------
    doc_topic_matrix:
        Array ``(n_docs, n_topics)``.
    doc_labels:
        Document names (used for hover).
    topic_labels:
        Topic display labels.
    file_boundaries:
        Indices where a new source file begins (for vertical rules).
        *None* → no boundaries drawn.

    Returns
    -------
    Figure
        Matplotlib line chart with optional vertical boundary rules.
    """
    import matplotlib.pyplot as plt

    n_docs, n_topics = doc_topic_matrix.shape
    x = np.arange(n_docs)

    fig, ax = plt.subplots(figsize=(10, 4))

    for j in range(n_topics):
        color = _TABLEAU10[j % len(_TABLEAU10)]
        label = topic_labels[j] if j < len(topic_labels) else f"Topic {j + 1}"
        ax.plot(x, doc_topic_matrix[:, j], linewidth=1.5, color=color, label=label)

    if file_boundaries:
        for boundary in file_boundaries:
            ax.axvline(x=boundary, color="#999999", linestyle="--", linewidth=1)

    ax.set_xlabel("Document order")
    ax.set_ylabel("Topic weight")
    ax.set_xlim(0, n_docs - 1)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TOPIC MAP (pyLDAvis / fallback)
# ═══════════════════════════════════════════════════════════════════════════════


def create_topic_map(
    model: Any,
    dtm: Any,
    vectorizer: Any,
) -> str | alt.Chart:
    """Create an interactive topic map.

    Tries pyLDAvis first; falls back to an Altair 2-D scatter if that
    fails (decision 16).

    Parameters
    ----------
    model:
        Fitted ``LatentDirichletAllocation``.
    dtm:
        Sparse document-term matrix.
    vectorizer:
        Fitted ``CountVectorizer``.

    Returns
    -------
    str or alt.Chart
        HTML string (pyLDAvis) **or** Altair scatter fallback.
    """
    try:
        import pyLDAvis
        import pyLDAvis.sklearn

        panel = pyLDAvis.sklearn.prepare(model, dtm, vectorizer, sort_topics=False)
        return pyLDAvis.prepared_data_to_html(panel)
    except Exception as exc:
        logger.info("pyLDAvis unavailable, using scatter fallback: %s", exc)
        return _topic_map_fallback(model, dtm)


def _topic_map_fallback(model: Any, dtm: Any) -> alt.Chart:
    """2-D PCA scatter of topic centres as pyLDAvis fallback."""
    from sklearn.decomposition import PCA

    components = model.components_  # (n_topics, n_features)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(components)  # (n_topics, 2)

    # Topic prevalence as circle size.
    doc_topic = model.transform(dtm)
    avg_weights = doc_topic.mean(axis=0)

    n_topics = components.shape[0]

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "Topic": [f"Topic {i+1}" for i in range(n_topics)],
            "Prevalence": avg_weights,
        }
    )

    color_scale = get_topic_color_scale(n_topics)

    chart = (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("x:Q", title="PC 1", axis=alt.Axis(grid=True)),
            y=alt.Y("y:Q", title="PC 2", axis=alt.Axis(grid=True)),
            size=alt.Size(
                "Prevalence:Q",
                scale=alt.Scale(range=[200, 2000]),
                legend=alt.Legend(title="Avg weight"),
            ),
            color=alt.Color("Topic:N", scale=color_scale),
            tooltip=[
                alt.Tooltip("Topic:N"),
                alt.Tooltip("Prevalence:Q", format=".4f"),
            ],
        )
        .properties(width=500, height=500, title="Topic Map (PCA)")
    )

    return chart


# ═══════════════════════════════════════════════════════════════════════════════
# TOP LEMMAS BAR CHART (Overview)
# ═══════════════════════════════════════════════════════════════════════════════


def create_top_lemmas_chart(
    lemma_counts: dict[str, int],
    n: int = 20,
) -> alt.Chart:
    """Create a bar chart of the most frequent lemmas (pre-LDA overview).

    Implements decision 92.

    Parameters
    ----------
    lemma_counts:
        Mapping of lemma → corpus frequency.
    n:
        Number of top lemmas to show.

    Returns
    -------
    alt.Chart
        Horizontal bar chart, teal colour.
    """
    top = sorted(lemma_counts.items(), key=lambda x: x[1], reverse=True)[:n]
    df = pd.DataFrame(top, columns=["Lemma", "Frequency"])
    df["rank"] = range(len(df))

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Frequency:Q", title="Frequency"),
            y=alt.Y("Lemma:N", sort=alt.EncodingSortField("rank"), title=None),
            color=alt.value(COLOR_PRIMARY),
            tooltip=[
                alt.Tooltip("Lemma:N"),
                alt.Tooltip("Frequency:Q"),
            ],
        )
        .properties(width=600, title="Top Lemmas (pre-LDA)")
    )

    return chart


# ═══════════════════════════════════════════════════════════════════════════════
# COHERENCE DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════


def get_coherence_display(c_v: float) -> dict[str, str]:
    """Return colour, label, and suggestion for a C_v coherence score.

    Implements decision 41: green / yellow / red bands, no emoji.

    Parameters
    ----------
    c_v:
        C_v coherence value.

    Returns
    -------
    dict
        ``{"color": str, "background": str, "label": str, "suggestion": str}``.
    """
    if c_v >= 0.5:
        return {
            "color": "#155724",
            "background": "#d4edda",
            "label": "Good",
            "suggestion": (
                "Coherence is solid. Review the topics qualitatively "
                "to confirm they make sense for your research question."
            ),
        }
    if c_v >= 0.3:
        return {
            "color": "#856404",
            "background": "#fff3cd",
            "label": "Moderate",
            "suggestion": (
                "Some topics may be mixed. Try adjusting the number of "
                "topics, changing POS filters, or adding custom stopwords."
            ),
        }
    return {
        "color": "#721c24",
        "background": "#f8d7da",
        "label": "Low",
        "suggestion": (
            "Topics are not well differentiated. Consider: fewer topics, "
            "a larger corpus, different chunk size, or reviewing the "
            "language selection."
        ),
    }
