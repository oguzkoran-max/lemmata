"""Lemmata — Streamlit application.

UI layer only.  All business logic lives in the other five modules.
This file wires config, preprocessing, modelling, visualisation, and
file_io together behind Streamlit widgets.
"""

from __future__ import annotations

import traceback
from collections import Counter
from typing import Any

import numpy as np
import streamlit as st

from lemmata.config import (
    ALLOWED_EXTENSIONS,
    APP_TITLE,
    APP_PAGE_ICON,
    APP_VERSION,
    BG_MAIN,
    BG_SIDEBAR,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    COLOR_PRIMARY,
    DEFAULT_POS_TAGS,
    LDA_MAX_ITER,
    MAX_FILE_SIZE_MB,
    MAX_TOTAL_SIZE_MB,
    NUM_TOPICS_DEFAULT,
    NUM_TOPICS_MAX,
    NUM_TOPICS_MIN,
    POS_PRESETS,
    RANDOM_SEED,
    SUPPORTED_LANGUAGES,
    WORDS_PER_TOPIC_DEFAULT,
    WORDS_PER_TOPIC_MAX,
    WORDS_PER_TOPIC_MIN,
    get_df_auto,
)
from lemmata.file_io import (
    FileReadError,
    export_figure_png,
    export_figure_svg,
    export_pdf_report,
    export_zip,
    get_environment_info,
    get_file_preview,
    get_zip_filename,
    read_files,
    text_from_paste,
)
from lemmata.modelling import build_dtm, compute_coherence, run_lda
from lemmata.preprocessing import check_language_match, load_spacy_model, process_documents
from lemmata.visualisation import (
    configure_altair_theme,
    create_diachronic_chart,
    create_distribution_chart,
    create_heatmap,
    create_top_lemmas_chart,
    create_topic_bars,
    create_topic_map,
    create_wordcloud,
    get_coherence_display,
    get_topic_color,
)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE SETUP
# ═══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_PAGE_ICON,
    layout="wide",
)

# Minimal custom CSS: hide hamburger menu, hide footer (decisions 48, 189).
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    div[data-testid="stDecoration"] {display: none;}
    .block-container {padding-top: 1rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

configure_altair_theme()


# ═══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ═══════════════════════════════════════════════════════════════════════════════


@st.cache_resource(show_spinner=False)
def _load_spacy(language: str):
    """Cache spaCy model across reruns (decision 53)."""
    return load_spacy_model(language)


# ═══════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ═══════════════════════════════════════════════════════════════════════════════


def _init_state() -> None:
    """Ensure all session-state keys exist."""
    defaults: dict[str, Any] = {
        "results": None,
        "params": None,
        "analysis_run": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════


def _render_sidebar() -> dict[str, Any]:
    """Build the sidebar and return the current parameter dict."""
    with st.sidebar:
        # ── Logo (decision 131) ───────────────────────────────────────────
        st.markdown(
            f'<h1 style="margin:0;padding:0;">'
            f'<span style="color:{COLOR_PRIMARY};font-size:2.2rem;">\u03bb</span> '
            f'<span style="color:{COLOR_PRIMARY};font-weight:600;">Lemmata</span>'
            f'</h1>'
            f'<p style="color:gray;font-size:0.85rem;margin-top:-0.3rem;">'
            f"Multilingual Topic Modeling</p>",
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Language ──────────────────────────────────────────────────────
        lang_options = list(SUPPORTED_LANGUAGES.keys())
        lang_labels = {"it": "Italian", "en": "English", "de": "German",
                       "fr": "French", "es": "Spanish"}
        language = st.selectbox(
            "Language",
            options=lang_options,
            format_func=lambda x: f"{lang_labels.get(x, x)} ({x})",
            help="Select the language of your corpus.",
        )

        st.divider()

        # ── Parameters ────────────────────────────────────────────────────
        st.markdown("**Parameters**")

        chunk_size = st.slider(
            "Chunk size (words)",
            min_value=CHUNK_SIZE_MIN,
            max_value=CHUNK_SIZE_MAX,
            value=CHUNK_SIZE_DEFAULT,
            step=100,
            help="Target word count per chunk when processing a single file. "
                 "Sentence boundaries are respected.",
        )

        n_topics = st.slider(
            "Number of topics",
            min_value=NUM_TOPICS_MIN,
            max_value=NUM_TOPICS_MAX,
            value=NUM_TOPICS_DEFAULT,
            help="How many topics the model should extract from your corpus.",
        )

        n_words = st.slider(
            "Words per topic",
            min_value=WORDS_PER_TOPIC_MIN,
            max_value=WORDS_PER_TOPIC_MAX,
            value=WORDS_PER_TOPIC_DEFAULT,
            help="Number of top words to display for each topic.",
        )

        st.divider()

        # ── POS Filter (decisions 95, 143) ────────────────────────────────
        st.markdown("**POS Filter**")

        preset_name = st.selectbox(
            "Preset",
            options=list(POS_PRESETS.keys()),
            help="Select a part-of-speech filter preset, or choose 'Custom'.",
        )

        if preset_name == "Custom":
            all_pos = ["NOUN", "PROPN", "ADJ", "VERB", "ADV", "ADP",
                       "AUX", "DET", "NUM", "PRON"]
            pos_tags = st.multiselect(
                "Custom POS tags",
                options=all_pos,
                default=DEFAULT_POS_TAGS,
            )
        else:
            pos_tags = POS_PRESETS[preset_name]
            st.caption(f"Tags: {', '.join(pos_tags)}")

        st.divider()

        # ── Custom Stopwords ──────────────────────────────────────────────
        custom_sw_text = st.text_area(
            "Custom stopwords",
            value="",
            height=68,
            help="One word per line. These will be removed in addition to "
                 "the built-in stopwords.",
        )
        custom_stopwords: set[str] = set()
        if custom_sw_text.strip():
            custom_stopwords = {
                w.strip().lower()
                for w in custom_sw_text.strip().splitlines()
                if w.strip()
            }

        # ── Advanced ──────────────────────────────────────────────────────
        with st.expander("Advanced"):
            seed = st.number_input(
                "Random seed",
                value=RANDOM_SEED,
                min_value=0,
                step=1,
                help="Set a seed for reproducible results (decision 121).",
            )
            max_iter = st.number_input(
                "Max iterations",
                value=LDA_MAX_ITER,
                min_value=10,
                max_value=500,
                step=10,
                help="Maximum EM iterations for LDA. Increase if "
                     "convergence warning appears.",
            )
            auto_min, auto_max = get_df_auto(10)
            min_df = st.number_input(
                "min_df",
                value=auto_min,
                min_value=1,
                step=1,
                help="Minimum document frequency. Terms appearing in fewer "
                     "documents are removed.",
            )
            max_df = st.number_input(
                "max_df",
                value=auto_max,
                min_value=0.1,
                max_value=1.0,
                step=0.05,
                format="%.2f",
                help="Maximum document frequency (proportion). Very common "
                     "terms are removed.",
            )
            use_auto_df = st.checkbox("Auto-adjust by corpus size", value=True)

        st.divider()

        # ── Reset to defaults (decision 129) ──────────────────────────────
        if st.button("Reset to defaults", type="secondary"):
            for key in ["results", "params", "analysis_run"]:
                st.session_state[key] = None if key != "analysis_run" else False
            st.rerun()

        st.divider()

        # ── Post-analysis summary (decision 186) ─────────────────────────
        if st.session_state.get("results"):
            res = st.session_state["results"]
            c_v = res.get("coherence", {}).get("c_v")
            mi = res.get("model_info", {})
            trace = res.get("preprocessing_trace", {})
            summary_parts = [f"{mi.get('n_topics', '?')} topics"]
            if c_v is not None:
                summary_parts.append(f"C_v: {c_v:.2f}")
            summary_parts.append(
                f"{trace.get('unique_lemmas', '?')} lemmas"
            )
            st.caption(" · ".join(summary_parts))
            st.divider()

        # ── About (decision 94) ──────────────────────────────────────────
        with st.expander("About"):
            st.markdown(
                "**Lemmata** is a browser-based, multilingual LDA topic "
                "modeling platform for humanities researchers.\n\n"
                "Built with spaCy, scikit-learn, Gensim, and Streamlit.\n\n"
                "**Authors:** Oğuz Koran, Hakan Cangır, Barış Yücesan\n\n"
                "[Source code](https://github.com/oguzkoran/lemmata)"
            )

        # ── How to cite (decision 24) ────────────────────────────────────
        with st.expander("How to cite"):
            st.code(
                "Koran, O., Cangır, H., & Yücesan, B. (2026). "
                "Lemmata: A Browser-Based Multilingual Topic Modeling "
                "Platform for Humanities Research. "
                "Digital Scholarship in the Humanities.",
                language=None,
            )
            st.code(
                "@article{koran2026lemmata,\n"
                "  title={Lemmata: A Browser-Based Multilingual Topic "
                "Modeling Platform for Humanities Research},\n"
                "  author={Koran, O{\\u{g}}uz and Cang{\\i}r, Hakan "
                "and Y{\\\"u}cesan, Bar{\\i}{\\c{s}}},\n"
                "  journal={Digital Scholarship in the Humanities},\n"
                "  year={2026}\n"
                "}",
                language="bibtex",
            )

        # ── Feedback (decision 60) ───────────────────────────────────────
        st.markdown(
            "[Report a bug · Request a feature]"
            "(https://github.com/oguzkoran/lemmata/issues)"
        )

        # ── Version footer (decision 47) ─────────────────────────────────
        st.caption(
            f"v{APP_VERSION} · "
            "[GitHub](https://github.com/oguzkoran/lemmata) · MIT License"
        )

    return {
        "language": language,
        "chunk_size": chunk_size,
        "n_topics": n_topics,
        "n_words": n_words,
        "pos_tags": pos_tags,
        "custom_stopwords": custom_stopwords,
        "seed": int(seed),
        "max_iter": int(max_iter),
        "min_df": min_df if not use_auto_df else None,
        "max_df": max_df if not use_auto_df else None,
        "use_auto_df": use_auto_df,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA — FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════


def _render_upload() -> list[dict[str, str]]:
    """Render the file upload area and return parsed texts."""
    st.markdown(
        "Upload your text files to begin. Lemmata accepts "
        f"**{', '.join(ALLOWED_EXTENSIONS)}** files "
        f"(max {MAX_FILE_SIZE_MB} MB each, {MAX_TOTAL_SIZE_MB} MB total)."
    )

    uploaded = st.file_uploader(
        "Upload files",
        type=ALLOWED_EXTENSIONS,
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Text paste alternative (decision 91).
    with st.expander("Or paste text directly"):
        pasted = st.text_area(
            "Paste your text here",
            height=150,
            label_visibility="collapsed",
        )

    st.caption(
        "By uploading files you confirm that you have the right to "
        "process them. No data is stored permanently."
    )

    texts: list[dict[str, str]] = []
    warnings: list[str] = []

    if uploaded:
        texts, warnings = read_files(uploaded)

        # File previews (decision 65).
        if texts:
            with st.expander(f"File preview ({len(texts)} files)"):
                for t in texts:
                    preview_words = t["content"].split()
                    wc = len(preview_words)
                    preview = " ".join(preview_words[:200])
                    st.markdown(f"**{t['filename']}** — {wc:,} words")
                    st.text(preview + ("..." if wc > 200 else ""))
                    st.divider()

    elif pasted and pasted.strip():
        texts = [text_from_paste(pasted)]

    for w in warnings:
        st.warning(w)

    return texts


# ═══════════════════════════════════════════════════════════════════════════════
# WELCOME SCREEN
# ═══════════════════════════════════════════════════════════════════════════════


def _render_welcome() -> None:
    """Show welcome screen for first-time visitors (decision 26)."""
    st.markdown("### Welcome to Lemmata")
    st.markdown(
        "Lemmata helps you discover themes in your text corpus using "
        "**LDA topic modeling** — no coding required."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**1. Upload**")
        st.markdown(
            "Upload your text files (txt, pdf, docx, odt, epub) "
            "or paste text directly."
        )
    with col2:
        st.markdown("**2. Configure**")
        st.markdown(
            "Choose language, number of topics, and filter settings "
            "in the sidebar."
        )
    with col3:
        st.markdown("**3. Analyse**")
        st.markdown(
            "Click **Run Analysis** to extract topics, visualise "
            "results, and export your findings."
        )

    with st.expander("What is topic modeling?"):
        st.markdown(
            "Topic modeling is an unsupervised machine learning method "
            "that discovers abstract 'topics' in a collection of texts. "
            "Each topic is a cluster of words that frequently co-occur. "
            "LDA (Latent Dirichlet Allocation) assumes each document is "
            "a mixture of topics, and each topic is a mixture of words.\n\n"
            "**Example:** In a literary corpus, LDA might find topics like "
            "'nature/landscape', 'family/relationships', and "
            "'war/conflict' — each represented by its most probable words."
        )

    with st.expander("Sample data"):
        st.markdown(
            "To try Lemmata, you can use short public-domain texts. "
            "Check the `examples/` folder in the "
            "[GitHub repository](https://github.com/oguzkoran/lemmata) "
            "for sample Italian, English, and German texts."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS RUNNER
# ═══════════════════════════════════════════════════════════════════════════════


def _run_analysis(
    texts: list[dict[str, str]], params: dict[str, Any]
) -> dict[str, Any]:
    """Execute the full pipeline with st.status progress (decision 35)."""
    # Run early language check so we can show a warning even if the
    # pipeline crashes (decision 58 / P022).
    early_lang_warning: str | None = None
    try:
        nlp = _load_spacy(params["language"])
        early_lang_warning = check_language_match(texts, nlp)
    except Exception:
        pass  # If the check itself fails, just skip it.

    try:
        return _run_analysis_inner(texts, params)
    except Exception as exc:
        # Show early language warning *before* the error so the user
        # sees the likely root cause first.
        if early_lang_warning:
            st.warning(early_lang_warning)
        st.error(
            "Analysis failed. This usually happens when the text is too "
            "short, the wrong language is selected, or too few words "
            "remain after preprocessing. Try uploading a larger text or "
            "checking your language setting."
        )
        with st.expander("Technical details"):
            st.code(traceback.format_exc())
        # Clean up session state so the app remains usable.
        for key in ("results", "analysis_done"):
            st.session_state.pop(key, None)
        return {}


def _run_analysis_inner(
    texts: list[dict[str, str]], params: dict[str, Any]
) -> dict[str, Any]:
    """Inner pipeline — may raise on bad input."""
    with st.status("Running analysis...", expanded=True) as status:
        # Step 1: Load spaCy model.
        st.write("Loading language model...")
        nlp = _load_spacy(params["language"])

        # Step 2: Preprocess.
        st.write("Preprocessing texts...")
        processed_texts, doc_labels, trace = process_documents(
            texts=texts,
            language=params["language"],
            pos_tags=params["pos_tags"],
            custom_stopwords=params["custom_stopwords"],
            chunk_size=params["chunk_size"],
            nlp=nlp,
        )

        if not processed_texts:
            status.update(label="Analysis failed", state="error")
            st.error(
                "No text remained after preprocessing. "
                "Try different POS filters, fewer stopwords, or check "
                "that your files match the selected language."
            )
            return {}

        # Language mismatch warning (decision 58).
        if trace.get("language_warning"):
            st.warning(trace["language_warning"])

        # Step 3: Build DTM.
        st.write("Building document-term matrix...")
        dtm, vectorizer, dtm_info = build_dtm(
            processed_texts,
            min_df=params["min_df"],
            max_df=params["max_df"],
        )

        # Clamp n_topics to doc count (decision 164).
        n_topics = min(params["n_topics"], len(processed_texts) // 2)
        if n_topics < NUM_TOPICS_MIN:
            n_topics = NUM_TOPICS_MIN
        if n_topics != params["n_topics"]:
            st.warning(
                f"Topic count adjusted to {n_topics} (corpus has only "
                f"{len(processed_texts)} documents)."
            )

        # Step 4: Run LDA.
        st.write("Fitting LDA model...")
        model, doc_topic_matrix, topics, model_info = run_lda(
            dtm=dtm,
            vectorizer=vectorizer,
            n_topics=n_topics,
            n_words=params["n_words"],
            random_seed=params["seed"],
            max_iter=params["max_iter"],
        )

        if model_info["convergence_warning"]:
            st.warning(model_info["convergence_warning"])

        # Step 5: Coherence.
        st.write("Computing coherence scores...")
        coherence = compute_coherence(topics, processed_texts)

        # Step 6: Lemma counts for overview.
        st.write("Finalising results...")
        all_lemmas: list[str] = []
        for t in processed_texts:
            all_lemmas.extend(t.split())
        lemma_counts = dict(Counter(all_lemmas))

        # File boundaries for diachronic chart.
        file_boundaries: list[int] = []
        if len(texts) > 1:
            idx = 0
            for text_entry in texts:
                # Count how many doc_labels belong to this file.
                prefix = text_entry["filename"].rsplit(".", 1)[0]
                count = sum(1 for lb in doc_labels if lb == prefix or lb.startswith(prefix + "_"))
                idx += count
                if idx < len(doc_labels):
                    file_boundaries.append(idx)

        status.update(label="Analysis complete", state="complete")

    return {
        "model": model,
        "dtm": dtm,
        "vectorizer": vectorizer,
        "processed_texts": processed_texts,
        "doc_labels": doc_labels,
        "doc_topic_matrix": doc_topic_matrix,
        "topics": topics,
        "model_info": model_info,
        "dtm_info": dtm_info,
        "coherence": coherence,
        "preprocessing_trace": trace,
        "lemma_counts": lemma_counts,
        "file_boundaries": file_boundaries,
        "params": params,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT TABS
# ═══════════════════════════════════════════════════════════════════════════════


def _render_results(results: dict[str, Any]) -> None:
    """Render the seven result tabs (decision 5)."""
    tab_names = [
        "Overview", "Topics", "Topic Map",
        "Heatmap", "Distribution", "Preprocessing", "Export",
    ]
    tabs = st.tabs(tab_names)

    topic_labels = [t["label"] for t in results["topics"]]

    with tabs[0]:
        _tab_overview(results, topic_labels)
    with tabs[1]:
        _tab_topics(results)
    with tabs[2]:
        _tab_topic_map(results)
    with tabs[3]:
        _tab_heatmap(results, topic_labels)
    with tabs[4]:
        _tab_distribution(results, topic_labels)
    with tabs[5]:
        _tab_preprocessing(results)
    with tabs[6]:
        _tab_export(results)


# ── Overview Tab ──────────────────────────────────────────────────────────────


def _tab_overview(results: dict[str, Any], topic_labels: list[str]) -> None:
    """Overview tab: coherence, DTM info, top lemmas (decisions 5, 41, 92, 103)."""
    coherence = results["coherence"]
    model_info = results["model_info"]
    dtm_info = results["dtm_info"]

    # Coherence display (decision 41).
    c_v = coherence["c_v"]
    cd = get_coherence_display(c_v)
    st.markdown(
        f'<div style="background:{cd["background"]};color:{cd["color"]};'
        f'padding:0.8rem 1rem;border-radius:0.4rem;margin-bottom:1rem;">'
        f'<strong>Coherence (C_v): {c_v:.4f} — {cd["label"]}</strong><br>'
        f'{cd["suggestion"]}</div>',
        unsafe_allow_html=True,
    )

    # Metrics row.
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Topics", model_info["n_topics"])
    col2.metric("Perplexity", f"{model_info['perplexity']:.1f}")
    col3.metric("Log-likelihood", f"{model_info['log_likelihood']:.1f}")
    col4.metric("Documents", dtm_info["n_docs"])

    # DTM transparency (decision 103).
    with st.container(border=True):
        st.markdown("**Document-Term Matrix**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Vocabulary (total)", f"{dtm_info['vocabulary_total']:,}")
        c2.metric("Vocabulary (kept)", f"{dtm_info['vocabulary_kept']:,}")
        c3.metric("Terms removed", f"{dtm_info['terms_removed']:,}")
        st.caption(
            f"DTM shape: {dtm_info['dtm_shape'][0]} documents "
            f"× {dtm_info['dtm_shape'][1]} terms | "
            f"min_df={dtm_info['min_df']}, max_df={dtm_info['max_df']}"
        )

    # Per-topic coherence.
    if coherence.get("per_topic"):
        with st.expander("Per-topic coherence"):
            for label, val in zip(topic_labels, coherence["per_topic"]):
                st.text(f"{label}: {val:.4f}")

    # Top lemmas (decision 92).
    st.markdown("**Top Lemmas (pre-LDA)**")
    chart = create_top_lemmas_chart(results["lemma_counts"])
    st.altair_chart(chart, use_container_width=False)


# ── Topics Tab ────────────────────────────────────────────────────────────────


def _tab_topics(results: dict[str, Any]) -> None:
    """Topics tab: selector, bar/wordcloud, table, excerpts (decisions 15, 33, 39, 51, 93, 149)."""
    topics = results["topics"]

    # Topic interpretation guide (decision 39).
    with st.expander("How to interpret topics"):
        st.markdown(
            "Each topic is a cluster of words that tend to co-occur in "
            "your corpus. The **weight** indicates how strongly a word "
            "is associated with the topic.\n\n"
            "- Look for **thematic coherence** — do the top words suggest "
            "a recognisable theme?\n"
            "- Check the **representative excerpt** to see how the topic "
            "appears in context.\n"
            "- If a topic seems noisy, try adding stopwords or adjusting "
            "POS filters."
        )

    # Topic selector (decision 149).
    selected_label = st.selectbox(
        "Select topic",
        options=[t["label"] for t in topics],
    )
    topic = next(t for t in topics if t["label"] == selected_label)
    color = get_topic_color(topic["topic_id"])

    # Bar chart / wordcloud toggle (decision 15).
    view = st.radio(
        "View", ["Bar chart", "Word cloud"], horizontal=True,
        label_visibility="collapsed",
    )

    if view == "Bar chart":
        chart = create_topic_bars(topic, color=color)
        st.altair_chart(chart, use_container_width=False)
    else:
        fig = create_wordcloud(topic, color=color)
        st.pyplot(fig, use_container_width=False)

    # Word + weight table (decision 51).
    with st.expander("Word details"):
        import pandas as pd

        df = pd.DataFrame({
            "Rank": range(1, len(topic["words"]) + 1),
            "Word": topic["words"],
            "Weight": [f"{w:.6f}" for w in topic["weights"]],
        })
        st.dataframe(df, hide_index=True, height=400)

    # Representative excerpt (decision 93).
    with st.expander("Representative document excerpt"):
        _show_representative_excerpt(results, topic)


def _show_representative_excerpt(
    results: dict[str, Any], topic: dict[str, Any]
) -> None:
    """Show the document with the highest weight for this topic."""
    tid = topic["topic_id"] - 1  # 0-indexed in the matrix
    matrix = np.asarray(results["doc_topic_matrix"])
    labels = results["doc_labels"]
    processed = results["processed_texts"]

    if tid < matrix.shape[1]:
        best_doc = int(np.argmax(matrix[:, tid]))
        st.markdown(f"**{labels[best_doc]}** (weight: {matrix[best_doc, tid]:.4f})")
        # Show first 200 words of the processed text.
        words = processed[best_doc].split()
        st.text(" ".join(words[:200]) + ("..." if len(words) > 200 else ""))
    else:
        st.info("No representative document available.")


# ── Topic Map Tab ─────────────────────────────────────────────────────────────


def _tab_topic_map(results: dict[str, Any]) -> None:
    """Topic Map tab: pyLDAvis or fallback (decision 16, lazy load 69)."""
    st.markdown("**Topic Map**")
    st.caption("Interactive visualisation of topic distances and prevalence.")

    result = create_topic_map(
        results["model"], results["dtm"], results["vectorizer"]
    )

    if isinstance(result, str):
        # pyLDAvis HTML.
        import streamlit.components.v1 as components
        components.html(result, height=800, scrolling=True)
    else:
        # Altair fallback.
        st.altair_chart(result, use_container_width=False)


# ── Heatmap Tab ───────────────────────────────────────────────────────────────


def _tab_heatmap(results: dict[str, Any], topic_labels: list[str]) -> None:
    """Heatmap tab (decisions 145, 150)."""
    chart = create_heatmap(
        np.asarray(results["doc_topic_matrix"]),
        results["doc_labels"],
        topic_labels,
    )
    st.altair_chart(chart, use_container_width=False)

    # Download buttons (gracefully skip if vl-convert is unavailable).
    try:
        png = export_figure_png(chart)
        svg = export_figure_svg(chart)
    except Exception:
        png, svg = None, None
    if png or svg:
        col1, col2 = st.columns(2)
        if png:
            with col1:
                st.download_button(
                    "Download PNG",
                    data=png,
                    file_name="heatmap.png",
                    mime="image/png",
                    key="heatmap_png",
                )
        if svg:
            with col2:
                st.download_button(
                    "Download SVG",
                    data=svg,
                    file_name="heatmap.svg",
                    mime="image/svg+xml",
                    key="heatmap_svg",
                )


# ── Distribution Tab ─────────────────────────────────────────────────────────


def _tab_distribution(results: dict[str, Any], topic_labels: list[str]) -> None:
    """Distribution tab: stacked bar + diachronic (decisions 114, 115)."""
    matrix = np.asarray(results["doc_topic_matrix"])
    labels = results["doc_labels"]

    _MAX_DIST_DOCS = 50

    st.markdown("**Document-Topic Distribution**")

    if len(labels) > _MAX_DIST_DOCS:
        # Show top-50 documents by dominant topic weight for performance.
        dominant_weights = matrix.max(axis=1)
        top_idx = np.argsort(dominant_weights)[::-1][:_MAX_DIST_DOCS]
        top_idx = np.sort(top_idx)  # Preserve document order.
        matrix_sub = matrix[top_idx]
        labels_sub = [labels[i] for i in top_idx]
        st.info(
            f"Showing top {_MAX_DIST_DOCS} of {len(labels)} documents "
            f"(by dominant topic weight) for performance."
        )
        chart = create_distribution_chart(matrix_sub, labels_sub, topic_labels)
    else:
        chart = create_distribution_chart(matrix, labels, topic_labels)
    st.pyplot(chart)

    st.divider()

    st.markdown("**Diachronic View**")
    st.caption("Topic weights across document order. Dashed lines mark file boundaries.")
    diachr = create_diachronic_chart(
        matrix, labels, topic_labels,
        file_boundaries=results.get("file_boundaries"),
    )
    st.pyplot(diachr)


# ── Preprocessing Tab ────────────────────────────────────────────────────────


def _tab_preprocessing(results: dict[str, Any]) -> None:
    """Preprocessing tab: summary + detailed token table (decisions 17, 59)."""
    trace = results["preprocessing_trace"]

    # Summary (decision 17).
    with st.container(border=True):
        st.markdown("**Preprocessing Summary**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Original tokens", f"{trace['original_tokens']:,}")
        col2.metric("After stopwords", f"{trace['after_stopwords']:,}")
        col3.metric("Final lemmas", f"{trace['final_lemmas']:,}")

        c1, c2 = st.columns(2)
        c1.metric("Unique lemmas", f"{trace['unique_lemmas']:,}")
        c2.metric("Documents/chunks", f"{trace['chunks_created']:,}")

    # Stopword transparency (decision 59).
    with st.container(border=True):
        st.markdown("**Stopword Removal**")
        st.text(
            f"Stopwords removed: "
            f"{trace['stopwords_removed_builtin'] + trace['stopwords_removed_custom']:,} "
            f"(built-in: {trace['stopwords_removed_builtin']:,}, "
            f"custom: {trace['stopwords_removed_custom']:,})"
        )
        if trace["custom_stopwords"]:
            st.caption(f"Custom: {', '.join(trace['custom_stopwords'])}")

    if trace.get("empty_chunks_removed", 0) > 0:
        st.info(
            f"{trace['empty_chunks_removed']} empty chunks were "
            "silently removed during processing."
        )

    # Warnings.
    for w in trace.get("warnings", []):
        st.warning(w)

    # Detailed per-document trace (decision 17).
    with st.expander("Per-document details"):
        import pandas as pd

        rows = []
        for d in trace.get("per_document", []):
            rows.append({
                "Document": d["label"],
                "Tokens": d["original_tokens"],
                "Stopwords (built-in)": d["stopwords_builtin"],
                "Stopwords (custom)": d["stopwords_custom"],
                "POS matched": d["pos_matched"],
                "Final lemmas": d["final_lemmas"],
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=400)

    # Token-level detail (very detailed).
    with st.expander("Token-level detail"):
        import pandas as pd

        doc_options = [d["label"] for d in trace.get("per_document", [])]
        if doc_options:
            sel = st.selectbox("Select document", options=doc_options, key="token_doc")
            doc_trace = next(
                (d for d in trace["per_document"] if d["label"] == sel), None
            )
            if doc_trace and doc_trace.get("token_details"):
                st.dataframe(
                    pd.DataFrame(doc_trace["token_details"]),
                    hide_index=True,
                    height=400,
                )


# ── Export Tab ────────────────────────────────────────────────────────────────


def _tab_export(results: dict[str, Any]) -> None:
    """Export tab: ZIP, PDF, individual downloads (decisions 45, 108, 117)."""
    params = results["params"]

    st.markdown("**Export Results**")

    # Main ZIP (decision 45).
    zip_bytes = export_zip(results, params)
    zip_name = get_zip_filename(params["language"], params["n_topics"])
    st.download_button(
        "Download ZIP (all files)",
        data=zip_bytes,
        file_name=zip_name,
        mime="application/zip",
        key="export_zip",
    )

    st.divider()

    # PDF report.
    try:
        pdf_bytes = export_pdf_report(results, params)
        st.download_button(
            "Download PDF report",
            data=pdf_bytes,
            file_name="lemmata_report.pdf",
            mime="application/pdf",
            key="export_pdf",
        )
    except RuntimeError as exc:
        st.warning(str(exc))

    st.divider()

    # Individual CSV/JSON downloads.
    st.markdown("**Individual files**")
    col1, col2 = st.columns(2)

    with col1:
        # Re-generate CSVs from the ZIP helper functions.
        from lemmata.file_io import (
            _topics_to_csv,
            _doc_topic_to_csv,
            _preprocessing_to_csv,
        )

        st.download_button(
            "topic_words.csv",
            data=_topics_to_csv(results["topics"]),
            file_name="topic_words.csv",
            mime="text/csv",
            key="dl_topic_words",
        )
        st.download_button(
            "doc_topic_matrix.csv",
            data=_doc_topic_to_csv(
                results["doc_topic_matrix"],
                results["doc_labels"],
                results["topics"],
            ),
            file_name="doc_topic_matrix.csv",
            mime="text/csv",
            key="dl_doc_topic",
        )

    with col2:
        st.download_button(
            "preprocessing_summary.csv",
            data=_preprocessing_to_csv(results["preprocessing_trace"]),
            file_name="preprocessing_summary.csv",
            mime="text/csv",
            key="dl_preproc",
        )
        env_info = get_environment_info(params)
        from lemmata.file_io import _to_json

        st.download_button(
            "environment.json",
            data=_to_json(env_info),
            file_name="environment.json",
            mime="application/json",
            key="dl_env",
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Application entry point."""
    params = _render_sidebar()
    texts = _render_upload()

    has_files = len(texts) > 0

    # Run Analysis button in sidebar (decision 55).
    with st.sidebar:
        run_clicked = st.button(
            "\u25b6\ufe0f Run Analysis",
            type="primary",
            use_container_width=True,
            disabled=not has_files,
        )

    if not has_files and not st.session_state.get("analysis_run"):
        _render_welcome()
        return

    if run_clicked:
        # Clear previous results (decision 44).
        if st.session_state.get("analysis_run"):
            st.info("Previous results cleared. Download them first if needed.")
        st.session_state["results"] = None
        st.session_state["analysis_run"] = False

        try:
            results = _run_analysis(texts, params)
            if results:
                st.session_state["results"] = results
                st.session_state["params"] = params
                st.session_state["analysis_run"] = True
                st.rerun()
        except Exception as exc:
            st.error(
                f"Analysis failed: {exc}\n\n"
                "Try different parameters or check your input files."
            )
            with st.expander("Technical details"):
                st.code(traceback.format_exc())

    if st.session_state.get("results"):
        _render_results(st.session_state["results"])


main()
