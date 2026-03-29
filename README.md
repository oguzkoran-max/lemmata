# Lemmata

![CI](https://github.com/oguzkoran-max/lemmata/actions/workflows/ci.yml/badge.svg) ![License: MIT](https://img.shields.io/badge/License-MIT-green.svg) ![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-deployed-FF4B4B.svg)

**A multilingual LDA topic modeling platform for humanities researchers.**

[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

---

## What is Lemmata?

Lemmata is a browser-based tool that lets humanities researchers perform LDA (Latent Dirichlet Allocation) topic modeling on literary and historical texts — without writing a single line of code.

Upload your texts, choose your language, adjust parameters, and download reproducible results. Everything runs in the browser.

**Supported languages:** Italian, English, German, French, Spanish

---

## Quick start

No installation required. Open the platform in your browser:

**[lemmata.app](https://lemmata.app)**

1. Select your language and POS filter in the sidebar.
2. Upload one or more files (.txt, .pdf, .odt, .docx, .epub) or paste text directly.
3. Click **Run Analysis**.
4. Explore the results across seven tabs (Overview, Topics, Topic Map, Heatmap, Distribution, Preprocessing, Export).
5. Download all outputs as a ZIP file or generate a PDF report.

---

## Features

| Feature | Description |
|---|---|
| **Multilingual NLP** | Five languages with dedicated spaCy pipelines and language-specific stopwords |
| **POS filtering** | Presets (content words, content + verbs, all open classes) or custom selection |
| **Custom stopwords** | Add domain-specific stopwords on top of the built-in lists |
| **Coherence scoring** | C_v coherence metric with interpretive guidance (good / fair / weak) |
| **Interactive charts** | Altair-based interactive visualizations with hover details |
| **Topic Map** | pyLDAvis visualization with graceful fallback |
| **Preprocessing trace** | Token-level table showing every step (original, lemma, kept/removed, reason) |
| **Deterministic results** | random_state=42 and batch learning ensure identical results on repeated runs |
| **PDF report** | Auto-generated analysis report suitable for course assignments or publications |
| **ZIP export** | All outputs (CSV, JSON, PNG, SVG, environment report) in one download |

---

## Local installation

### 1. Clone the repository

```bash
git clone https://github.com/oguzkoran-max/lemmata.git
cd lemmata
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download spaCy language models

```bash
python -m spacy download it_core_news_sm
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
```

### 5. Run the application

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

---

## Architecture

Lemmata follows a modular design with strict separation of concerns. The UI layer (`app.py`) contains no business logic.

```
lemmata/
├── app.py              # Streamlit UI only
├── config.py           # Constants, language configs, slider ranges
├── preprocessing.py    # spaCy NLP pipeline, POS filtering, lemmatization, trace
├── modelling.py        # LDA (scikit-learn), coherence (Gensim), corpus stats
├── visualisation.py    # Charts, wordclouds, pyLDAvis (zero st.* calls)
├── file_io.py          # File readers, ZIP/PDF export, environment report
├── requirements.txt    # Pinned dependencies
├── ARCHITECTURE.md     # 200 design decisions
├── prompts/            # Vibe coding development logs
└── tests/              # Pytest test suite
```

**Design decisions:**
- scikit-learn LDA over Gensim: deterministic output with random_state=42.
- Gensim CoherenceModel used independently for C_v evaluation.
- Language configurations reviewed and approved by a corpus linguistics specialist (Doc. Dr. Hakan Cangir).

---

## Reproducibility

- **Fixed random state:** random_state=42 across all stochastic processes.
- **Batch learning:** learning_method='batch' eliminates document-order effects.
- **Environment report:** Every analysis exports Python version, package versions, and all parameters.
- **Same data + same parameters = same results.** Guaranteed and verifiable.

---

## How it was built: vibe coding

Lemmata was developed entirely through **vibe coding** — a researcher with no programming expertise communicated requirements to a large language model exclusively in natural language, and the LLM generated all code. Development was carried out using [Claude Code](https://docs.anthropic.com/en/docs/claude-code) (Anthropic).

Three safeguards ensured methodological rigor:

1. **Full prompt-response documentation.** Every prompt and response is archived in the [`prompts/`](prompts/) directory and published as supplementary material.

2. **Expert validation at every decision point.** Domain-specific decisions followed a human-in-the-loop model: corpus linguistics choices were reviewed by a specialist; literary interpretations were provided by a scholar. Technical code generation operated under a human-on-the-loop model: the LLM produced code, the principal investigator tested outputs.

3. **Benchmark validation.** Results are compared against MALLET using the same corpus and parameters.

An accompanying article is in preparation for submission to *Digital Scholarship in the Humanities* (Oxford University Press).

---

## Citation

```bibtex
@software{koran_lemmata_2026,
  author       = {Koran, Oğuz and Cangır, Hakan and Yücesan, Barış},
  title        = {Lemmata: A Multilingual LDA Topic Modeling Platform for Digital Humanities},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.XXXXXXX},
  url          = {https://github.com/oguzkoran-max/lemmata}
}
```

Click **"Cite this repository"** on the GitHub page for auto-generated citation via [`CITATION.cff`](CITATION.cff).

---

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

MIT License. See [`LICENSE`](LICENSE).

---

## Acknowledgments

Lemmata is developed at [Ankara University](https://www.ankara.edu.tr/), School of Foreign Languages (Italian Language and Literature) and Faculty of Languages, History and Geography — DTCF (Italian Language and Literature).

The entire codebase was generated through LLM-assisted development using [Claude](https://claude.ai/) and [Claude Code](https://docs.anthropic.com/en/docs/claude-code) by [Anthropic](https://www.anthropic.com/). Built with [spaCy](https://spacy.io/), [scikit-learn](https://scikit-learn.org/), [Gensim](https://radimrehurek.com/gensim/), [Altair](https://altair-viz.github.io/), and [Streamlit](https://streamlit.io/).
