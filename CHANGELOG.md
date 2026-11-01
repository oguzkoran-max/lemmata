# Changelog

All notable changes to Lemmata will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — Unreleased

Changes follow a validation study that reproduced Cooper et al. (2021,
*Decameron*) with Lemmata v0.2.0 and with MALLET. See prompt log P045.

### Added
- Gibbs sampling inference (tomotopy), now the default method. Defaults
  are MALLET 2.0.8's (alpha sum 5, beta 0.01, 1000 iterations, burn-in
  200) with hyperparameter optimisation every 10 iterations, as in
  little-mallet-wrapper. They were fixed before any validation run.
  tomotopy optimises the document-topic prior only (MALLET also
  optimises beta).
- "Inference method" selector in Advanced settings. The variational
  method (scikit-learn), Lemmata's only method before this release,
  remains available to reproduce earlier results.
- Method, iterations used and priors in the results caption,
  `metrics.json` and the environment report (with the tomotopy version).

### Changed
- Topic count upper limit raised from 15 to 30.
- Variational method: perplexity is checked every 10 iterations and
  fitting stops once it improves by less than 0.1 (scikit-learn's own
  criterion). A fit that reaches the iteration limit is identical to
  earlier versions; a fit that converges earlier now stops there.
- Topic Map numbers topics in the same order as the topic list
  (by prevalence).

### Fixed
- The convergence warning appeared on every run, converged or not: it
  only tested whether the iteration limit had been reached.
- The preprocessing summary subtracted tokens on both stopword lists
  twice, so "After stopwords" could be smaller than "Final lemmas"; the
  "Stopwords removed" total had the same double count. Each token is now
  counted once and the overlap is shown.
- The Topic Map never showed pyLDAvis with pyLDAvis 3.4 or later
  (`pyLDAvis.sklearn` was removed; its default PCoA also fails to
  serialise with numpy 2), so the PCA fallback was always used. It now
  uses `pyLDAvis.prepare` with metric MDS.

### Known issues
- pyLDAvis passes a scikit-learn MDS parameter scheduled for removal in
  scikit-learn 1.10; with that version the map falls back to the PCA chart.
- tomotopy 0.14.0 raises `IndexError` when indexing `model.docs` beyond
  the first document; Lemmata iterates instead.
- Lemmatisation still uses modern-language spaCy models; historical
  forms may receive wrong lemmas.

## [0.1.0] — 2026-04-02

### Core Platform
- 6-module architecture: config, preprocessing, modelling, file_io, visualisation, app
- 5 language support (Italian, English, French, German, Spanish) via spaCy
- 7 analysis tabs: Overview, Topics, Topic Map, Heatmap, Distribution, Preprocessing, Export
- scikit-learn LDA with Gensim C_v coherence scoring
- Deterministic analysis with configurable random seed (default: 42)

### Text Processing
- 5 input formats: TXT, PDF, DOCX, ODT, EPUB
- Automatic chunking with sentence boundary respect
- POS filtering with presets (Content words, Content+verbs, All open, Custom)
- Custom stopword support
- Full preprocessing trace (summary, per-document, token-level detail)
- Language mismatch detection via stop-word recognition rate

### Visualization
- Topic bar charts and word clouds
- Topic Map (PCA scatter with bubble size)
- Document-topic heatmap (viridis)
- Topic distribution (matplotlib stacked bars)
- Diachronic view (topic weight trends with file boundaries)
- Top lemmas frequency chart (pre-LDA)
- Coherence sweep chart (optimal topic finder, k=2-15)

### Export
- Complete ZIP archive (CSV matrices, JSON, charts, environment info)
- PDF report (cover, parameters, topics, heatmap, excerpts)
- PNG 300 DPI and SVG chart exports
- Individual file downloads per tab

### User Experience
- Welcome screen with 3-step guide and "What is topic modeling?" explainer
- Topic label editing with propagation to all tabs and exports
- File preview (size, word count, first 200 characters)
- Duplicate file detection (filename + content fingerprint)
- Estimated analysis time display
- "What to do next?" guidance after analysis
- Topic interpretation guide in Topics tab
- Chunk size help text
- Convergence warning when model does not converge
- Per-file chunk count distribution in Preprocessing tab

### Safeguards
- Crash recovery with user-friendly error messages
- Topic-document safeguard (dynamic slider cap at n_docs/2)
- Imbalanced corpus warning (10x+ file size difference)
- Chunk imbalance note (3x+ difference)
- Empty file handling (skip + warning)
- Copyright notice and privacy note

### Infrastructure
- Deployed on Streamlit Cloud
- Landing page at lemmata.app (GitHub Pages)
- GitHub Actions CI (162 tests, pytest)
- UptimeRobot monitoring (landing page + app)
- OpenGraph and Twitter Card meta tags
- Custom favicon (teal lambda)
- Custom 404 page
- Scroll animations on landing page
- Powered By technology band
- 38 prompt logs documenting development process

### Development
- Built entirely through human-LLM dialogue (vibe coding methodology)
- 39 numbered prompts (P001-P039)
- 162 automated tests including determinism verification
- MIT License
