# Changelog

All notable changes to Lemmata will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
