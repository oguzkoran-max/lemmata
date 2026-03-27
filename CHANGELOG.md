# Changelog

All notable changes to Lemmata will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — Unreleased

### Added
- Six-module architecture: config, preprocessing, modelling, visualisation, file_io, app
- Five-language support: Italian, English, German, French, Spanish (spaCy sm models)
- LDA topic modeling with scikit-learn (deterministic, random_state=42)
- C_v coherence scoring via Gensim CoherenceModel
- File upload: txt, pdf, docx, odt, epub with auto-encoding detection
- Text paste input alternative
- Sentence-boundary-respecting chunking for single-file uploads
- POS filtering with presets (Content words, Content + verbs, All open class, Custom)
- Custom stopword support with transparent counting
- Seven result tabs: Overview, Topics, Topic Map, Heatmap, Distribution, Preprocessing, Export
- Interactive Altair charts with consistent tableau10 colour palette
- Wordcloud generation via matplotlib
- pyLDAvis integration with PCA scatter fallback
- Document-topic heatmap (viridis)
- Diachronic topic weight trend chart with file boundary markers
- Export: ZIP package (CSV + JSON), PDF report (reportlab), PNG 300 DPI, SVG
- Full preprocessing trace with per-document and per-token detail
- DTM transparency: vocabulary total/kept/removed, matrix dimensions
- Coherence display: colour-coded (green/yellow/red) with action suggestions
- Welcome screen with 3-step guide and topic modeling explainer
- Streamlit theme: teal primary (#0F6E56), white/light background, sans-serif
- Prompt log system for vibe coding documentation

### Changed
- Nothing yet (initial release)

### Fixed
- Nothing yet (initial release)
