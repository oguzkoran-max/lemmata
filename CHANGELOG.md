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
- Gibbs runs report how much the log-likelihood per word changed in the
  second half of the run (a straight-line fit over the values recorded
  every 10 iterations), in the results and in `metrics.json`. They get no
  converged/not-converged verdict: with MALLET's defaults the
  log-likelihood still rises slowly at 1000 iterations (0.3–1.2% over
  iterations 500–1000 in 31 of 32 runs of the validation study), so a
  fixed threshold would either warn on nearly every run or miss the rise.
- "Check topic stability" in the Overview tab (P046). The model is fitted
  10 times with seeds that follow the user's seed, and each topic shows in
  how many runs a topic sharing at least 5 of its top 9 words appears; the
  median over topics (R_n) is given for 1 to 9 shared words. The measure
  and the 10 runs follow Agrawal, Fu and Menzies (2018, *Information and
  Software Technology* 98: 74–88); runs differ only in the seed, as in
  Rieger et al. (2024, *PeerJ Computer Science* 10: e2279).
- The Document-Term Matrix box reports how many terms were removed as
  rare and as common, their share of the vocabulary and of all word
  occurrences, and any documents left with no words (P047). The same
  figures are in `metrics.json`.

### Changed
- Vocabulary pruning follows the standard that Maier et al. (2020,
  *Computational Communication Research* 2: 139–152) report for topic
  models in the social sciences: terms in fewer than 0.5% or in more than 99% of the documents
  are removed (P047). The earlier thresholds (decision 25: at least 1 to
  10 documents and at most 95% to 75% of them, by corpus size) had no
  source. Both thresholds are set as percentages under Advanced; the
  "Auto-adjust by corpus size" checkbox is gone, since shares of the
  documents already follow the corpus size.
- The Language help text says what spaCy's reported lemma accuracy
  (97.15% for Italian) was measured on, and gives the authors'
  measurement on older literary Italian: about 87% of nouns, proper
  nouns and adjectives received the correct lemma in Manzoni's *I
  promessi sposi* (1840) and about 76% in Dante's *Commedia* (UD gold
  standards, CoNLL 2018 evaluation script) (P047).
- Default number of topics raised from 5 to 10, the default of MALLET and
  scikit-learn (P046). The help text says that no number is right for
  every corpus.
- The coherence sweep ("Compare numbers of topics", formerly "Find
  optimal number of topics") no longer announces an optimal number. It
  reports where C_v is highest and asks the user to compare topics,
  because automated coherence does not reliably agree with human
  judgements (Hoyle et al. 2021; Chang et al. 2009) (P046).
- The coherence box no longer labels C_v as Good, Moderate or Low (cut-offs
  0.5 and 0.3, decision 41). The cut-offs had no source; the box now
  explains what C_v measures (P046, wording P048).
- Topic count upper limit raised from 15 to 30.
- Chunk size minimum lowered from 300 to 200 words, so that a single file
  can be split into documents of the length Cooper et al. (2021) used for
  the Decameron (P048). Chunks still end at sentence boundaries.
- Help texts for chunk size, words per topic, part-of-speech filter,
  custom stopwords and random seed now say what each setting does and on
  what the default rests, with sources (P048).
- Variational method: perplexity is checked every 10 iterations and
  fitting stops once it improves by less than 0.1 (scikit-learn's own
  criterion). A fit that reaches the iteration limit is identical to
  earlier versions; a fit that converges earlier now stops there.
- Topic Map numbers topics in the same order as the topic list
  (by prevalence).

### Fixed
- C_v was computed on at most 20 words per topic (gensim's default) even
  when more were shown, and the coherence sweep always used 10 words. Both
  now use every word shown (P048).
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
- A variational fit that stopped early reported one iteration too few:
  scikit-learn's `n_iter_` does not count the iteration whose perplexity
  check stopped the fit.
- The app's GitHub links (About, bug report, footer, sample data) pointed
  to a repository address that does not exist; they now point to
  github.com/oguzkoran-max/lemmata.
- "How to cite" in the app cited an article in *Digital Scholarship in
  the Humanities* that has not been published, with the authors in the
  wrong order. It now cites the software (Zenodo DOI), as README and
  CITATION.cff do. The About text and `pyproject.toml` list the authors
  as Koran, Yücesan, Cangır.

### Known issues
- pyLDAvis passes a scikit-learn MDS parameter scheduled for removal in
  scikit-learn 1.10; with that version the map falls back to the PCA chart.
- tomotopy 0.14.0 raises `IndexError` when indexing `model.docs` beyond
  the first document; Lemmata iterates instead.
- Lemmatisation still uses modern-language spaCy models; historical
  forms may receive wrong lemmas (see the Language help text for the
  measured accuracy on older Italian).

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
