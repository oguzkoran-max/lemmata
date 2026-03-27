# CLAUDE.md — Lemmata Project Configuration

## Identity
Lemmata is a browser-based, multilingual LDA topic modeling platform for humanities researchers. Developed entirely through vibe coding. DSH article in preparation.

## Authors
- Oğuz Koran — PI, all prompts, testing (Ankara University YDYO, Italian)
- Hakan Cangır — corpus linguistics validation (Ankara University YDYO)
- Barış Yücesan — literary interpretation (Ankara University DTCF, Italian)

## Critical Rules

### Architecture
- **NEW PROJECT from scratch.** Do NOT reference run_topic_model.py or any existing code.
- Six modules only: config.py, preprocessing.py, modelling.py, visualisation.py, file_io.py, app.py
- app.py = Streamlit UI only, ZERO business logic
- visualisation.py = ZERO st.* calls, returns figures/data only

### Technical (Hakan-approved)
- LDA: scikit-learn LatentDirichletAllocation (NOT Gensim)
- random_state=42, learning_method='batch' — deterministic
- Coherence: Gensim CoherenceModel C_v (evaluation only)
- NLP: spaCy sm models (it, en, de, fr, es)
- Default POS: NOUN, PROPN, ADJ
- No Portuguese. No Turkish (future work).
- Unigram only (bigram future work)
- CountVectorizer (no TF-IDF)

### UI
- English interface, i18n-ready architecture
- NO emoji in tab names, sidebar headings, or footer
- Emoji ONLY in: Run Analysis button, analysis complete message, coherence indicator
- 7 tabs: Overview, Topics, Topic Map, Heatmap, Distribution, Preprocessing, Export
- Tooltip + expander pedagogical system
- Altair interactive charts (not matplotlib except wordcloud)
- Theme: white/light, teal (#0F6E56) primary color, sans-serif font
- Progressive disclosure: simple defaults, advanced in expanders

### Files & Export
- Formats: txt, pdf, odt, docx, epub
- Auto-detect: 1 file → chunk, multiple → separate docs
- Chunking: word count target + sentence boundary respect
- Export: ZIP (main) + PDF report + individual downloads
- CSV: topic_words.csv, doc_topic_matrix.csv, preprocessing_summary.csv, metrics.json, analysis_results.json
- PNG 300 DPI + SVG for visuals
- ZIP naming: lemmata_{lang}_{n}topics_{date}_{time}.zip

### Error Handling
- Specific PDF error messages (protected, scanned, corrupted)
- Low token warning for language mismatch
- Empty chunk silent removal + preprocessing summary note
- Duplicate filename warning
- Topic > document count warning + slider limit
- File size limit: 50MB/file, 100MB total
- Encoding: auto-detect (chardet) with UTF-8 fallback
- Text hygiene: BOM, null bytes, control chars removal

### Performance
- @st.cache_resource for spaCy models
- @st.cache_data for LDA results
- Model loads on "Run Analysis" (not on page load)
- pyLDAvis lazy loading (when tab opened)
- Status updates during analysis (st.status)

## Prompt Logging (CRITICAL)
At the END of every session, create/update a log file in prompts/ with this format:

```
## PXXX — [short title]
- **Date:** YYYY-MM-DD
- **Category:** [architecture | implementation | bug-fix | refactoring | testing | documentation]
- **Prompt:** [exact prompt text]
- **Response summary:** [what was created/changed]
- **Decision:** [methodological decision, if any]
- **Validation:** [what was tested]
- **Files changed:** [list]
```

Log meaningful errors too. Maintain prompts/PROMPT_LOG_INDEX.md.

## Commit Messages
Format: `P001: short description` — links git history to prompt log.

## Reference
For detailed decisions (200 total), see ARCHITECTURE.md in this repository.
