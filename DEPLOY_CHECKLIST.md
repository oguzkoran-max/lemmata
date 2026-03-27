# Lemmata v0.1.0 — Deploy Checklist

10-point acceptance test before release (ARCHITECTURE.md decision 192).

- [ ] **1. All tests pass** — `pytest` exits 0, including determinism test
- [ ] **2. Sample data works** — upload `examples/` texts, run analysis, verify all 7 tabs render
- [ ] **3. PDF report generates** — download PDF from Export tab, verify cover + params + topics + environment
- [ ] **4. ZIP export works** — download ZIP, verify all 6 files (topic_words.csv, doc_topic_matrix.csv, preprocessing_summary.csv, metrics.json, analysis_results.json, environment.json)
- [ ] **5. All 5 languages** — run at least one analysis per language (it, en, de, fr, es)
- [ ] **6. Landing page ready** — lemmata.app resolves, links to Streamlit Cloud app
- [ ] **7. README complete** — installation, usage, citation, authors, license sections present
- [ ] **8. CITATION.cff valid** — `cffconvert --validate` passes
- [ ] **9. Zenodo ready** — GitHub release triggers Zenodo DOI minting
- [ ] **10. Version number correct** — config.py, pyproject.toml, sidebar footer all show v0.1.0
