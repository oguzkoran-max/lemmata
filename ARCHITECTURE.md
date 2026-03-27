# Lemmata — Architecture Decisions (200 Questions)
**Date:** 27 March 2026
**Status:** Pre-development planning complete

This document records 200 architectural, design, and strategic decisions made through structured dialogue before any code was written. It serves as both a development reference and a methodological artifact for the accompanying DSH article.

---

## Core Architecture (Decisions 1-10)

1. **Interface language:** English v1, i18n-ready for v2
2. **Help system:** Tooltip (short, technical) + expander (pedagogical, collapsed by default)
3. **File formats:** txt, pdf, odt, docx, epub
4. **Upload mode:** Auto-detect — 1 file → chunk, multiple → separate documents
5. **Result tabs:** 7 — Overview (new), Topics, Topic Map, Heatmap, Distribution, Preprocessing, Export
6. **Auto topic suggestion:** Future version (sweep_coherence() infrastructure ready)
7. **Theme:** White/light, academic, clean
8. **Error handling:** Clear messages + empty result guide + progress bar + summary box
9. **Report:** Basic PDF (cover + params + summary + topics + heatmap + text excerpts + environment)
10. **Comparative analysis:** Future version (ZIP export for manual comparison)

## Parameters (11-20)

11. **Chunk size:** Slider 300-3000, default 1000, guide info box
12. **POS filter:** User selectable, default NOUN/PROPN/ADJ (Hakan-approved)
13. **Topic count:** Slider 2-15, default 5
14. **Stopwords:** spaCy built-in + user custom (no pre-built academic list)
15. **Topic visualization:** Bar chart (default) + wordcloud toggle
16. **pyLDAvis:** Optional, try-except + 2D scatter fallback
17. **Preprocessing trace:** Summary (default) + detailed token table in expander
18. **Metrics:** C_v in Overview; C_v + Perplexity + Log-likelihood in Corpus Stats
19. **Words per topic:** Slider 5-30, default 10
20. **Logo:** Simple typographic, lambda (λ) symbol + "Lemmata" + tagline, SVG

## Logging & Infrastructure (21-30)

21. **Log production:** Claude Code auto-generates, Oğuz verifies
22. **Error logging:** Meaningful errors logged (transparency)
23. **Log structure:** Per-session files + PROMPT_LOG_INDEX.md
24. **Citation in app:** Sidebar expander ("How to cite" + BibTeX)
25. **min_df/max_df:** Auto-adjust by corpus size + Advanced slider
26. **First visit:** Welcome screen + 3-step guide + TM explainer + sample data link
27. **Deploy:** Streamlit Cloud + local installation supported
28. **Data source:** File upload only (no URL fetching)
29. **User profiles:** Single mode, progressive disclosure
30. **Data storage:** In-memory during session, deleted on session end

## Detail Decisions (31-40)

31. **File size:** 50MB/file, 100MB total. No limit on local.
32. **Chunking:** Word count target + sentence boundary respect (spaCy sentence detection)
33. **Topic labels:** "Topic 1 (vita, morte, uomo)" + user-editable
34. **Chart interactivity:** Altair interactive (hover). Wordcloud matplotlib static.
35. **Analysis progress:** st.status() step-by-step updates
36. **Error language:** English v1 (i18n v2)
37. **Pre-spaCy cleaning:** Line-end hyphen joining + multi-space + Unicode NFC
38. **N-grams:** Unigram only. Bigram future (NGRAM_RANGE constant ready).
39. **Topic interpretation guide:** Expander in Topics tab ("How to interpret topics")
40. **PDF report content:** Cover + params + summary + topics + heatmap + text excerpts + environment

## Accessibility & Edge Cases (41-50)

41. **Coherence display:** Color-coded background (green/yellow/red) + text + action suggestion. No emoji.
42. **Document labels:** Filename without extension. Chunks: [filename]_001 format.
43. **Page title:** "Lemmata — Multilingual Topic Modeling", page_icon="📊"
44. **Re-run:** Previous results cleared + warning ("Download first if needed")
45. **Export options:** ZIP (main) + PDF report button + individual file downloads
46. **Environment report:** Full detail (version + packages + params + seed + corpus)
47. **Version display:** Sidebar footer "v0.1.0 · GitHub · MIT License"
48. **Navigation:** Hamburger menu hidden via CSS, sidebar-focused
49. **Analytics:** None (privacy)
50. **Testing:** Pytest, per-module, determinism test critical

## Advanced Technical (51-60)

51. **Topic word detail:** Bar chart hover + table (word + weight). Corpus frequency in expander.
52. **spaCy models:** sm only. lg future (MODEL_SIZE constant ready).
53. **Model loading:** On "Run Analysis", @st.cache_resource cached.
54. **Multi-file processing:** Each file preprocessed separately, merged in DTM.
55. **Analysis trigger:** Sidebar "Run Analysis" button (disabled without files).
56. **CSV export:** topic_words.csv + doc_topic_matrix.csv + preprocessing_summary.csv + metrics.json
57. **Caching:** @st.cache_resource (model) + @st.cache_data (results)
58. **Language mismatch:** Low token ratio (<10%) → automatic warning
59. **Stopword transparency:** Preprocessing summary shows "Stopwords removed: N (built-in: X, custom: Y)"
60. **Feedback:** GitHub Issues, sidebar link "Report a bug · Request a feature"

## Platform Quality (61-70)

61. **Accessibility:** Color-blind-friendly palette (viridis/tableau10) + table alternatives
62. **Mobile:** Streamlit default responsive + small screen info note
63. **Mixed language:** Not supported, single language enforced, info box explanation
64. **Empty file:** Skip + warning, continue with remaining files
65. **File preview:** Metadata (size, word count) + first 200 words
66. **Color palette:** Categorical tableau10, heatmap viridis
67. **Visual export:** PNG 300 DPI + SVG (wordcloud PNG only)
68. **Session state:** st.session_state preserves results; F5 clears + warning note
69. **Tab loading:** Core tabs immediate, pyLDAvis lazy
70. **Versioning:** Semantic (0.1.0), CHANGELOG.md, Zenodo concept DOI

## Svevo & DSH (71-76) — Deferred to article phase

71-76. Svevo pilot design, text cleaning, figures, supplementary material, MALLET comparison — to be decided during article writing phase.

## Deployment & Strategy (77-90)

77. **Streamlit Cloud:** Free plan + lemmata.app landing page redirect
78. **Pre-deploy check:** Detailed per-module test + deploy checklist
79. **Documentation:** README sufficient for v1
80. **CI/CD:** GitHub Actions pytest on push, Streamlit auto-deploy on main
81. **Sample data:** examples/ folder with short public domain Italian texts
82. **Feature flags:** None, all features active in v1
83. **Configuration:** All from config.py (no .env, no CLI args)
84. **Application logging:** Python logging module, console only, INFO default
85. **Vibe coding docs:** README "How it was built" + prompts/ folder
86. **Academic usage guide:** Short expander in welcome screen
87. **Landing page:** Simple GitHub Pages single page
88. **Prompt log privacy:** Public, no sensitive info
89. **Repo timing:** Public as soon as platform ready
90. **Release notes:** Detailed, structured, Zenodo reads this

## Interaction Details (91-100)

91. **Upload UX:** Streamlit multi-uploader + text paste area (expander)
92. **Pre-LDA analysis:** Top 20 frequent lemmas bar chart in Overview
93. **Topic-text matching:** Representative document excerpt per topic. Color highlighting v2.
94. **About section:** Sidebar expander (who, how, why, GitHub link)
95. **POS presets:** Dropdown (Content words / Content+verbs / All open / Custom) + multiselect
96. **Visual customization:** None in v1. Users edit exported SVG.
97. **Vectorization:** CountVectorizer only. TF-IDF future (VECTORIZER_TYPE constant).
98. **Security:** File type validation, size limit, no st.markdown unsafe_allow_html
99. **Memory:** All text in memory (50MB limit prevents issues). Large corpus → local install.
100. **Extension vision:** method parameter in modelling.py for future NMF/BERTopic

## Rakip Analizi & USP (101-110)

101. **Competitors:** MALLET (CLI, no GUI), Voyant (no LDA), Gensim (code required). Lemmata unique: preprocessing trace + no-code + deterministic + documented development.
102. **USP:** "Browser-based topic modeling where you see exactly what happened to every word."
103. **DTM transparency:** Overview shows vocabulary size, terms removed by min_df/max_df, final DTM dimensions.
104. **Document length imbalance:** Informational only (show chunk counts per document).
105. **spaCy error tolerance:** Per-token try-except, log issues, continue.
106. **Minimum corpus:** Warning if <50 unique lemmas, no blocking.
107. **Analysis history:** Last analysis only. Previous → ZIP download.
108. **Download points:** Per-tab download icons + Export tab ZIP.
109. **Tab customization:** None, fixed 7 tabs.
110. **Sub-corpus:** Future version (metadata-based filtering).

## Visual Details (111-120) — NO EMOJI except Run button, analysis message, coherence indicator

111. **Wordcloud shape:** Rectangle, white background, 800x400px.
112. **Tokenization:** spaCy default sufficient. Italian contractions handled correctly.
113. **Chunk overlap:** None. Sentence-boundary chunking sufficient.
114. **Document classification:** Dominant topic auto-assigned, shown in Distribution.
115. **Diachronic view:** Topic weight trend line chart (X=doc order, Y=weight, vertical lines at file boundaries).
116. **Onboarding:** FAQ expander sufficient (no interactive tutorial).
117. **Download mechanism:** st.download_button with key parameter + cache protection.
118. **Additional metrics:** v1 has C_v/Perplexity/Log-likelihood. Topic diversity future.
119. **Machine-readable export:** analysis_results.json in ZIP.
120. **Tool integration:** Standard CSV/JSON output. No tool-specific formats.

## Seed, Performance, Ordering (121-130)

121. **Seed control:** Editable in Advanced (number input), default 42.
122. **Topic ordering:** By corpus prevalence (descending average weight).
123. **Performance target:** Under 30 seconds for typical analysis.
124. **Convergence:** max_iter in Advanced + warning if model used all iterations.
125. **Topic color identity:** Fixed color per topic (tableau10), consistent across all tabs.
126. **Lemmatization quality:** spaCy default. Wrong lemmas → add to custom stopwords.
127. **Screen layout:** Streamlit wide layout, no extra responsive work.
128. **Imbalanced corpus warning:** 10x size difference → warning with suggestion.
129. **Parameter reset:** "Reset to defaults" link in sidebar.
130. **Analysis naming:** Auto: lemmata_{lang}_{n}topics_{date}_{time}.zip

## Visual Design (131-140)

131. **Logo:** Lambda (λ) symbol + "Lemmata" (teal, medium weight) + "Multilingual Topic Modeling" (gray, small).
132. **Typography:** "sans serif" via config.toml (system default).
133. **Background:** Light gray #F8F9FA main, slightly darker #F0F2F6 sidebar.
134. **Primary color:** Teal #0F6E56 (buttons, sliders, active tabs).
135. **Run button:** st.button type="primary", use_container_width=True. Disabled without files.
136. **Tab design:** Streamlit default horizontal. NO emoji in tab names.
137. **Analysis complete:** st.success green box + short summary (topics, C_v, lemmas). No emoji.
138. **Info boxes:** st.info (blue background, left border) for pedagogical content.
139. **Spacing:** Streamlit default + st.divider between sections + st.container(border=True) for grouping.
140. **File upload area:** Streamlit default uploader + explanatory text above and below.

## Sidebar Layout (141-150)

141. **Sidebar sections:** Bold headings + st.divider between. Logo top, Run button middle, footer bottom.
142. **Sliders:** Streamlit default with help="..." tooltip parameter.
143. **POS filter:** Preset dropdown + multiselect below.
144. **Chart style:** White background, minimal axes, light grid, sans-serif, tableau10 colors.
145. **Heatmap values:** Hover tooltip (Altair). No numbers in cells.
146. **Chart sizes:** Variable by type. Bar 600px, heatmap full width, wordcloud 600x400.
147. **Long tables:** st.dataframe height=400, virtual scrolling, sortable/filterable.
148. **Wordcloud info:** Topic name above, nothing below. Download via tab-level icon.
149. **Topics layout:** Topic selector (dropdown/buttons) → one topic at a time, full area.
150. **Heatmap size:** Dynamic height=max(200, n_docs*25), dynamic width=max(300, n_topics*60).

## Landing Page & Strategy (151-160)

151. **Landing page structure:** Single page, scroll sections, anchor nav.
152. **Landing page content:** Hero (logo + tagline + screenshot + Launch button) → Features → How it works → Citation → Footer.
153. **Landing page design:** Same teal color, consistent identity, more whitespace.
154. **SEO:** Basic meta tags + OpenGraph. Google Scholar via DSH article.
155. **Success metrics:** GitHub stars + Zenodo downloads + citations. Realistic expectations.
156. **Promotion:** DSH article + DH conference poster + academic network sharing.
157. **Monitoring:** UptimeRobot free (5-min checks, email alerts).
158. **Landing page language:** English only.
159. **Landing page tone:** Academic + open-source: "Topic modeling for the humanities — no code required."
160. **Landing page links:** Launch Lemmata, Source code, Cite, Paper.

## Error Recovery & Edge Cases (161-170)

161. **Timeout:** Estimated time display before analysis. Warning if >60s expected.
162. **PDF errors:** Three specific messages (protected, scanned/image, corrupted).
163. **Duplicate files:** Same filename warning, no blocking.
164. **Topic > document:** Warning + dynamic slider max = document_count / 2.
165. **Character set:** Full UTF-8 support. Non-Latin scripts tokenized but not lemmatized.
166. **Empty chunks:** Silently removed, noted in preprocessing summary.
167. **Encoding:** Auto-detect (chardet) → UTF-8 → Latin-1 fallback. Shown in trace.
168. **Missing spaCy model:** Auto-download attempt, clear error message if fails.
169. **Session recovery:** URL query params preserve parameters (not files).
170. **Text hygiene:** Auto-clean BOM, null bytes, control chars, normalize line endings.

## Maintenance & Sustainability (171-200)

171. **Powered by:** Landing page footer only: "Built with spaCy, scikit-learn, Gensim, Streamlit"
172. **Legal:** Short privacy note in About section and landing page footer. No separate ToS.
173. **Copyright notice:** Small note under file upload: user responsible for upload rights.
174. **Maintenance:** 6-month check + critical fixes on demand.
175. **Dependency security:** GitHub Dependabot enabled.
176. **Maintenance responsibility:** Oğuz primary. Community contributions welcome.
177. **Ethics statement:** DSH article Discussion section, not README.
178. **Data processing:** "Texts processed on server during session. No permanent storage. No third-party sharing."
179. **Git branches:** Direct push to main. Feature branches when multiple contributors.
180. **Commit messages:** "P001: short description" — links to prompt log.
181. **Dependency locking:** requirements.txt with >= minimum. Environment report has exact versions.
182. **Integration test:** Smoke test — upload sample, run analysis, verify outputs.
183. **Gitignore:** Standard Python + macOS + .streamlit/secrets.toml.
184. **Data entry points:** Upload + text paste. No third option.
185. **Concurrency:** Streamlit session isolation built-in. Free plan resource limits accepted.
186. **Sidebar status:** Post-analysis summary below Run button ("5 topics, C_v: 0.58, 1,247 lemmas").
187. **Critical error screen:** st.error user-friendly message + expander with technical traceback.
188. **Error message format:** What happened + what to try. No error codes.
189. **Custom CSS:** Minimal (~15 lines) via st.markdown. Hamburger menu hide, footer hide.
190. **Community:** GitHub Discussions after DSH publication.
191. **Open source credits:** README Acknowledgments expanded list.
192. **Acceptance test:** 10-point DEPLOY_CHECKLIST.md before v0.1.0 release.
193. **Keyboard accessibility:** Streamlit default (Tab navigation built-in).
194. **Concurrency limits:** Free plan accepted. Heavy use → local install recommendation.
195. **JOSS:** After DSH acceptance, consider separate software paper.
196. **Getting started guide:** README installation section sufficient.
197. **Python versions:** >=3.10 in pyproject.toml, CI tests 3.11 only.
198. **PDF preview:** No. Direct download only.
199. **Technology migration:** Architecture already framework-agnostic (only app.py is Streamlit).
200. **CLAUDE.md size:** Critical rules in CLAUDE.md (~100 lines). Full decisions in ARCHITECTURE.md.
