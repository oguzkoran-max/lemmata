# Prompt Log

This directory records the Claude Code sessions that built and later
revised Lemmata. Each file (P001.md – P062.md) describes one change and
carries the number of its git commit. P043 and P044 (4 July 2026) have
commits but no log files. The list is in
[PROMPT_LOG_INDEX.md](PROMPT_LOG_INDEX.md).

## Format

**P001–P042 (March–June 2026).** Each file contains:

- **Prompt number and title** matching the git commit
- **Date** of the interaction
- **Commit message** and test count
- **Prompt Summary** — concise description of what was asked
- **Full Prompt** — the English text sent to Claude Code
- **Result** — outcome including test counts

The full prompt texts of P001–P041 were added afterwards, in P042.

**P045 onwards (November 2026).** These changes did not start from a
single English prompt. They were discussed with the first author in
Turkish-language Claude Code sessions. Each file records the date, the
model, the files changed, the test count, what was asked and done, the
decisions taken and who took them, and how the change was checked.

## Methodology Note

The prompts in P001–P042 were written in English. Planning, discussion,
and decision-making were conducted in Turkish via claude.ai chat sessions.
Those transcripts are kept by the authors and are not published here.

## Sub-prompts

Some prompts were sent as multiple sequential instructions:

- **P011** (a–d): Four iterative Streamlit Cloud deployment fixes
- **P012** (a–c): Initial landing page + full redesign + URL fix
- **P019** + hotfix: Bug fixes + sidebar CSS hotfix
- **P022** + debug + final: Language warning fix iterations
- **P023** + P023b: Meta tags + OG preview image

These are consolidated into single files for clarity.
