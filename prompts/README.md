# Prompt Log

This directory contains the complete prompt archive for each Claude Code
interaction during Lemmata's development (P001–P041). These logs document
the human–LLM dialogue that built the platform from scratch using
vibe coding methodology.

## Format

Each file (P001.md – P041.md) contains:

- **Prompt number and title** matching the git commit
- **Date** of the interaction
- **Commit message** and test count
- **Prompt Summary** — concise description of what was asked
- **Full Prompt** — the exact English text sent to Claude Code
- **Result** — outcome including test counts

## Methodology Note

All Claude Code prompts were written in English. Planning, discussion,
and decision-making were conducted in Turkish via claude.ai chat sessions
(transcripts archived separately as session logs).

This dual-language workflow — Turkish for thinking, English for coding —
is documented as part of the DSH article's vibe coding methodology
analysis.

## Sub-prompts

Some prompts were sent as multiple sequential instructions:

- **P011** (a–d): Four iterative Streamlit Cloud deployment fixes
- **P012** (a–c): Initial landing page + full redesign + URL fix
- **P019** + hotfix: Bug fixes + sidebar CSS hotfix
- **P022** + debug + final: Language warning fix iterations
- **P023** + P023b: Meta tags + OG preview image

These are consolidated into single files for clarity.

## Files

P001.md – P041.md (41 prompt logs)
