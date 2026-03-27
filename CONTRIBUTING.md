# Contributing to Lemmata

Thank you for your interest in contributing to Lemmata. This project is an academic research tool, and we welcome contributions that improve its functionality, documentation, or language support.

## How to contribute

1. **Open an issue** describing the bug, feature request, or improvement you have in mind.
2. **Fork the repository** and create a new branch for your changes.
3. **Make your changes** with clear commit messages.
4. **Submit a pull request** referencing the related issue.

## Development notes

- Lemmata is developed through vibe coding (LLM-assisted development). If you contribute code, please include clear comments explaining your changes so the development log remains transparent.
- All NLP and preprocessing decisions should be linguistically motivated. If you propose a change to tokenization, POS filtering, or stopword handling, please include a brief justification.
- The UI layer (app.py) must remain free of business logic. Processing, modeling, and visualization belong in their respective modules.

## Language support

Adding a new language requires:
1. A spaCy model for the target language.
2. A new entry in config.py with the model name and default POS tags.
3. Testing with representative texts in that language.

## Code of conduct

Please be respectful and constructive in all interactions. This is an academic project and we value collegial exchange.

## Questions?

Open an issue or contact the maintainers through the repository.
