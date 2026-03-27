"""Shared fixtures for Lemmata test suite."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest


# ── Sample Texts ──────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_italian_text() -> str:
    """Short Italian paragraph for unit tests."""
    return (
        "La vita è bella e il mondo è pieno di meraviglie. "
        "Ogni giorno porta nuove avventure e scoperte. "
        "Gli uomini e le donne camminano per le strade della città. "
        "I bambini giocano nel parco sotto il sole caldo. "
        "La natura offre spettacoli straordinari in ogni stagione."
    )


@pytest.fixture()
def sample_english_text() -> str:
    """Short English paragraph for unit tests."""
    return (
        "The world is full of wonders and discoveries. "
        "Every day brings new adventures and opportunities. "
        "People walk through the streets of the ancient city. "
        "Children play in the park under the warm sun. "
        "Nature offers extraordinary spectacles in every season."
    )


@pytest.fixture()
def sample_corpus() -> list[dict[str, str]]:
    """Three-document Italian corpus for pipeline tests."""
    return [
        {
            "filename": "doc1.txt",
            "content": (
                "La letteratura italiana ha una lunga storia. "
                "Dante Alighieri scrisse la Divina Commedia nel Trecento. "
                "L'opera rappresenta un viaggio attraverso Inferno, "
                "Purgatorio e Paradiso. Il poeta fiorentino è considerato "
                "il padre della lingua italiana moderna."
            ),
        },
        {
            "filename": "doc2.txt",
            "content": (
                "Il Rinascimento fu un periodo di grande fioritura artistica. "
                "Leonardo da Vinci dipinse la Gioconda e l'Ultima Cena. "
                "Michelangelo scolpì il David e affrescò la Cappella Sistina. "
                "Firenze divenne il centro culturale dell'Europa."
            ),
        },
        {
            "filename": "doc3.txt",
            "content": (
                "La cucina italiana è famosa in tutto il mondo. "
                "La pasta e la pizza sono piatti conosciuti ovunque. "
                "Ogni regione ha le proprie specialità gastronomiche. "
                "Il vino italiano accompagna i pasti con grande varietà."
            ),
        },
    ]


# ── Processed Text Fixtures ──────────────────────────────────────────────────


@pytest.fixture()
def processed_texts() -> list[str]:
    """Pre-processed lemma strings (simulates preprocessing output)."""
    return [
        "letteratura italiano lungo storia dante alighieri divino commedia "
        "trecento opera viaggio inferno purgatorio paradiso poeta fiorentino "
        "padre lingua italiano moderno",
        "rinascimento periodo grande fioritura artistico leonardo vinci "
        "gioconda ultimo cena michelangelo david cappella sistino firenze "
        "centro culturale europa",
        "cucina italiano famoso mondo pasta pizza piatto conosciuto "
        "regione specialità gastronomico vino italiano pasto grande varietà",
    ]


@pytest.fixture()
def doc_labels() -> list[str]:
    """Document labels matching processed_texts."""
    return ["doc1", "doc2", "doc3"]


# ── Topic Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture()
def sample_topics() -> list[dict[str, Any]]:
    """Minimal topic dicts for visualisation tests."""
    return [
        {
            "topic_id": 1,
            "label": "Topic 1 (letteratura, storia, poeta)",
            "words": ["letteratura", "storia", "poeta", "opera", "lingua"],
            "weights": [0.35, 0.25, 0.18, 0.12, 0.10],
            "avg_weight": 0.45,
        },
        {
            "topic_id": 2,
            "label": "Topic 2 (arte, firenze, rinascimento)",
            "words": ["arte", "firenze", "rinascimento", "cappella", "centro"],
            "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
            "avg_weight": 0.35,
        },
    ]


@pytest.fixture()
def sample_doc_topic_matrix() -> np.ndarray:
    """3-doc × 2-topic matrix for visualisation tests."""
    return np.array([
        [0.7, 0.3],
        [0.2, 0.8],
        [0.5, 0.5],
    ])


# ── Mock spaCy ────────────────────────────────────────────────────────────────


class _MockToken:
    """Minimal spaCy Token mock."""

    def __init__(
        self, text: str, lemma: str, pos: str,
        is_stop: bool = False, is_alpha: bool = True,
        is_space: bool = False, is_punct: bool = False,
    ):
        self.text = text
        self.lemma_ = lemma
        self.pos_ = pos
        self.is_stop = is_stop
        self.is_alpha = is_alpha
        self.is_space = is_space
        self.is_punct = is_punct


class _MockSent:
    """Minimal spaCy Span mock for sentences."""

    def __init__(self, text: str):
        self.text = text


class _MockDoc:
    """Minimal spaCy Doc mock."""

    def __init__(self, tokens: list[_MockToken], sents: list[_MockSent]):
        self._tokens = tokens
        self.sents = sents

    def __iter__(self):
        return iter(self._tokens)


@pytest.fixture()
def mock_nlp():
    """A mock spaCy Language that returns a simple Doc with sentences."""
    nlp = MagicMock()
    nlp.pipe_names = ["parser"]

    class _Defaults:
        stop_words = {"il", "la", "e", "di", "in", "è", "un", "the", "a", "is"}

    nlp.Defaults = _Defaults

    def _call(text):
        sents = [_MockSent(s.strip()) for s in text.split(".") if s.strip()]
        tokens = []
        for word in text.split():
            clean = word.strip(".,;:!?")
            if not clean:
                continue
            tokens.append(
                _MockToken(
                    text=clean,
                    lemma=clean.lower(),
                    pos="NOUN",
                    is_stop=clean.lower() in _Defaults.stop_words,
                )
            )
        return _MockDoc(tokens, sents)

    nlp.side_effect = _call
    nlp.__call__ = _call

    def _pipe(texts):
        return [_call(t) for t in texts]

    nlp.pipe = _pipe

    return nlp
