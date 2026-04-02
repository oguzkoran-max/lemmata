"""Tests for corpus safeguards (decisions 128, 164)."""

from __future__ import annotations

import pytest

from lemmata.app import (
    calc_topic_max,
    check_imbalanced_corpus,
    estimate_n_documents,
)


class TestCalcTopicMax:
    """Verify topic-document safeguard (decision 164)."""

    def test_10_docs_gives_5(self):
        assert calc_topic_max(10) == 5

    def test_3_docs_gives_2(self):
        # 3 // 2 = 1, but minimum is 2.
        assert calc_topic_max(3) == 2

    def test_1_doc_gives_2(self):
        assert calc_topic_max(1) == 2

    def test_0_docs_gives_2(self):
        assert calc_topic_max(0) == 2

    def test_100_docs_gives_50(self):
        assert calc_topic_max(100) == 50


class TestEstimateNDocuments:
    """Verify document count estimation."""

    def test_multi_file_returns_file_count(self):
        texts = [
            {"filename": "a.txt", "content": "word " * 100},
            {"filename": "b.txt", "content": "word " * 200},
            {"filename": "c.txt", "content": "word " * 50},
        ]
        assert estimate_n_documents(texts, chunk_size=1000) == 3

    def test_single_file_estimates_chunks(self):
        texts = [{"filename": "a.txt", "content": "word " * 5000}]
        est = estimate_n_documents(texts, chunk_size=1000)
        assert est == 5

    def test_empty_returns_zero(self):
        assert estimate_n_documents([], chunk_size=1000) == 0

    def test_short_single_file_gives_1(self):
        texts = [{"filename": "a.txt", "content": "short text"}]
        assert estimate_n_documents(texts, chunk_size=1000) == 1


class TestCheckImbalancedCorpus:
    """Verify imbalanced corpus warning (decision 128)."""

    def test_triggers_at_10x(self):
        texts = [
            {"filename": "a.txt", "content": "word " * 100},
            {"filename": "b.txt", "content": "word " * 1000},
        ]
        result = check_imbalanced_corpus(texts)
        assert result is not None
        assert "10x" in result

    def test_no_trigger_at_5x(self):
        texts = [
            {"filename": "a.txt", "content": "word " * 100},
            {"filename": "b.txt", "content": "word " * 500},
        ]
        result = check_imbalanced_corpus(texts)
        assert result is None

    def test_no_trigger_balanced(self):
        texts = [
            {"filename": "a.txt", "content": "word " * 100},
            {"filename": "b.txt", "content": "word " * 120},
        ]
        result = check_imbalanced_corpus(texts)
        assert result is None

    def test_no_trigger_single_file(self):
        texts = [{"filename": "a.txt", "content": "word " * 1000}]
        result = check_imbalanced_corpus(texts)
        assert result is None

    def test_no_trigger_empty(self):
        result = check_imbalanced_corpus([])
        assert result is None

    def test_custom_threshold(self):
        texts = [
            {"filename": "a.txt", "content": "word " * 100},
            {"filename": "b.txt", "content": "word " * 600},
        ]
        assert check_imbalanced_corpus(texts, threshold=5.0) is not None
        assert check_imbalanced_corpus(texts, threshold=10.0) is None
