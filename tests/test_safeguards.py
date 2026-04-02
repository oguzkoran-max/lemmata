"""Tests for corpus safeguards (decisions 128, 161, 163, 164)."""

from __future__ import annotations

import pytest

from lemmata.app import (
    calc_topic_max,
    check_imbalanced_corpus,
    estimate_analysis_time,
    estimate_n_documents,
    find_content_duplicates,
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


class TestFindContentDuplicates:
    """Verify content-based duplicate detection (decision 163)."""

    def test_detects_identical_content(self):
        texts = [
            {"filename": "a.txt", "content": "Hello world this is a test."},
            {"filename": "b.txt", "content": "Hello world this is a test."},
        ]
        result = find_content_duplicates(texts)
        assert len(result) == 1
        assert "b.txt" in result[0]
        assert "a.txt" in result[0]

    def test_no_false_positive_different_content(self):
        texts = [
            {"filename": "a.txt", "content": "Hello world."},
            {"filename": "b.txt", "content": "Goodbye world."},
        ]
        result = find_content_duplicates(texts)
        assert result == []

    def test_no_warning_single_file(self):
        texts = [{"filename": "a.txt", "content": "Hello world."}]
        assert find_content_duplicates(texts) == []

    def test_empty_list(self):
        assert find_content_duplicates([]) == []

    def test_same_prefix_different_length(self):
        texts = [
            {"filename": "a.txt", "content": "word " * 100},
            {"filename": "b.txt", "content": "word " * 200},
        ]
        result = find_content_duplicates(texts)
        assert result == []


class TestEstimateAnalysisTime:
    """Verify analysis time estimation (decision 161)."""

    def test_small_corpus(self):
        result = estimate_analysis_time(n_documents=5, n_topics=3, total_words=5000)
        assert isinstance(result, int)
        assert result >= 5

    def test_rounds_to_nearest_5(self):
        result = estimate_analysis_time(n_documents=1, n_topics=2, total_words=1000)
        assert result % 5 == 0

    def test_large_corpus_higher_estimate(self):
        small = estimate_analysis_time(5, 3, 5000)
        large = estimate_analysis_time(100, 10, 500000)
        assert large > small

    def test_minimum_is_5(self):
        result = estimate_analysis_time(1, 2, 100)
        assert result >= 5
