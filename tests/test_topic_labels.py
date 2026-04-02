"""Tests for topic label editing (decision 33)."""

from __future__ import annotations

import pytest

from lemmata.app import resolve_topic_labels


@pytest.fixture()
def sample_topics():
    return [
        {"topic_id": 1, "label": "Topic 1 (word1, word2, word3)"},
        {"topic_id": 2, "label": "Topic 2 (word4, word5, word6)"},
        {"topic_id": 3, "label": "Topic 3 (word7, word8, word9)"},
    ]


class TestResolveTopicLabels:
    """Verify resolve_topic_labels logic."""

    def test_returns_default_when_no_custom(self, sample_topics):
        result = resolve_topic_labels(sample_topics)
        assert result == [t["label"] for t in sample_topics]

    def test_returns_default_when_custom_is_none(self, sample_topics):
        result = resolve_topic_labels(sample_topics, custom_labels=None)
        assert result == [t["label"] for t in sample_topics]

    def test_returns_custom_when_valid(self, sample_topics):
        custom = ["Family", "Travel", "Food"]
        result = resolve_topic_labels(sample_topics, custom_labels=custom)
        assert result == custom

    def test_ignores_custom_wrong_length(self, sample_topics):
        custom = ["Family", "Travel"]  # Only 2 labels for 3 topics
        result = resolve_topic_labels(sample_topics, custom_labels=custom)
        assert result == [t["label"] for t in sample_topics]

    def test_ignores_custom_empty_list(self, sample_topics):
        result = resolve_topic_labels(sample_topics, custom_labels=[])
        assert result == [t["label"] for t in sample_topics]

    def test_ignores_custom_wrong_type(self, sample_topics):
        result = resolve_topic_labels(sample_topics, custom_labels="not a list")
        assert result == [t["label"] for t in sample_topics]
