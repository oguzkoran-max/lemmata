"""Tests for lemmata.file_io."""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from typing import Any

import numpy as np
import pytest

from lemmata.file_io import (
    FileReadError,
    export_zip,
    get_environment_info,
    get_file_preview,
    get_zip_filename,
    read_file,
    text_from_paste,
)


# ── read_file ─────────────────────────────────────────────────────────────────


class TestReadFile:
    """Verify single-file reading."""

    def test_read_txt_utf8(self):
        content = "Hello world, café résumé"
        result = read_file(content.encode("utf-8"), "test.txt")
        assert result["filename"] == "test.txt"
        assert "Hello world" in result["content"]

    def test_read_txt_latin1(self):
        content = "café"
        result = read_file(content.encode("latin-1"), "test.txt")
        assert result["filename"] == "test.txt"
        assert isinstance(result["content"], str)

    def test_unsupported_extension(self):
        with pytest.raises(FileReadError, match="Unsupported file type"):
            read_file(b"data", "test.xyz")

    def test_file_too_large(self):
        # 51 MB of zeros.
        huge = b"\x00" * (51 * 1024 * 1024)
        with pytest.raises(FileReadError, match="exceeding"):
            read_file(huge, "big.txt")

    def test_returns_dict_with_required_keys(self):
        result = read_file(b"some text", "doc.txt")
        assert "filename" in result
        assert "content" in result


# ── text_from_paste ───────────────────────────────────────────────────────────


class TestTextFromPaste:
    """Verify paste text wrapping."""

    def test_returns_correct_format(self):
        result = text_from_paste("Hello world")
        assert result["filename"] == "pasted_text.txt"
        assert result["content"] == "Hello world"

    def test_preserves_content(self):
        text = "Line one\nLine two\nLine three"
        result = text_from_paste(text)
        assert result["content"] == text


# ── get_file_preview ──────────────────────────────────────────────────────────


class TestGetFilePreview:
    """Verify file preview generation."""

    def test_returns_required_keys(self):
        preview = get_file_preview(b"Hello world test", "test.txt")
        required = {"filename", "size_bytes", "size_display", "word_count",
                     "preview_text", "error"}
        assert required.issubset(set(preview.keys()))

    def test_word_count(self):
        text = "one two three four five"
        preview = get_file_preview(text.encode(), "test.txt")
        assert preview["word_count"] == 5

    def test_preview_limited_to_200_words(self):
        text = " ".join(f"word{i}" for i in range(500))
        preview = get_file_preview(text.encode(), "test.txt")
        preview_words = preview["preview_text"].split()
        assert len(preview_words) <= 200

    def test_size_bytes(self):
        data = b"Hello"
        preview = get_file_preview(data, "test.txt")
        assert preview["size_bytes"] == 5

    def test_error_for_bad_format(self):
        preview = get_file_preview(b"data", "test.xyz")
        assert preview["error"] != ""


# ── get_zip_filename ──────────────────────────────────────────────────────────


class TestGetZipFilename:
    """Verify ZIP filename generation."""

    def test_format(self):
        name = get_zip_filename("it", 5)
        assert name.startswith("lemmata_it_5topics_")
        assert name.endswith(".zip")

    def test_different_params(self):
        name = get_zip_filename("en", 10)
        assert "en" in name
        assert "10topics" in name


# ── export_zip ────────────────────────────────────────────────────────────────


class TestExportZip:
    """Verify ZIP export with all required files."""

    @pytest.fixture()
    def mock_results(self, sample_topics, sample_doc_topic_matrix) -> dict[str, Any]:
        return {
            "topics": sample_topics,
            "doc_topic_matrix": sample_doc_topic_matrix,
            "doc_labels": ["doc1", "doc2", "doc3"],
            "preprocessing_trace": {
                "per_document": [
                    {"label": "doc1", "original_tokens": 50,
                     "stopwords_builtin": 10, "stopwords_custom": 2,
                     "pos_matched": 30, "final_lemmas": 20},
                    {"label": "doc2", "original_tokens": 45,
                     "stopwords_builtin": 8, "stopwords_custom": 1,
                     "pos_matched": 28, "final_lemmas": 18},
                    {"label": "doc3", "original_tokens": 40,
                     "stopwords_builtin": 9, "stopwords_custom": 0,
                     "pos_matched": 25, "final_lemmas": 15},
                ],
            },
            "model_info": {
                "n_topics": 2,
                "perplexity": 42.5,
                "log_likelihood": -1234.5,
                "converged": True,
            },
            "dtm_info": {
                "n_docs": 3,
                "vocabulary_total": 50,
                "vocabulary_kept": 40,
            },
            "coherence": {"c_v": 0.55, "per_topic": [0.6, 0.5]},
        }

    @pytest.fixture()
    def mock_params(self) -> dict[str, Any]:
        return {
            "language": "it",
            "n_topics": 2,
            "n_words": 5,
            "seed": 42,
        }

    def test_returns_bytes(self, mock_results, mock_params):
        data = export_zip(mock_results, mock_params)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_is_valid_zip(self, mock_results, mock_params):
        data = export_zip(mock_results, mock_params)
        with zipfile.ZipFile(BytesIO(data)) as zf:
            assert zf.testzip() is None

    def test_contains_all_files(self, mock_results, mock_params):
        data = export_zip(mock_results, mock_params)
        with zipfile.ZipFile(BytesIO(data)) as zf:
            names = set(zf.namelist())
        expected = {
            "topic_words.csv",
            "doc_topic_matrix.csv",
            "preprocessing_summary.csv",
            "metrics.json",
            "analysis_results.json",
            "environment.json",
        }
        assert expected.issubset(names)

    def test_metrics_json_parseable(self, mock_results, mock_params):
        data = export_zip(mock_results, mock_params)
        with zipfile.ZipFile(BytesIO(data)) as zf:
            metrics = json.loads(zf.read("metrics.json"))
        assert "coherence" in metrics
        assert "perplexity" in metrics

    def test_analysis_results_has_version(self, mock_results, mock_params):
        data = export_zip(mock_results, mock_params)
        with zipfile.ZipFile(BytesIO(data)) as zf:
            results = json.loads(zf.read("analysis_results.json"))
        assert "lemmata_version" in results
        assert "parameters" in results
        assert "topics" in results


# ── get_environment_info ──────────────────────────────────────────────────────


class TestGetEnvironmentInfo:
    """Verify environment info collection."""

    def test_returns_dict(self):
        info = get_environment_info({"seed": 42})
        assert isinstance(info, dict)

    def test_has_required_keys(self):
        info = get_environment_info({"seed": 42})
        assert "lemmata_version" in info
        assert "python_version" in info
        assert "platform" in info
        assert "parameters" in info

    def test_includes_params(self):
        params = {"seed": 42, "language": "it"}
        info = get_environment_info(params)
        assert info["parameters"] == params
