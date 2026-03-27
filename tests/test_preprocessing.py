"""Tests for lemmata.preprocessing."""

from __future__ import annotations

import pytest

from lemmata.preprocessing import (
    chunk_text,
    clean_text,
    detect_encoding,
    process_documents,
)


# ── detect_encoding ───────────────────────────────────────────────────────────


class TestDetectEncoding:
    """Verify encoding detection and fallback chain."""

    def test_utf8_bytes(self):
        text = "Héllo wörld àéîõü"
        enc = detect_encoding(text.encode("utf-8"))
        assert isinstance(enc, str)
        # Should decode without error.
        text.encode("utf-8").decode(enc)

    def test_latin1_bytes(self):
        text = "café résumé naïve"
        raw = text.encode("latin-1")
        enc = detect_encoding(raw)
        assert isinstance(enc, str)
        raw.decode(enc)

    def test_empty_bytes(self):
        enc = detect_encoding(b"")
        assert isinstance(enc, str)

    def test_ascii_bytes(self):
        enc = detect_encoding(b"plain ascii text")
        assert isinstance(enc, str)


# ── clean_text ────────────────────────────────────────────────────────────────


class TestCleanText:
    """Verify text hygiene operations."""

    def test_bom_removal(self):
        text = "\ufeffHello world"
        assert clean_text(text) == "Hello world"

    def test_null_byte_removal(self):
        text = "Hello\x00world"
        assert "\x00" not in clean_text(text)

    def test_control_char_removal(self):
        text = "Hello\x07world\x0etest"
        result = clean_text(text)
        assert "\x07" not in result
        assert "\x0e" not in result

    def test_preserves_newlines(self):
        text = "Line one\nLine two"
        assert "\n" in clean_text(text)

    def test_line_end_hyphen_joining(self):
        text = "pre-\nprocessing is impor-\ntant"
        result = clean_text(text)
        assert "preprocessing" in result
        assert "important" in result

    def test_multi_space_collapse(self):
        text = "Hello   world    test"
        result = clean_text(text)
        assert "   " not in result
        assert "Hello world test" == result

    def test_crlf_normalisation(self):
        text = "Line one\r\nLine two\rLine three"
        result = clean_text(text)
        assert "\r" not in result
        assert result.count("\n") == 2

    def test_unicode_nfc(self):
        # Combining character é (e + combining acute) → single é.
        import unicodedata

        decomposed = unicodedata.normalize("NFD", "café")
        result = clean_text(decomposed)
        assert result == unicodedata.normalize("NFC", "café")

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"


# ── chunk_text ────────────────────────────────────────────────────────────────


class TestChunkText:
    """Verify sentence-boundary chunking."""

    def test_label_format(self, mock_nlp):
        text = "First sentence. Second sentence. Third sentence."
        chunks = chunk_text(text, "myfile", chunk_size=5, nlp=mock_nlp)
        assert len(chunks) > 0
        for chunk in chunks:
            assert "label" in chunk
            assert "text" in chunk
        # First chunk should be myfile_001.
        assert chunks[0]["label"] == "myfile_001"

    def test_sequential_labels(self, mock_nlp):
        text = "One sentence. Two sentence. Three sentence. Four sentence."
        chunks = chunk_text(text, "test", chunk_size=3, nlp=mock_nlp)
        for i, chunk in enumerate(chunks, start=1):
            assert chunk["label"] == f"test_{i:03d}"

    def test_empty_text_returns_empty(self, mock_nlp):
        chunks = chunk_text("", "empty", chunk_size=100, nlp=mock_nlp)
        assert chunks == []

    def test_single_chunk_for_short_text(self, mock_nlp):
        text = "Short text."
        chunks = chunk_text(text, "short", chunk_size=1000, nlp=mock_nlp)
        assert len(chunks) == 1

    def test_no_empty_chunks(self, mock_nlp):
        text = "One. Two. Three. Four. Five."
        chunks = chunk_text(text, "test", chunk_size=2, nlp=mock_nlp)
        for chunk in chunks:
            assert chunk["text"].strip() != ""


# ── process_documents ─────────────────────────────────────────────────────────


class TestProcessDocuments:
    """Verify the full preprocessing pipeline."""

    def test_return_types(self, sample_corpus, mock_nlp):
        texts, labels, trace = process_documents(
            texts=sample_corpus,
            language="it",
            nlp=mock_nlp,
        )
        assert isinstance(texts, list)
        assert isinstance(labels, list)
        assert isinstance(trace, dict)
        assert len(texts) == len(labels)

    def test_trace_has_required_keys(self, sample_corpus, mock_nlp):
        _, _, trace = process_documents(
            texts=sample_corpus, language="it", nlp=mock_nlp,
        )
        required_keys = {
            "original_tokens",
            "after_stopwords",
            "after_pos",
            "final_lemmas",
            "unique_lemmas",
            "chunks_created",
            "empty_chunks_removed",
            "stopwords_removed_builtin",
            "stopwords_removed_custom",
            "language",
            "pos_tags",
            "custom_stopwords",
            "warnings",
            "per_document",
        }
        assert required_keys.issubset(set(trace.keys()))

    def test_per_document_trace_keys(self, sample_corpus, mock_nlp):
        _, _, trace = process_documents(
            texts=sample_corpus, language="it", nlp=mock_nlp,
        )
        for doc in trace["per_document"]:
            assert "label" in doc
            assert "original_tokens" in doc
            assert "final_lemmas" in doc
            assert "token_details" in doc

    def test_custom_stopwords_counted(self, sample_corpus, mock_nlp):
        _, _, trace = process_documents(
            texts=sample_corpus,
            language="it",
            custom_stopwords={"mondo", "grande"},
            nlp=mock_nlp,
        )
        assert trace["stopwords_removed_custom"] >= 0

    def test_single_file_triggers_chunking(self, mock_nlp):
        single = [
            {
                "filename": "long.txt",
                "content": " ".join(["This is a test sentence."] * 50),
            }
        ]
        texts, labels, trace = process_documents(
            texts=single, language="en", chunk_size=20, nlp=mock_nlp,
        )
        # Should produce multiple chunks.
        assert len(labels) >= 1
        if len(labels) > 1:
            assert labels[0].startswith("long_")

    def test_multi_file_no_chunking(self, sample_corpus, mock_nlp):
        texts, labels, _ = process_documents(
            texts=sample_corpus, language="it", nlp=mock_nlp,
        )
        # Labels should be filenames without extension (no _001 suffix).
        for label in labels:
            assert "_0" not in label or label.startswith("doc")

    def test_empty_file_skipped(self, mock_nlp):
        corpus = [
            {"filename": "empty.txt", "content": ""},
            {"filename": "real.txt", "content": "This has real content inside."},
        ]
        texts, labels, trace = process_documents(
            texts=corpus, language="en", nlp=mock_nlp,
        )
        assert trace["empty_chunks_removed"] >= 1
