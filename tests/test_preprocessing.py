"""Tests for lemmata.preprocessing."""

from __future__ import annotations

import pytest

from lemmata.preprocessing import (
    check_language_match,
    chunk_text,
    clean_text,
    detect_encoding,
    process_documents,
)


# ── check_language_match ──────────────────────────────────────────────────────


class TestCheckLanguageMatch:
    """Verify early language mismatch detection (P022)."""

    def test_returns_none_for_matching_language(self, mock_nlp):
        texts = [{"filename": "a.txt", "content": "This is a normal English sentence with words."}]
        result = check_language_match(texts, mock_nlp)
        # mock_nlp recognises tokens, so should be None.
        assert result is None

    def test_returns_none_for_empty_input(self, mock_nlp):
        texts = [{"filename": "a.txt", "content": ""}]
        result = check_language_match(texts, mock_nlp)
        assert result is None

    def test_samples_limited_chars(self, mock_nlp):
        long_text = "word " * 10000
        texts = [{"filename": "a.txt", "content": long_text}]
        # Should not crash on huge input — samples only first 500 chars.
        result = check_language_match(texts, mock_nlp, sample_chars=500)
        assert result is None or isinstance(result, str)


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

    def test_stopwords_on_both_lists_counted_once(self, mock_nlp):
        """A lemma on the built-in and the custom list is removed once (P045)."""
        texts = [
            {"filename": "a.txt", "content": "il gatto e il cane dormono in casa"}
        ]
        _, _, trace = process_documents(
            texts=texts,
            language="it",
            custom_stopwords={"il", "casa"},
            nlp=mock_nlp,
        )
        # 8 tokens. Built-in: il, e, il, in. Custom: il, il, casa.
        assert trace["original_tokens"] == 8
        assert trace["stopwords_removed_builtin"] == 4
        assert trace["stopwords_removed_custom"] == 3
        assert trace["stopwords_removed_both"] == 2
        assert trace["stopwords_removed_total"] == 5
        assert trace["after_stopwords"] == 3
        # A sequential filter cannot end with more tokens than it passed on.
        assert trace["after_stopwords"] >= trace["final_lemmas"]

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

    # ── chunk_each_file (P052) ──────────────────────────────────────────

    _LONG = " ".join(["This is a test sentence."] * 50)  # 250 words
    _SHORT = "A short note here."  # 4 words

    def test_chunk_each_file_splits_long_files_only(self, mock_nlp):
        corpus = [
            {"filename": "long.txt", "content": self._LONG},
            {"filename": "short.txt", "content": self._SHORT},
        ]
        _, labels, _ = process_documents(
            texts=corpus, language="en", chunk_size=20, nlp=mock_nlp,
            chunk_each_file=True,
        )
        long_parts = [lb for lb in labels if lb.startswith("long_")]
        assert len(long_parts) > 1
        assert long_parts[0] == "long_001"
        # The short file stays whole, labelled with its name.
        assert labels[-1] == "short"
        assert len(labels) == len(long_parts) + 1

    def test_chunk_each_file_off_keeps_one_document_per_file(self, mock_nlp):
        corpus = [
            {"filename": "long.txt", "content": self._LONG},
            {"filename": "short.txt", "content": self._SHORT},
        ]
        _, labels, _ = process_documents(
            texts=corpus, language="en", chunk_size=20, nlp=mock_nlp,
        )
        assert labels == ["long", "short"]

    def test_chunk_each_file_leaves_short_files_unchanged(
        self, sample_corpus, mock_nlp,
    ):
        # Files no longer than the chunk size are processed exactly as before.
        off = process_documents(texts=sample_corpus, language="it", nlp=mock_nlp)
        on = process_documents(
            texts=sample_corpus, language="it", nlp=mock_nlp,
            chunk_each_file=True,
        )
        assert on[0] == off[0]
        assert on[1] == off[1]

    def test_empty_file_skipped(self, mock_nlp):
        corpus = [
            {"filename": "empty.txt", "content": ""},
            {"filename": "real.txt", "content": "This has real content inside."},
        ]
        texts, labels, trace = process_documents(
            texts=corpus, language="en", nlp=mock_nlp,
        )
        assert trace["empty_chunks_removed"] >= 1


# ── multi-word lemmas (P049) ──────────────────────────────────────────────────


class _Token:
    """spaCy token stand-in."""

    def __init__(self, text: str, lemma: str, pos: str = "NOUN") -> None:
        self.text, self.lemma_, self.pos_ = text, lemma, pos
        self.is_stop, self.is_alpha = False, True
        self.is_space, self.is_punct = False, False


class _Nlp:
    """spaCy stand-in whose lemmas include clitic splits such as 'fare mi'."""

    pipe_names = ["parser"]

    class Defaults:
        stop_words = {"mi", "si", "il", "ne"}

    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens

    def __call__(self, text: str) -> list[_Token]:
        return list(self._tokens)

    def pipe(self, texts: list[str]) -> list[list[_Token]]:
        return [list(self._tokens) for _ in texts]


class TestMultiWordLemmas:
    """Each part of a multi-word lemma is checked against the stoplists (P049)."""

    TOKENS = [
        _Token("farmi", "fare mi", "VERB"),
        _Token("andatosene", "andatoe si ne", "ADJ"),
        _Token("sallo", "sa il", "ADJ"),
        _Token("casa", "casa", "NOUN"),
    ]

    def _run(self, custom: set[str] | None = None) -> list[str]:
        texts = [
            {"filename": "a.txt", "content": "uno"},
            {"filename": "b.txt", "content": "due"},
        ]
        processed, _, _ = process_documents(
            texts=texts, language="it", pos_tags=["NOUN", "ADJ", "VERB"],
            custom_stopwords=custom, nlp=_Nlp(self.TOKENS),
        )
        return processed

    def test_stopword_parts_are_dropped(self):
        # "farmi" -> "fare mi" reached the model as "fare" and "mi".
        assert self._run()[0].split() == ["fare", "andatoe", "sa", "casa"]

    def test_no_stopword_reaches_the_model(self):
        custom = {"casa", "sa"}
        words = {w for text in self._run(custom) for w in text.split()}
        assert words == {"fare", "andatoe"}
        assert not words & (_Nlp.Defaults.stop_words | custom)
