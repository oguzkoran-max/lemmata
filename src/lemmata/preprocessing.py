"""Text preprocessing pipeline for Lemmata.

Handles everything between raw file content and the document-term matrix:
encoding detection, text hygiene, chunking, spaCy NLP, POS filtering,
stopword removal, and lemmatisation.  Returns processed texts, document
labels, and a full preprocessing trace.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any

import chardet
import spacy
from spacy.language import Language
from spacy.tokens import Doc

from lemmata.config import (
    CHUNK_SIZE_DEFAULT,
    DEFAULT_POS_TAGS,
    ENCODING_FALLBACK_CHAIN,
    LOW_TOKEN_RATIO_THRESHOLD,
    MIN_UNIQUE_LEMMAS_WARNING,
    SUPPORTED_LANGUAGES,
)

logger = logging.getLogger(__name__)

# ── Regex patterns (compiled once) ────────────────────────────────────────────

_RE_HYPHEN_LINEBREAK = re.compile(r"(\w)-\n(\w)")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_RE_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# ── Encoding Detection ───────────────────────────────────────────────────────


def detect_encoding(raw_bytes: bytes) -> str:
    """Detect text encoding using chardet with UTF-8 → Latin-1 fallback.

    Parameters
    ----------
    raw_bytes:
        Raw file bytes.

    Returns
    -------
    str
        Name of the detected (or fallback) encoding.
    """
    result = chardet.detect(raw_bytes)
    encoding = result.get("encoding")
    confidence = result.get("confidence", 0.0)

    if encoding and confidence >= 0.5:
        # Normalise name and try decoding to verify.
        encoding = encoding.lower().replace("-", "_")
        try:
            raw_bytes.decode(encoding)
            return encoding
        except (UnicodeDecodeError, LookupError):
            pass

    # Walk the fallback chain.
    for enc in ENCODING_FALLBACK_CHAIN:
        try:
            raw_bytes.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue

    # Last resort — will replace undecodable bytes.
    return "utf-8"


# ── Text Hygiene ──────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Apply text hygiene: BOM, control chars, hyphen joining, normalisation.

    Implements decisions 37 and 170.

    Parameters
    ----------
    text:
        Raw decoded text.

    Returns
    -------
    str
        Cleaned text ready for NLP processing.
    """
    # BOM removal.
    text = text.lstrip("\ufeff")

    # Unicode NFC normalisation.
    text = unicodedata.normalize("NFC", text)

    # Control characters (keep \n, \r, \t).
    text = _RE_CONTROL_CHARS.sub("", text)

    # Normalise line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Line-end hyphen joining: "pre-\nprocessing" → "preprocessing".
    text = _RE_HYPHEN_LINEBREAK.sub(r"\1\2", text)

    # Collapse multiple spaces/tabs (not newlines).
    text = _RE_MULTI_SPACE.sub(" ", text)

    return text.strip()


# ── Chunking ──────────────────────────────────────────────────────────────────


def chunk_text(
    text: str,
    filename: str,
    chunk_size: int,
    nlp: Language,
) -> list[dict[str, Any]]:
    """Split text into chunks respecting sentence boundaries.

    Implements decisions 4, 32, 42, 113, 166.

    Parameters
    ----------
    text:
        Cleaned text.
    filename:
        Base filename (without extension) for labelling.
    chunk_size:
        Target word count per chunk.
    nlp:
        Loaded spaCy Language for sentence detection.

    Returns
    -------
    list[dict]
        Each dict: ``{"label": str, "text": str}``.
        Empty chunks are silently dropped.
    """
    # Use sentencizer for fast sentence detection if no parser.
    doc = nlp(text)

    sentences: list[str] = []
    for sent in doc.sents:
        s = sent.text.strip()
        if s:
            sentences.append(s)

    if not sentences:
        return []

    chunks: list[dict[str, Any]] = []
    current_words: list[str] = []
    current_sentences: list[str] = []

    for sent in sentences:
        words = sent.split()
        # If adding this sentence would exceed target and we already have
        # content, finalise the current chunk first.
        if current_words and len(current_words) + len(words) > chunk_size:
            chunk_text_joined = " ".join(current_sentences)
            if chunk_text_joined.strip():
                chunks.append({"label": "", "text": chunk_text_joined})
            current_words = []
            current_sentences = []

        current_words.extend(words)
        current_sentences.append(sent)

    # Flush remaining.
    if current_sentences:
        chunk_text_joined = " ".join(current_sentences)
        if chunk_text_joined.strip():
            chunks.append({"label": "", "text": chunk_text_joined})

    # Assign labels: [filename]_001 format.
    for i, chunk in enumerate(chunks, start=1):
        chunk["label"] = f"{filename}_{i:03d}"

    return chunks


# ── spaCy Model Loading ──────────────────────────────────────────────────────


def load_spacy_model(language: str) -> Language:
    """Load the spaCy model for *language*.

    The caller (app.py) should wrap this with ``@st.cache_resource``.

    Parameters
    ----------
    language:
        ISO 639-1 code present in ``SUPPORTED_LANGUAGES``.

    Returns
    -------
    Language
        Loaded spaCy pipeline.

    Raises
    ------
    ValueError
        If the language is not supported.
    OSError
        If the spaCy model is not installed.
    """
    model_name = SUPPORTED_LANGUAGES.get(language)
    if model_name is None:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Choose from: {', '.join(SUPPORTED_LANGUAGES)}"
        )
    nlp = spacy.load(model_name)
    # Ensure sentencizer is available.
    if "parser" not in nlp.pipe_names and "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp


# ── Token-Level Processing ────────────────────────────────────────────────────


def _process_token(
    token: Any,
    pos_tags: set[str],
    builtin_stops: set[str],
    custom_stops: set[str],
) -> dict[str, Any] | None:
    """Process a single spaCy token.  Returns trace dict or None if filtered.

    Per-token try-except for error tolerance (decision 105).
    """
    try:
        form = token.text
        lemma = token.lemma_.lower()
        pos = token.pos_

        is_space = token.is_space or token.is_punct
        is_builtin_stop = lemma in builtin_stops or token.is_stop
        is_custom_stop = lemma in custom_stops
        is_pos_match = pos in pos_tags
        is_alpha = token.is_alpha

        kept = (
            is_alpha
            and not is_space
            and not is_builtin_stop
            and not is_custom_stop
            and is_pos_match
        )

        return {
            "form": form,
            "lemma": lemma,
            "pos": pos,
            "is_stop_builtin": is_builtin_stop,
            "is_stop_custom": is_custom_stop,
            "pos_match": is_pos_match,
            "kept": kept,
        }
    except Exception:
        logger.debug("Skipping token due to processing error", exc_info=True)
        return None


def _process_doc(
    doc: Doc,
    pos_tags: set[str],
    builtin_stops: set[str],
    custom_stops: set[str],
) -> tuple[list[str], list[dict[str, Any]]]:
    """Process a spaCy Doc, returning kept lemmas and token trace rows."""
    lemmas: list[str] = []
    token_details: list[dict[str, Any]] = []

    for token in doc:
        result = _process_token(token, pos_tags, builtin_stops, custom_stops)
        if result is None:
            continue
        token_details.append(result)
        if result["kept"]:
            lemmas.append(result["lemma"])

    return lemmas, token_details


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def process_documents(
    texts: list[dict[str, str]],
    language: str,
    pos_tags: list[str] | None = None,
    custom_stopwords: set[str] | None = None,
    chunk_size: int = CHUNK_SIZE_DEFAULT,
    nlp: Language | None = None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Run the full preprocessing pipeline.

    Parameters
    ----------
    texts:
        Each dict has ``"filename"`` and ``"content"`` (raw decoded text).
    language:
        ISO 639-1 code.
    pos_tags:
        POS tags to keep.  Defaults to ``DEFAULT_POS_TAGS``.
    custom_stopwords:
        User-supplied stopwords (lowercased).
    chunk_size:
        Target word count per chunk (only used for single-file uploads).
    nlp:
        Pre-loaded spaCy model.  If *None*, loaded here.

    Returns
    -------
    tuple[list[str], list[str], dict]
        ``(processed_texts, doc_labels, preprocessing_trace)``

        - *processed_texts*: one string of space-joined lemmas per document.
        - *doc_labels*: matching document/chunk labels.
        - *preprocessing_trace*: summary and per-document detail.
    """
    if pos_tags is None:
        pos_tags = DEFAULT_POS_TAGS
    if custom_stopwords is None:
        custom_stopwords = set()
    else:
        custom_stopwords = {w.lower() for w in custom_stopwords}

    if nlp is None:
        nlp = load_spacy_model(language)

    pos_tag_set: set[str] = set(pos_tags)
    builtin_stops: set[str] = nlp.Defaults.stop_words.copy()

    # ── Decide: chunk or keep as separate docs ────────────────────────────
    single_file = len(texts) == 1
    documents: list[dict[str, str]] = []  # {"label": ..., "text": ...}
    empty_chunks_removed = 0

    for entry in texts:
        filename = _strip_extension(entry["filename"])
        cleaned = clean_text(entry["content"])

        if not cleaned.strip():
            empty_chunks_removed += 1
            logger.info("Empty file skipped: %s", entry["filename"])
            continue

        if single_file:
            chunks = chunk_text(cleaned, filename, chunk_size, nlp)
            if not chunks:
                empty_chunks_removed += 1
                continue
            documents.extend(chunks)
        else:
            documents.append({"label": filename, "text": cleaned})

    # ── spaCy processing ──────────────────────────────────────────────────
    processed_texts: list[str] = []
    doc_labels: list[str] = []
    per_doc_traces: list[dict[str, Any]] = []
    all_lemmas_flat: list[str] = []

    total_original = 0
    total_after_stopwords = 0
    total_after_pos = 0
    total_final = 0
    total_stop_builtin = 0
    total_stop_custom = 0

    spacy_docs = list(nlp.pipe([d["text"] for d in documents]))

    for doc_obj, doc_entry in zip(spacy_docs, documents):
        lemmas, token_details = _process_doc(
            doc_obj, pos_tag_set, builtin_stops, custom_stopwords
        )

        n_original = len([t for t in token_details])
        n_stop_builtin = sum(1 for t in token_details if t["is_stop_builtin"])
        n_stop_custom = sum(1 for t in token_details if t["is_stop_custom"])
        n_pos_match = sum(1 for t in token_details if t["pos_match"])
        n_final = len(lemmas)

        total_original += n_original
        total_stop_builtin += n_stop_builtin
        total_stop_custom += n_stop_custom
        total_after_stopwords += n_original - n_stop_builtin - n_stop_custom
        total_after_pos += n_pos_match
        total_final += n_final

        joined = " ".join(lemmas)

        if not joined.strip():
            empty_chunks_removed += 1
            logger.info("Empty after processing, removed: %s", doc_entry["label"])
            continue

        processed_texts.append(joined)
        doc_labels.append(doc_entry["label"])
        all_lemmas_flat.extend(lemmas)

        per_doc_traces.append(
            {
                "label": doc_entry["label"],
                "original_tokens": n_original,
                "stopwords_builtin": n_stop_builtin,
                "stopwords_custom": n_stop_custom,
                "pos_matched": n_pos_match,
                "final_lemmas": n_final,
                "token_details": token_details,
            }
        )

    # ── Warnings ──────────────────────────────────────────────────────────
    warnings: list[str] = []

    # Language mismatch (decision 58).
    if total_original > 0:
        ratio = total_final / total_original
        if ratio < LOW_TOKEN_RATIO_THRESHOLD:
            warnings.append(
                f"Low token ratio ({ratio:.1%}). "
                "The selected language may not match the corpus."
            )

    # Minimum corpus (decision 106).
    unique_lemmas = len(set(all_lemmas_flat))
    if unique_lemmas < MIN_UNIQUE_LEMMAS_WARNING:
        warnings.append(
            f"Only {unique_lemmas} unique lemmas found. "
            "Results may not be meaningful."
        )

    # Language mismatch warning for UI (decision 58).
    language_warning: str | None = None
    if total_original > 0:
        ratio = total_final / total_original
        if ratio < LOW_TOKEN_RATIO_THRESHOLD:
            pct = f"{ratio:.0%}"
            language_warning = (
                f"Low token recognition rate ({pct}). This may indicate a "
                "language mismatch. Check that the selected language "
                "matches your text."
            )

    # ── Trace ─────────────────────────────────────────────────────────────
    trace: dict[str, Any] = {
        "original_tokens": total_original,
        "after_stopwords": total_after_stopwords,
        "after_pos": total_after_pos,
        "final_lemmas": total_final,
        "unique_lemmas": unique_lemmas,
        "chunks_created": len(doc_labels),
        "empty_chunks_removed": empty_chunks_removed,
        "stopwords_removed_builtin": total_stop_builtin,
        "stopwords_removed_custom": total_stop_custom,
        "language": language,
        "pos_tags": pos_tags,
        "custom_stopwords": sorted(custom_stopwords),
        "warnings": warnings,
        "language_warning": language_warning,
        "per_document": per_doc_traces,
    }

    return processed_texts, doc_labels, trace


# ── Helpers ───────────────────────────────────────────────────────────────────


def _strip_extension(filename: str) -> str:
    """Remove file extension from a filename."""
    if "." in filename:
        return filename.rsplit(".", 1)[0]
    return filename
