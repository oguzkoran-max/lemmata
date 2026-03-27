"""Tests for lemmata.config."""

from __future__ import annotations

from lemmata.config import (
    ALLOWED_EXTENSIONS,
    APP_VERSION,
    BG_MAIN,
    BG_SIDEBAR,
    CHUNK_SIZE_DEFAULT,
    CHUNK_SIZE_MAX,
    CHUNK_SIZE_MIN,
    COLOR_HEATMAP,
    COLOR_PALETTE,
    COLOR_PRIMARY,
    DEFAULT_POS_TAGS,
    ENCODING_FALLBACK_CHAIN,
    LDA_LEARNING_METHOD,
    LDA_MAX_ITER,
    LOW_TOKEN_RATIO_THRESHOLD,
    MAX_FILE_SIZE_MB,
    MAX_TOTAL_SIZE_MB,
    MIN_UNIQUE_LEMMAS_WARNING,
    MODEL_SIZE,
    NGRAM_RANGE,
    NUM_TOPICS_DEFAULT,
    NUM_TOPICS_MAX,
    NUM_TOPICS_MIN,
    POS_PRESETS,
    RANDOM_SEED,
    SUPPORTED_LANGUAGES,
    VECTORIZER_TYPE,
    WORDS_PER_TOPIC_DEFAULT,
    WORDS_PER_TOPIC_MAX,
    WORDS_PER_TOPIC_MIN,
    get_df_auto,
)


class TestConstants:
    """Verify all constants exist with correct types."""

    def test_app_version_is_string(self):
        assert isinstance(APP_VERSION, str)
        assert APP_VERSION == "0.1.0"

    def test_supported_languages(self):
        assert isinstance(SUPPORTED_LANGUAGES, dict)
        assert len(SUPPORTED_LANGUAGES) == 5
        expected = {"it", "en", "de", "fr", "es"}
        assert set(SUPPORTED_LANGUAGES.keys()) == expected

    def test_spacy_model_names(self):
        for lang, model in SUPPORTED_LANGUAGES.items():
            assert isinstance(model, str)
            assert model.endswith("_sm"), f"{lang} model should be sm: {model}"
            assert lang in model or (lang == "en" and "web" in model)

    def test_chunk_size_range(self):
        assert isinstance(CHUNK_SIZE_MIN, int)
        assert isinstance(CHUNK_SIZE_MAX, int)
        assert isinstance(CHUNK_SIZE_DEFAULT, int)
        assert CHUNK_SIZE_MIN < CHUNK_SIZE_DEFAULT < CHUNK_SIZE_MAX
        assert CHUNK_SIZE_MIN == 300
        assert CHUNK_SIZE_MAX == 3000
        assert CHUNK_SIZE_DEFAULT == 1000

    def test_topic_count_range(self):
        assert NUM_TOPICS_MIN < NUM_TOPICS_DEFAULT < NUM_TOPICS_MAX
        assert NUM_TOPICS_MIN == 2
        assert NUM_TOPICS_MAX == 15

    def test_words_per_topic_range(self):
        assert WORDS_PER_TOPIC_MIN < WORDS_PER_TOPIC_DEFAULT < WORDS_PER_TOPIC_MAX

    def test_random_seed(self):
        assert RANDOM_SEED == 42

    def test_lda_params(self):
        assert LDA_LEARNING_METHOD == "batch"
        assert isinstance(LDA_MAX_ITER, int)
        assert LDA_MAX_ITER > 0

    def test_file_size_limits(self):
        assert isinstance(MAX_FILE_SIZE_MB, int)
        assert isinstance(MAX_TOTAL_SIZE_MB, int)
        assert MAX_FILE_SIZE_MB == 50
        assert MAX_TOTAL_SIZE_MB == 100

    def test_default_pos_tags(self):
        assert isinstance(DEFAULT_POS_TAGS, list)
        assert DEFAULT_POS_TAGS == ["NOUN", "PROPN", "ADJ"]

    def test_allowed_extensions(self):
        assert set(ALLOWED_EXTENSIONS) == {"txt", "pdf", "odt", "docx", "epub"}

    def test_color_constants(self):
        assert COLOR_PRIMARY == "#0F6E56"
        assert isinstance(COLOR_PALETTE, str)
        assert isinstance(COLOR_HEATMAP, str)
        assert BG_MAIN.startswith("#")
        assert BG_SIDEBAR.startswith("#")

    def test_future_ready_constants(self):
        assert NGRAM_RANGE == (1, 1)
        assert MODEL_SIZE == "sm"
        assert VECTORIZER_TYPE == "count"

    def test_encoding_fallback(self):
        assert isinstance(ENCODING_FALLBACK_CHAIN, list)
        assert "utf-8" in ENCODING_FALLBACK_CHAIN

    def test_thresholds(self):
        assert 0 < LOW_TOKEN_RATIO_THRESHOLD < 1
        assert isinstance(MIN_UNIQUE_LEMMAS_WARNING, int)


class TestPosPresets:
    """Verify POS preset dictionary."""

    def test_has_four_presets(self):
        assert len(POS_PRESETS) == 4

    def test_preset_names(self):
        expected = {"Content words", "Content + verbs", "All open class", "Custom"}
        assert set(POS_PRESETS.keys()) == expected

    def test_content_words_matches_default(self):
        assert POS_PRESETS["Content words"] == DEFAULT_POS_TAGS

    def test_custom_is_empty(self):
        assert POS_PRESETS["Custom"] == []

    def test_all_values_are_lists(self):
        for name, tags in POS_PRESETS.items():
            assert isinstance(tags, list), f"{name} should be a list"


class TestGetDfAuto:
    """Verify auto df threshold selection."""

    def test_small_corpus(self):
        min_df, max_df = get_df_auto(5)
        assert min_df == 1
        assert max_df == 0.95

    def test_medium_corpus(self):
        min_df, max_df = get_df_auto(30)
        assert min_df == 2
        assert max_df == 0.90

    def test_large_corpus(self):
        min_df, max_df = get_df_auto(200)
        assert min_df == 5
        assert max_df == 0.80

    def test_very_large_corpus(self):
        min_df, max_df = get_df_auto(50_000)
        assert min_df == 10
        assert max_df == 0.75

    def test_returns_tuple(self):
        result = get_df_auto(10)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_max_df_always_less_than_one(self):
        for n in [1, 5, 10, 50, 100, 500, 10_000, 100_000]:
            _, max_df = get_df_auto(n)
            assert 0 < max_df <= 1.0

    def test_boundary_values(self):
        # Exactly at boundary should use that tier.
        min_df_10, _ = get_df_auto(10)
        min_df_11, _ = get_df_auto(11)
        assert min_df_10 == 1
        assert min_df_11 == 2
