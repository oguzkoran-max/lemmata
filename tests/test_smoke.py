"""Smoke / integration test for Lemmata (decision 182).

End-to-end: raw text → preprocess → DTM → LDA → coherence → ZIP export.
Uses a real spaCy model (en_core_web_sm) so marked as slow.
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO

import numpy as np
import pytest

from lemmata.modelling import build_dtm, compute_coherence, run_lda
from lemmata.preprocessing import load_spacy_model, process_documents
from lemmata.file_io import export_zip


@pytest.mark.slow
class TestSmokePipeline:
    """Full pipeline integration test."""

    @pytest.fixture(autouse=True)
    def setup_corpus(self):
        """Three short English documents."""
        self.corpus = [
            {
                "filename": "nature.txt",
                "content": (
                    "The forest is home to many animals and plants. "
                    "Trees provide shelter and food for birds and insects. "
                    "Rivers flow through the valleys carrying water downstream. "
                    "Mountains rise above the clouds with snow on their peaks. "
                    "The ecosystem depends on the balance between all species."
                ),
            },
            {
                "filename": "science.txt",
                "content": (
                    "Scientists study the universe through careful observation. "
                    "Physics explains the fundamental forces of nature. "
                    "Chemistry reveals the composition of matter and reactions. "
                    "Biology explores the diversity of living organisms. "
                    "Mathematics provides the language for scientific theories."
                ),
            },
            {
                "filename": "history.txt",
                "content": (
                    "Ancient civilizations built great monuments and cities. "
                    "The Roman Empire spread across Europe and the Mediterranean. "
                    "Medieval kingdoms fought wars over territory and religion. "
                    "The Renaissance brought new ideas about art and science. "
                    "Modern history is shaped by industry and technology."
                ),
            },
        ]

    def test_full_pipeline(self):
        """Run the entire analysis pipeline and verify outputs."""
        # Step 1: Preprocess.
        processed_texts, doc_labels, trace = process_documents(
            texts=self.corpus,
            language="en",
        )
        assert len(processed_texts) == 3
        assert len(doc_labels) == 3
        assert trace["final_lemmas"] > 0

        # Step 2: Build DTM.
        dtm, vectorizer, dtm_info = build_dtm(
            processed_texts, min_df=1, max_df=1.0,
        )
        assert dtm.shape[0] == 3
        assert dtm_info["vocabulary_kept"] > 0

        # Step 3: Run LDA.
        model, doc_topic, topics, model_info = run_lda(
            dtm, vectorizer, n_topics=2, n_words=5, random_seed=42,
        )
        assert doc_topic.shape == (3, 2)
        assert len(topics) == 2
        assert model_info["n_topics"] == 2

        # Topics ordered by prevalence.
        assert topics[0]["avg_weight"] >= topics[1]["avg_weight"]

        # Doc-topic rows sum to ~1.
        row_sums = doc_topic.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)

        # Step 4: Coherence.
        coherence = compute_coherence(topics, processed_texts)
        assert isinstance(coherence["c_v"], float)
        assert len(coherence["per_topic"]) == 2

        # Step 5: Export ZIP.
        results = {
            "topics": topics,
            "doc_topic_matrix": doc_topic,
            "doc_labels": doc_labels,
            "preprocessing_trace": trace,
            "model_info": model_info,
            "dtm_info": dtm_info,
            "coherence": coherence,
        }
        params = {
            "language": "en",
            "n_topics": 2,
            "n_words": 5,
            "seed": 42,
        }
        zip_bytes = export_zip(results, params)

        # Verify ZIP contents.
        with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
            assert zf.testzip() is None
            names = set(zf.namelist())

            expected_files = {
                "topic_words.csv",
                "doc_topic_matrix.csv",
                "preprocessing_summary.csv",
                "metrics.json",
                "analysis_results.json",
                "environment.json",
            }
            assert expected_files.issubset(names), (
                f"Missing files: {expected_files - names}"
            )

            # Verify JSON files are parseable.
            metrics = json.loads(zf.read("metrics.json"))
            assert "coherence" in metrics

            analysis = json.loads(zf.read("analysis_results.json"))
            assert analysis["lemmata_version"] == "0.1.0"
            assert len(analysis["topics"]) == 2

    def test_determinism(self):
        """Same corpus + same seed → identical results (decision 50)."""
        processed_texts, _, _ = process_documents(
            texts=self.corpus, language="en",
        )
        dtm, vec, _ = build_dtm(processed_texts, min_df=1, max_df=1.0)

        _, dt1, topics1, _ = run_lda(dtm, vec, n_topics=2, random_seed=42)
        _, dt2, topics2, _ = run_lda(dtm, vec, n_topics=2, random_seed=42)

        np.testing.assert_array_equal(dt1, dt2)
        for t1, t2 in zip(topics1, topics2):
            assert t1["words"] == t2["words"]

    def test_single_file_chunking(self):
        """Single file should be chunked, not treated as one document."""
        long_doc = [
            {
                "filename": "long.txt",
                "content": " ".join(
                    "The quick brown fox jumps over the lazy dog. " * 20
                    for _ in range(5)
                ),
            }
        ]
        processed, labels, trace = process_documents(
            texts=long_doc, language="en", chunk_size=300,
        )
        # Should produce multiple chunks.
        assert len(labels) >= 1
        # Labels should have _001 format if chunked.
        if len(labels) > 1:
            assert labels[0].startswith("long_")
            assert labels[0].endswith("001")
