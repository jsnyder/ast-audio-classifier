"""Tests for AST classifier wrapper."""

import numpy as np
import pytest

from src.classifier import ASTClassifier, ClassificationResult


class TestClassificationResult:
    def test_fields(self):
        result = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[
                ("Dog", 0.85),
                ("Bark", 0.3),
                ("Speech", 0.1),
                ("Music", 0.05),
                ("Silence", 0.01),
            ],
            db_level=-25.0,
        )
        assert result.label == "Dog"
        assert result.group == "dog_bark"
        assert result.confidence == 0.85
        assert len(result.top_5) == 5
        assert result.db_level == -25.0

    def test_to_dict(self):
        result = ClassificationResult(
            label="Cat",
            group="cat_meow",
            confidence=0.72,
            top_5=[("Cat", 0.72), ("Meow", 0.5)],
            db_level=-30.0,
        )
        d = result.to_dict()
        assert d["label"] == "Cat"
        assert d["group"] == "cat_meow"
        assert d["confidence"] == 0.72
        assert isinstance(d["top_5"], list)
        assert d["db_level"] == -30.0
        # CLAP fields should be omitted when None/default
        assert "clap_verified" not in d
        assert "clap_score" not in d
        assert "clap_label" not in d
        assert "source" not in d

    def test_clap_fields_defaults(self):
        result = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[("Dog", 0.85)],
            db_level=-25.0,
        )
        assert result.clap_verified is None
        assert result.clap_score is None
        assert result.clap_label is None
        assert result.source == "ast"

    def test_clap_fields_in_to_dict(self):
        result = ClassificationResult(
            label="Dog",
            group="dog_bark",
            confidence=0.85,
            top_5=[("Dog", 0.85)],
            db_level=-25.0,
            clap_verified=True,
            clap_score=0.72,
            clap_label="a dog barking",
        )
        d = result.to_dict()
        assert d["clap_verified"] is True
        assert d["clap_score"] == 0.72
        assert d["clap_label"] == "a dog barking"
        assert "source" not in d  # Still "ast" default, omitted

    def test_clap_source_in_to_dict(self):
        result = ClassificationResult(
            label="a vacuum cleaner running",
            group="vacuum_cleaner",
            confidence=0.65,
            top_5=[],
            db_level=-25.0,
            clap_verified=True,
            clap_score=0.65,
            clap_label="a vacuum cleaner running",
            source="clap",
        )
        d = result.to_dict()
        assert d["source"] == "clap"


class TestASTClassifier:
    """Test classifier with mocked model to avoid downloading weights in CI."""

    @pytest.fixture()
    def mock_classifier(self):
        """Create a classifier with mocked transformers pipeline."""
        # Mock the pipeline so we don't need the actual model
        fake_outputs = [
            {"label": "Dog", "score": 0.85},
            {"label": "Bark", "score": 0.30},
            {"label": "Speech", "score": 0.10},
            {"label": "Music", "score": 0.05},
            {"label": "Silence", "score": 0.01},
        ]

        class FakePipeline:
            def __init__(self, *args, **kwargs):
                pass

            def __call__(self, audio, **kwargs):
                return [fake_outputs]

        clf = ASTClassifier.__new__(ASTClassifier)
        clf._pipeline = FakePipeline()
        clf._sample_rate = 16000
        clf._loaded = True
        return clf

    def test_classify_returns_results(self, mock_classifier):
        audio = np.random.randn(48000).astype(np.float32)  # 3 seconds at 16kHz
        results = mock_classifier.classify(audio, db_level=-25.0, threshold=0.1)

        assert len(results) > 0
        # Dog and Bark both map to dog_bark, should get best one
        dog_results = [r for r in results if r.group == "dog_bark"]
        assert len(dog_results) == 1
        assert dog_results[0].confidence == 0.85

    def test_classify_respects_threshold(self, mock_classifier):
        audio = np.random.randn(48000).astype(np.float32)
        # High threshold should filter out low-confidence results
        results = mock_classifier.classify(audio, db_level=-25.0, threshold=0.5)
        assert all(r.confidence >= 0.5 for r in results)

    def test_classify_empty_audio(self, mock_classifier):
        audio = np.array([], dtype=np.float32)
        results = mock_classifier.classify(audio, db_level=-60.0, threshold=0.1)
        assert results == []

    def test_top_5_included(self, mock_classifier):
        audio = np.random.randn(48000).astype(np.float32)
        results = mock_classifier.classify(audio, db_level=-25.0, threshold=0.01)
        for r in results:
            assert len(r.top_5) <= 5
            # Scores should be descending
            scores = [s for _, s in r.top_5]
            assert scores == sorted(scores, reverse=True)


class TestDBCalculation:
    def test_rms_db_silence(self):
        from src.audio_pipeline import compute_rms_db

        silence = np.zeros(16000, dtype=np.int16)
        db = compute_rms_db(silence)
        assert db == -96.0  # Floor value

    def test_rms_db_loud(self):
        from src.audio_pipeline import compute_rms_db

        # Full-scale signal
        loud = np.full(16000, 32767, dtype=np.int16)
        db = compute_rms_db(loud)
        assert db == pytest.approx(0.0, abs=0.1)

    def test_rms_db_mid(self):
        from src.audio_pipeline import compute_rms_db

        # ~50% amplitude
        mid = np.full(16000, 16384, dtype=np.int16)
        db = compute_rms_db(mid)
        assert -7.0 < db < -5.0  # ~-6 dB
