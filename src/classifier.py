"""AST model wrapper — load once, classify many.

Uses the MIT Audio Spectrogram Transformer (0.459 mAP on AudioSet)
via the Hugging Face transformers pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .labels import get_top_group_match

logger = logging.getLogger(__name__)

MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000


@dataclass
class ClassificationResult:
    """A single classification result with group mapping."""

    label: str
    group: str
    confidence: float
    top_5: list[tuple[str, float]]
    db_level: float
    # CLAP verification fields (None when CLAP is disabled)
    clap_verified: bool | None = None
    clap_score: float | None = None
    clap_label: str | None = None
    source: str = "ast"
    # Confounder fields (set when a camera confounder is active for this group)
    confounded: bool = False
    confounder_entity: str | None = None

    def to_dict(self) -> dict:
        d = {
            "label": self.label,
            "group": self.group,
            "confidence": self.confidence,
            "top_5": self.top_5,
            "db_level": self.db_level,
        }
        # Include CLAP fields only when set (keeps payloads clean when CLAP disabled)
        if self.clap_verified is not None:
            d["clap_verified"] = self.clap_verified
        if self.clap_score is not None:
            d["clap_score"] = self.clap_score
        if self.clap_label is not None:
            d["clap_label"] = self.clap_label
        if self.source != "ast":
            d["source"] = self.source
        if self.confounded:
            d["confounded"] = True
            if self.confounder_entity:
                d["confounder"] = self.confounder_entity
        return d


class ASTClassifier:
    """Wraps the AST model for audio classification.

    Loads the model once at startup and reuses it for all inferences.
    Thread-safe when called via asyncio.to_thread with a semaphore.
    """

    def __init__(self, model_id: str = MODEL_ID) -> None:
        from transformers import pipeline as hf_pipeline

        logger.info("Loading AST model: %s", model_id)
        self._pipeline = hf_pipeline(
            "audio-classification",
            model=model_id,
            device=-1,  # CPU
        )
        self._sample_rate = SAMPLE_RATE
        self._loaded = True
        logger.info("AST model loaded successfully")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    def classify(
        self,
        audio: np.ndarray,
        db_level: float,
        threshold: float = 0.15,
        *,
        group_thresholds: dict[str, float] | None = None,
        disabled_groups: set[str] | None = None,
    ) -> list[ClassificationResult]:
        """Classify an audio clip and return grouped results.

        Args:
            audio: Float32 numpy array of audio samples at 16kHz.
            db_level: The dB level that triggered this classification.
            threshold: Minimum confidence for a group to be included.
            group_thresholds: Per-group confidence overrides.
            disabled_groups: Groups to exclude from results.

        Returns:
            List of ClassificationResult for groups above threshold.
        """
        if len(audio) == 0:
            return []

        # AST expects float32 audio at 16kHz
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        raw_results = self._pipeline(
            audio,
            sampling_rate=self._sample_rate,
            top_k=20,
        )

        # raw_results is list of dicts with 'label' and 'score'
        if (
            isinstance(raw_results, list)
            and raw_results
            and isinstance(raw_results[0], list)
        ):
            raw_results = raw_results[0]

        predictions = [(r["label"], r["score"]) for r in raw_results]
        top_5_global = [(r["label"], round(r["score"], 4)) for r in raw_results[:5]]

        matches = get_top_group_match(
            predictions,
            threshold=threshold,
            all_groups=True,
            group_thresholds=group_thresholds,
            disabled_groups=disabled_groups,
        )
        if not matches:
            return []

        return [
            ClassificationResult(
                label=raw_label,
                group=group,
                confidence=round(confidence, 4),
                top_5=top_5_global,
                db_level=round(db_level, 1),
            )
            for group, confidence, raw_label in matches
        ]
