"""LLM-based audio classification judge for ground-truth evaluation.

Randomly samples detections, saves audio clips as WAV files, sends them
to a multimodal LLM for independent classification, and logs the verdict
to OpenObserve alongside AST/CLAP results.

Non-blocking: runs async in the background, never delays MQTT publish.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import random
import re
import wave
from datetime import UTC, datetime

import numpy as np

from .classifier import ClassificationResult
from .config import LLMJudgeConfig
from .openobserve import log_event

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

_SAFE_NAME_RE = re.compile(r"[^\w\-]")


def _safe_name(name: str) -> str:
    """Strip characters unsafe for filenames."""
    return _SAFE_NAME_RE.sub("_", name)


class LLMJudge:
    """Evaluates audio classifications using a multimodal LLM."""

    def __init__(self, config: LLMJudgeConfig) -> None:
        from openai import AsyncOpenAI

        self._config = config
        self._client = AsyncOpenAI(
            base_url=config.api_base,
            api_key=config.api_key,
            timeout=config.timeout_seconds,
        )
        self._prune_lock = asyncio.Lock()
        self._eval_count = 0
        self._PRUNE_INTERVAL = 100  # Only scan directory every N evaluations

    def should_sample(self) -> bool:
        """Decide whether to sample this detection based on configured rate."""
        return random.random() < self._config.sample_rate

    @staticmethod
    def _encode_wav(audio_16k: np.ndarray) -> bytes:
        """Encode float32 audio as 16-bit PCM WAV bytes."""
        audio_int16 = np.clip(audio_16k * 32768.0, -32768, 32767).astype(np.int16)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_int16.tobytes())
        return buf.getvalue()

    def _save_wav(
        self, audio_16k: np.ndarray, camera_name: str, group: str
    ) -> tuple[str, bytes]:
        """Save float32 audio as a 16-bit PCM WAV file.

        Returns (path, wav_bytes) so the caller can reuse the bytes for the API.
        """
        os.makedirs(self._config.clip_dir, exist_ok=True)

        wav_bytes = self._encode_wav(audio_16k)
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        safe_camera = _safe_name(camera_name)
        safe_group = _safe_name(group)
        filename = f"{safe_camera}_{timestamp}_{safe_group}.wav"
        path = os.path.join(self._config.clip_dir, filename)

        # Verify the resolved path stays inside clip_dir
        clip_dir = os.path.realpath(self._config.clip_dir)
        resolved = os.path.realpath(path)
        if not resolved.startswith(clip_dir + os.sep) and resolved != clip_dir:
            msg = f"Resolved path {resolved!r} escapes clip_dir {clip_dir!r}"
            raise ValueError(msg)

        with open(path, "wb") as f:
            f.write(wav_bytes)

        return path, wav_bytes

    def _build_prompt(
        self, results: list[ClassificationResult], camera_name: str
    ) -> str:
        """Build the evaluation prompt with AST/CLAP context."""
        classifications = []
        for r in results:
            entry = (
                f"- Group: {r.group}, AST confidence: {r.confidence}, "
                f"CLAP verified: {r.clap_verified}, CLAP score: {r.clap_score}"
            )
            classifications.append(entry)

        classifications_text = "\n".join(classifications)

        return f"""You are an audio classification judge. Listen to this audio clip \
and evaluate the classifications below.

Camera: {camera_name}
Classifications:
{classifications_text}

For each classification, respond with a JSON object containing a "verdicts" array. \
Each verdict should have:
1. "group": The group name being evaluated
2. "verdict": "correct", "incorrect", or "plausible"
3. "actual_sound": What you actually hear in the audio
4. "confidence": Your confidence in the verdict (0.0-1.0)
5. "notes": Brief explanation

Respond ONLY with valid JSON, no markdown formatting."""

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Strip markdown code fences from LLM response."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def _parse_response(self, response_text: str) -> list[dict]:
        """Extract structured verdicts from LLM response text."""
        try:
            data = json.loads(self._strip_markdown(response_text))
        except (json.JSONDecodeError, TypeError):
            return [{
                "group": "unknown",
                "verdict": "error",
                "actual_sound": "unknown",
                "confidence": 0.0,
                "notes": f"Failed to parse LLM response: {response_text[:200]}",
            }]

        # Handle both {"verdicts": [...]} and single-object responses
        if isinstance(data, dict):
            verdicts_raw = data.get("verdicts", [data])
        elif isinstance(data, list):
            verdicts_raw = data
        else:
            return [{
                "group": "unknown",
                "verdict": "error",
                "actual_sound": "unknown",
                "confidence": 0.0,
                "notes": f"Unexpected response format: {type(data).__name__}",
            }]

        verdicts = []
        for v in verdicts_raw:
            raw_conf = v.get("confidence", 0.0)
            confidence = float(raw_conf) if isinstance(raw_conf, (int, float)) else 0.0
            verdicts.append({
                "group": v.get("group", "unknown"),
                "verdict": v.get("verdict", "unknown"),
                "actual_sound": v.get("actual_sound", "unknown"),
                "confidence": confidence,
                "notes": v.get("notes", ""),
            })

        return verdicts

    def _prune_clips(self) -> None:
        """Delete oldest WAV files when clip count exceeds max_clips."""
        clip_dir = self._config.clip_dir
        if not os.path.isdir(clip_dir):
            return

        try:
            entries = [
                e for e in os.scandir(clip_dir)
                if e.name.endswith(".wav") and e.is_file()
            ]
        except OSError:
            return

        if len(entries) <= self._config.max_clips:
            return

        # Sort by modification time, oldest first
        entries.sort(key=lambda e: e.stat().st_mtime)
        to_delete = len(entries) - self._config.max_clips
        for entry in entries[:to_delete]:
            try:
                os.remove(entry.path)
            except OSError:
                logger.debug("Failed to prune clip: %s", entry.path)

        logger.debug("Pruned %d old clips from %s", to_delete, clip_dir)

    def _should_prune(self) -> bool:
        """Check if pruning is needed without a full directory scan.

        Only triggers a real prune every _PRUNE_INTERVAL evaluations.
        """
        self._eval_count += 1
        return self._eval_count % self._PRUNE_INTERVAL == 0

    async def evaluate(
        self,
        audio_16k: np.ndarray,
        results: list[ClassificationResult],
        camera_name: str,
    ) -> None:
        """Evaluate classifications via LLM and log results.

        This is fire-and-forget — exceptions are logged, not propagated.
        """
        try:
            # Save WAV clip and get encoded bytes in one pass
            primary_group = results[0].group if results else "unknown"
            clip_path, wav_bytes = self._save_wav(audio_16k, camera_name, primary_group)
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

            # Build prompt
            prompt = self._build_prompt(results, camera_name)

            # Call LLM
            response = await self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_b64,
                                    "format": "wav",
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                temperature=0.1,
            )

            response_text = response.choices[0].message.content or ""
            verdicts = self._parse_response(response_text)

            # Log each verdict to OpenObserve — match by group name, not position
            results_by_group = {r.group: r for r in results}
            for verdict in verdicts:
                verdict_group = verdict.get("group", "unknown")
                result = results_by_group.get(verdict_group, results[0])
                log_event(
                    "llm_judge",
                    camera=camera_name,
                    group=result.group,
                    ast_confidence=result.confidence,
                    clap_score=result.clap_score,
                    clap_verified=result.clap_verified,
                    llm_verdict=verdict["verdict"],
                    llm_actual_sound=verdict["actual_sound"],
                    llm_confidence=verdict["confidence"],
                    llm_notes=verdict["notes"],
                    llm_model=self._config.model,
                    clip_path=clip_path,
                )

            # Prune old clips periodically (not every call)
            if self._should_prune():
                async with self._prune_lock:
                    await asyncio.to_thread(self._prune_clips)

            logger.debug(
                "[%s] LLM judge: %s → %s",
                camera_name,
                primary_group,
                verdicts[0]["verdict"] if verdicts else "no_verdict",
            )

        except Exception:
            logger.warning(
                "[%s] LLM judge evaluation failed", camera_name, exc_info=True
            )
