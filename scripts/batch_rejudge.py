#!/usr/bin/env python3
"""Batch re-judge existing audio clips with a second LLM model.

Reads WAV clips from the clip directory, sends each to a multimodal LLM
(e.g. Gemini 3 Flash via LiteLLM), and writes JSON sidecar files with
the verdict alongside each clip.

Usage:
    python3 scripts/batch_rejudge.py \
        --clip-dir /media/ast-audio-classifier/clips \
        --api-base https://litellm.5745.house/v1 \
        --api-key sk-... \
        --model gemini-3-flash-preview \
        [--concurrency 3] \
        [--skip-existing]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# Regex to parse clip filenames: {camera}_{timestamp}_{group}.wav
CLIP_RE = re.compile(r"^(.+?)_(\d{8}T\d{6})_(.+)\.wav$")


def parse_clip_filename(filename: str) -> dict | None:
    """Extract camera, timestamp, and group from a clip filename."""
    m = CLIP_RE.match(filename)
    if not m:
        return None
    return {
        "camera": m.group(1),
        "timestamp": m.group(2),
        "group": m.group(3),
    }


def build_prompt(camera: str, group: str) -> str:
    """Build the evaluation prompt for the LLM judge."""
    return f"""You are an audio classification judge. Listen to this audio clip and evaluate whether the classification is correct.

Camera: {camera}
Detected sound: {group}

Respond with a JSON object containing:
1. "verdict": "correct", "incorrect", or "plausible"
2. "actual_sound": What you actually hear in the audio (be specific)
3. "confidence": Your confidence in the verdict (0.0-1.0)
4. "notes": Brief explanation of what you hear and why you agree or disagree

Additionally, list ALL distinct sounds you can identify in the clip:
5. "all_sounds": Array of objects, each with "sound" (what it is) and "confidence" (0.0-1.0)

Respond ONLY with valid JSON, no markdown formatting."""


def strip_markdown(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


async def judge_clip(
    client,
    model: str,
    wav_path: str,
    camera: str,
    group: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Send a WAV clip to the LLM and return the verdict."""
    async with semaphore:
        with open(wav_path, "rb") as f:
            wav_bytes = f.read()
        audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
        prompt = build_prompt(camera, group)

        try:
            response = await client.chat.completions.create(
                model=model,
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
            try:
                verdict = json.loads(strip_markdown(response_text))
            except (json.JSONDecodeError, TypeError):
                verdict = {
                    "verdict": "error",
                    "actual_sound": "unknown",
                    "confidence": 0.0,
                    "notes": f"Failed to parse: {response_text[:200]}",
                }
            return verdict
        except Exception as e:
            return {
                "verdict": "error",
                "actual_sound": "unknown",
                "confidence": 0.0,
                "notes": f"API error: {e!s}",
            }


async def process_clip(
    client,
    model: str,
    wav_path: str,
    skip_existing: bool,
    semaphore: asyncio.Semaphore,
    stats: dict,
) -> None:
    """Process a single clip: judge it and write sidecar."""
    filename = os.path.basename(wav_path)
    parsed = parse_clip_filename(filename)
    if not parsed:
        logger.warning("Cannot parse filename: %s", filename)
        stats["skipped"] += 1
        return

    sidecar_path = wav_path.rsplit(".", 1)[0] + ".json"

    # Check for existing sidecar
    if skip_existing and os.path.exists(sidecar_path):
        # Check if this model already judged it
        try:
            with open(sidecar_path) as f:
                existing = json.load(f)
            if model in [v.get("judge_model") for v in existing.get("rejudge_verdicts", [])]:
                stats["skipped"] += 1
                return
        except (json.JSONDecodeError, OSError):
            pass

    verdict = await judge_clip(
        client, model, wav_path, parsed["camera"], parsed["group"], semaphore,
    )

    # Read or create sidecar
    sidecar = {}
    if os.path.exists(sidecar_path):
        try:
            with open(sidecar_path) as f:
                sidecar = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Add/update rejudge verdicts
    if "rejudge_verdicts" not in sidecar:
        sidecar["rejudge_verdicts"] = []

    sidecar["rejudge_verdicts"].append({
        "judge_model": model,
        "verdict": verdict.get("verdict", "unknown"),
        "actual_sound": verdict.get("actual_sound", "unknown"),
        "confidence": verdict.get("confidence", 0.0),
        "notes": verdict.get("notes", ""),
        "all_sounds": verdict.get("all_sounds", []),
    })

    # Also populate top-level fields if not already present
    if "camera" not in sidecar:
        sidecar["camera"] = parsed["camera"]
    if "original_group" not in sidecar:
        sidecar["original_group"] = parsed["group"]
    if "timestamp" not in sidecar:
        sidecar["timestamp"] = parsed["timestamp"]

    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    verdict_str = verdict.get("verdict", "unknown")
    actual = verdict.get("actual_sound", "?")
    stats[verdict_str] = stats.get(verdict_str, 0) + 1
    stats["processed"] += 1

    if stats["processed"] % 50 == 0:
        logger.info(
            "Progress: %d/%d processed, %d skipped",
            stats["processed"], stats["total"], stats["skipped"],
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description="Batch re-judge audio clips")
    parser.add_argument("--clip-dir", required=True, help="Directory containing WAV clips")
    parser.add_argument("--api-base", required=True, help="LLM API base URL")
    parser.add_argument("--api-key", required=True, help="LLM API key")
    parser.add_argument("--model", required=True, help="Model name (e.g. gemini-3-flash-preview)")
    parser.add_argument("--concurrency", type=int, default=3, help="Max concurrent API calls")
    parser.add_argument("--skip-existing", action="store_true", help="Skip clips already judged by this model")
    parser.add_argument("--limit", type=int, default=0, help="Max clips to process (0 = all)")
    args = parser.parse_args()

    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        base_url=args.api_base,
        api_key=args.api_key,
        timeout=60.0,
    )

    # Find all WAV clips
    wav_files = sorted(
        [
            os.path.join(args.clip_dir, f)
            for f in os.listdir(args.clip_dir)
            if f.endswith(".wav")
        ]
    )

    if args.limit > 0:
        wav_files = wav_files[:args.limit]

    logger.info("Found %d WAV clips in %s", len(wav_files), args.clip_dir)
    logger.info("Model: %s, Concurrency: %d", args.model, args.concurrency)

    semaphore = asyncio.Semaphore(args.concurrency)
    stats = {"processed": 0, "skipped": 0, "total": len(wav_files)}

    start = time.monotonic()

    # Process in batches to avoid overwhelming the API
    tasks = [
        process_clip(client, args.model, wav_path, args.skip_existing, semaphore, stats)
        for wav_path in wav_files
    ]
    await asyncio.gather(*tasks)

    elapsed = time.monotonic() - start
    logger.info("Done in %.1fs", elapsed)
    logger.info("Processed: %d, Skipped: %d", stats["processed"], stats["skipped"])

    # Print verdict breakdown
    for key in ["correct", "incorrect", "plausible", "error"]:
        if key in stats:
            logger.info("  %s: %d", key, stats[key])


if __name__ == "__main__":
    asyncio.run(main())
