#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--sample-rate", required=True, type=int)
    parser.add_argument("--expected-sample", required=True, type=float)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    if not audio_path.is_file():
        raise RuntimeError(f"missing audio: {audio_path}")
    if not model_path.is_file():
        raise RuntimeError(f"missing model: {model_path}")
    if not args.language:
        raise RuntimeError("language is required")
    if args.sample_rate <= 0:
        raise RuntimeError("sample rate must be positive")

    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        raise RuntimeError("empty audio")
    if len(audio_bytes) % 4 != 0:
        raise RuntimeError("expected float32le audio")
    samples = [sample for (sample,) in struct.iter_unpack("<f", audio_bytes)]
    if not samples:
        raise RuntimeError("empty float32le audio")
    if not all(math.isfinite(sample) for sample in samples):
        raise RuntimeError("expected finite float32le audio")
    if not all(abs(sample - args.expected_sample) <= 1.0e-7 for sample in samples):
        raise RuntimeError("unexpected float32le sample values")

    transcript = model_path.read_text(encoding="utf-8").strip()
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(transcript, encoding="utf-8")
    else:
        print(transcript)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
