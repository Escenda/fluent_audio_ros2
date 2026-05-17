#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--sample-rate", required=True, type=int)
    parser.add_argument("--probability", default="")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    model_path = Path(args.model)
    if not audio_path.is_file():
        raise RuntimeError(f"missing audio: {audio_path}")
    if not model_path.is_dir():
        raise RuntimeError(f"missing model: {model_path}")
    if args.provider != "cpu":
        raise RuntimeError(f"unexpected provider: {args.provider}")

    audio_bytes = audio_path.read_bytes()
    if not audio_bytes:
        raise RuntimeError("empty audio")
    if len(audio_bytes) % 4 != 0:
        raise RuntimeError("expected raw float32le")

    probability_text = str(args.probability).strip()
    probability_file = model_path / "probability.txt"
    if not probability_text and probability_file.is_file():
        probability_text = probability_file.read_text(encoding="utf-8").strip()
    if not probability_text:
        probability_text = "0.75"
    probability = float(probability_text)
    if probability < 0.0 or probability > 1.0:
        raise RuntimeError("probability must be in [0.0, 1.0]")

    print(f"{probability:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
