#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--sample-rate", required=True, type=int)
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
