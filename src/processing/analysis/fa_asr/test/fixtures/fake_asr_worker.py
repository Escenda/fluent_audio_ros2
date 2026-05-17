#!/usr/bin/env python3
from __future__ import annotations

import argparse
import wave
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", required=True)
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

    with wave.open(str(audio_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise RuntimeError("expected mono audio")
        if wav_file.getsampwidth() != 2:
            raise RuntimeError("expected PCM16 audio")
        if wav_file.getframerate() <= 0:
            raise RuntimeError("invalid sample rate")
        if wav_file.getnframes() == 0:
            raise RuntimeError("empty audio")

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
