#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct


@dataclass(frozen=True)
class WorkerConfig:
    audio_path: Path
    model_id: str
    sample_rate: int
    dimension: int
    source_id: str
    stream_id: str


def parse_args() -> WorkerConfig:
    parser = argparse.ArgumentParser(description="Fake audio embedding worker")
    parser.add_argument("--audio", required=True, help="Mono raw float32le audio path")
    parser.add_argument("--model-id", required=True, help="Model id to validate")
    parser.add_argument("--sample-rate", required=True, type=int, help="Expected sample rate")
    parser.add_argument("--dimension", required=True, type=int, help="Embedding dimension")
    parser.add_argument("--source-id", default="", help="Optional source identity")
    parser.add_argument("--stream-id", default="", help="Optional stream identity")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser()
    if not audio_path.is_file():
        raise RuntimeError(f"audio file does not exist: {audio_path}")
    if not str(args.model_id).strip():
        raise RuntimeError("model-id is required")
    if int(args.sample_rate) <= 0:
        raise RuntimeError("sample-rate must be > 0")
    if int(args.dimension) <= 0:
        raise RuntimeError("dimension must be > 0")
    if not str(args.source_id).strip():
        raise RuntimeError("source-id is required")
    if not str(args.stream_id).strip():
        raise RuntimeError("stream-id is required")
    return WorkerConfig(
        audio_path=audio_path,
        model_id=str(args.model_id),
        sample_rate=int(args.sample_rate),
        dimension=int(args.dimension),
        source_id=str(args.source_id),
        stream_id=str(args.stream_id),
    )


def read_audio(config: WorkerConfig) -> tuple[float, ...]:
    audio_bytes = config.audio_path.read_bytes()
    if not audio_bytes:
        raise RuntimeError("audio file is empty")
    if len(audio_bytes) % 4 != 0:
        raise RuntimeError("audio file must be raw float32le")
    samples = tuple(value for (value,) in struct.iter_unpack("<f", audio_bytes))
    for sample in samples:
        if not (-1.0 <= sample <= 1.0):
            raise RuntimeError("audio samples must be normalized to [-1.0, 1.0]")
    return samples


def main() -> int:
    config = parse_args()
    samples = read_audio(config)
    mean_abs = sum(abs(sample) for sample in samples) / len(samples)
    embedding = [mean_abs + (index * 0.01) for index in range(config.dimension)]
    print(" ".join(f"{value:.8f}" for value in embedding))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
