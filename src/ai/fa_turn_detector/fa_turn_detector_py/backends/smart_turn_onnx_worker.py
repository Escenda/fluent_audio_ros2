from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fa_turn_detector_py.backends.smart_turn_onnx_runtime import SmartTurnOnnxRuntime


@dataclass(frozen=True)
class WorkerConfig:
    model_path: Path
    execution_provider: str
    audio_path: Path | None
    health_check: bool


def parse_args() -> WorkerConfig:
    parser = argparse.ArgumentParser(description="Smart Turn ONNX external worker")
    parser.add_argument("--model", required=True, help="Smart Turn ONNX model path")
    parser.add_argument("--provider", required=True, help="ONNX Runtime execution provider")
    parser.add_argument("--audio", default="", help="float32 .npy audio payload")
    parser.add_argument("--health-check", action="store_true", help="validate model contract only")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise RuntimeError(f"model file does not exist: {model_path}")
    provider = str(args.provider).strip()
    if not provider:
        raise RuntimeError("provider is required")
    audio_path = None
    if not bool(args.health_check):
        if not args.audio:
            raise RuntimeError("audio path is required")
        audio_path = Path(args.audio).expanduser()
        if not audio_path.is_file():
            raise RuntimeError(f"audio file does not exist: {audio_path}")

    return WorkerConfig(
        model_path=model_path,
        execution_provider=provider,
        audio_path=audio_path,
        health_check=bool(args.health_check),
    )


def load_audio_payload(audio_path: Path) -> np.ndarray:
    audio = np.load(audio_path, allow_pickle=False)
    if audio.dtype != np.float32:
        raise RuntimeError("audio payload must be float32")
    if audio.ndim != 1:
        raise RuntimeError("audio payload must be one-dimensional")
    if audio.size == 0:
        raise RuntimeError("audio payload is required")
    return audio


def main() -> int:
    config = parse_args()
    runtime = SmartTurnOnnxRuntime(
        model_path=config.model_path,
        execution_provider=config.execution_provider,
    )
    if config.health_check:
        print("ok")
        return 0
    if config.audio_path is None:
        raise RuntimeError("audio path is required")
    probability = runtime.detect_probability(load_audio_payload(config.audio_path))
    print(f"{probability:.8f}")
    return 0
