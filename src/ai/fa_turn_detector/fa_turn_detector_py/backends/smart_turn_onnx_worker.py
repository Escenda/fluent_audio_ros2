from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from fa_turn_detector_py.backends.smart_turn_onnx_runtime import SmartTurnOnnxRuntime


@dataclass(frozen=True)
class HealthCheckConfig:
    model_path: Path
    execution_provider: str


@dataclass(frozen=True)
class DetectionConfig:
    model_path: Path
    execution_provider: str
    audio_path: Path


def parse_args() -> HealthCheckConfig | DetectionConfig:
    parser = argparse.ArgumentParser(description="Smart Turn ONNX external worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser("health", help="validate model contract")
    health_parser.add_argument("--model", required=True, help="Smart Turn ONNX model path")
    health_parser.add_argument(
        "--provider",
        required=True,
        help="ONNX Runtime execution provider",
    )

    detect_parser = subparsers.add_parser("detect", help="run turn detection")
    detect_parser.add_argument("--model", required=True, help="Smart Turn ONNX model path")
    detect_parser.add_argument(
        "--provider",
        required=True,
        help="ONNX Runtime execution provider",
    )
    detect_parser.add_argument(
        "--audio",
        required=True,
        help="float32 .npy audio payload",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.is_file():
        raise RuntimeError(f"model file does not exist: {model_path}")
    provider = str(args.provider).strip()
    if not provider:
        raise RuntimeError("provider is required")

    if args.command == "health":
        return HealthCheckConfig(
            model_path=model_path,
            execution_provider=provider,
        )
    if args.command == "detect":
        audio_path = Path(args.audio).expanduser()
        if not audio_path.is_file():
            raise RuntimeError(f"audio file does not exist: {audio_path}")
        return DetectionConfig(
            model_path=model_path,
            execution_provider=provider,
            audio_path=audio_path,
        )

    raise RuntimeError(f"unsupported command: {args.command}")


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
    if isinstance(config, HealthCheckConfig):
        print("ok")
        return 0
    probability = runtime.detect_probability(load_audio_payload(config.audio_path))
    print(f"{probability:.8f}")
    return 0
