from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class WorkerConfig:
    audio_path: Path
    model_path: Path
    provider: str
    sample_rate: int
    window_samples: int


WINDOW_SAMPLES_BY_SAMPLE_RATE = {
    8000: 256,
    16000: 512,
}


def parse_args() -> WorkerConfig:
    parser = argparse.ArgumentParser(description="Silero VAD external worker")
    parser.add_argument("--audio", required=True, help="Mono raw float32le audio path")
    parser.add_argument("--model", required=True, help="Local torch.hub Silero repository")
    parser.add_argument("--provider", required=True, help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--sample-rate", required=True, type=int, help="Expected sample rate")
    parser.add_argument(
        "--window-samples",
        required=True,
        type=int,
        help="Expected model window",
    )
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser()
    if not audio_path.is_file():
        raise RuntimeError(f"audio file does not exist: {audio_path}")
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_path}")
    if args.sample_rate not in (8000, 16000):
        raise RuntimeError("sample-rate must be 8000 or 16000")
    expected_window_samples = WINDOW_SAMPLES_BY_SAMPLE_RATE[int(args.sample_rate)]
    if args.window_samples != expected_window_samples:
        raise RuntimeError(
            "window-samples must match Silero sample-rate contract: "
            f"{expected_window_samples} for {args.sample_rate} Hz"
        )

    return WorkerConfig(
        audio_path=audio_path,
        model_path=model_path,
        provider=validate_provider(str(args.provider)),
        sample_rate=int(args.sample_rate),
        window_samples=int(args.window_samples),
    )


def validate_provider(provider: str) -> str:
    if provider == "cpu":
        return provider
    if provider == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("provider=cuda requested but CUDA is unavailable")
        return provider
    if provider.startswith("cuda:") and provider.removeprefix("cuda:").isdigit():
        if not torch.cuda.is_available():
            raise RuntimeError(f"provider={provider} requested but CUDA is unavailable")
        device_index = int(provider.removeprefix("cuda:"))
        if device_index >= torch.cuda.device_count():
            raise RuntimeError(f"provider={provider} requested unavailable CUDA device")
        return provider
    raise RuntimeError(f"unsupported provider: {provider}")


def read_audio_window(config: WorkerConfig) -> np.ndarray:
    audio_bytes = config.audio_path.read_bytes()
    if not audio_bytes:
        raise RuntimeError("audio file is empty")
    if len(audio_bytes) % np.dtype("<f4").itemsize != 0:
        raise RuntimeError("audio file must be raw float32le")
    samples = np.frombuffer(audio_bytes, dtype="<f4")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("audio file contains non-finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise RuntimeError("audio samples must be normalized to [-1.0, 1.0]")
    if samples.size < config.window_samples:
        raise RuntimeError(
            "audio window too short: "
            f"need {config.window_samples} samples, got {samples.size}"
        )
    return samples[-config.window_samples:]


def main() -> int:
    config = parse_args()
    samples = read_audio_window(config)

    device = torch.device(config.provider)
    model, _ = torch.hub.load(
        str(config.model_path),
        "silero_vad",
        source="local",
        trust_repo=True,
    )
    model.eval()
    model.to(device)

    tensor = torch.from_numpy(samples).to(device)
    with torch.no_grad():
        probability = float(model(tensor, config.sample_rate).item())
    if probability < 0.0 or probability > 1.0:
        raise RuntimeError(f"model probability out of range: {probability}")
    print(f"{probability:.8f}")
    return 0
