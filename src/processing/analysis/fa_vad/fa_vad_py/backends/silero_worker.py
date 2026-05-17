from __future__ import annotations

import argparse
import wave
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


def parse_args() -> WorkerConfig:
    parser = argparse.ArgumentParser(description="Silero VAD external worker")
    parser.add_argument("--audio", required=True, help="Mono PCM16 WAV file path")
    parser.add_argument("--model", required=True, help="Local torch.hub Silero repository")
    parser.add_argument("--provider", required=True, help="cpu, cuda, or cuda:<index>")
    parser.add_argument("--sample-rate", required=True, type=int, help="Expected sample rate")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser()
    if not audio_path.is_file():
        raise RuntimeError(f"audio file does not exist: {audio_path}")
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise RuntimeError(f"model directory does not exist: {model_path}")
    if args.sample_rate <= 0:
        raise RuntimeError("sample-rate must be > 0")

    return WorkerConfig(
        audio_path=audio_path,
        model_path=model_path,
        provider=validate_provider(str(args.provider)),
        sample_rate=int(args.sample_rate),
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
    with wave.open(str(config.audio_path), "rb") as wav_file:
        if wav_file.getnchannels() != 1:
            raise RuntimeError("audio file must be mono")
        if wav_file.getsampwidth() != 2:
            raise RuntimeError("audio file must be PCM16")
        if wav_file.getframerate() != config.sample_rate:
            raise RuntimeError(
                f"audio sample rate mismatch: {wav_file.getframerate()} != {config.sample_rate}"
            )
        pcm = wav_file.readframes(wav_file.getnframes())

    samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    required_samples = 512 if config.sample_rate == 16000 else 256
    if samples.size < required_samples:
        raise RuntimeError(
            f"audio window too short: need {required_samples} samples, got {samples.size}"
        )
    return samples[-required_samples:]


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
