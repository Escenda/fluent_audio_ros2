from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx


@dataclass(frozen=True)
class WorkerConfig:
    encoder: Path
    decoder: Path
    joiner: Path
    tokens: Path
    keywords: Path
    provider: str
    sample_rate: int
    num_threads: int
    max_active_paths: int
    num_trailing_blanks: int
    keywords_score: float
    keywords_threshold: float


@dataclass(frozen=True)
class DetectConfig:
    worker: WorkerConfig
    audio_path: Path


def _existing_file(value: str, label: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{label} file does not exist: {path}")
    return path


def _positive_int(value: int, label: str) -> int:
    if value <= 0:
        raise RuntimeError(f"{label} must be greater than zero")
    return value


def _non_negative_int(value: int, label: str) -> int:
    if value < 0:
        raise RuntimeError(f"{label} must be zero or greater")
    return value


def _positive_float(value: float, label: str) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise RuntimeError(f"{label} must be finite and greater than zero")
    return float(value)


def _worker_config(args: argparse.Namespace) -> WorkerConfig:
    provider = str(args.provider).strip()
    if provider not in {"cpu", "cuda", "coreml"}:
        raise RuntimeError(f"unsupported provider: {provider}")
    return WorkerConfig(
        encoder=_existing_file(args.encoder, "encoder"),
        decoder=_existing_file(args.decoder, "decoder"),
        joiner=_existing_file(args.joiner, "joiner"),
        tokens=_existing_file(args.tokens, "tokens"),
        keywords=_existing_file(args.keywords, "keywords"),
        provider=provider,
        sample_rate=_positive_int(int(args.sample_rate), "sample-rate"),
        num_threads=_positive_int(int(args.num_threads), "num-threads"),
        max_active_paths=_positive_int(int(args.max_active_paths), "max-active-paths"),
        num_trailing_blanks=_non_negative_int(
            int(args.num_trailing_blanks),
            "num-trailing-blanks",
        ),
        keywords_score=_positive_float(float(args.keywords_score), "keywords-score"),
        keywords_threshold=_positive_float(
            float(args.keywords_threshold),
            "keywords-threshold",
        ),
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--encoder", required=True)
    parser.add_argument("--decoder", required=True)
    parser.add_argument("--joiner", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--keywords", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--sample-rate", required=True, type=int)
    parser.add_argument("--num-threads", required=True, type=int)
    parser.add_argument("--max-active-paths", required=True, type=int)
    parser.add_argument("--num-trailing-blanks", required=True, type=int)
    parser.add_argument("--keywords-score", required=True, type=float)
    parser.add_argument("--keywords-threshold", required=True, type=float)


def parse_args() -> WorkerConfig | DetectConfig:
    parser = argparse.ArgumentParser(description="sherpa-onnx KWS external worker")
    subparsers = parser.add_subparsers(dest="command", required=True)

    health_parser = subparsers.add_parser("health", help="validate model/runtime contract")
    _add_common_args(health_parser)

    detect_parser = subparsers.add_parser("detect", help="run keyword spotting")
    detect_parser.add_argument("--audio", required=True, help="raw mono float32le audio path")
    _add_common_args(detect_parser)

    stream_parser = subparsers.add_parser("stream", help="run persistent streaming keyword spotting")
    _add_common_args(stream_parser)

    args = parser.parse_args()
    worker = _worker_config(args)
    if args.command == "health":
        return worker
    if args.command == "stream":
        return worker
    if args.command == "detect":
        audio_path = _existing_file(args.audio, "audio")
        return DetectConfig(worker=worker, audio_path=audio_path)
    raise RuntimeError(f"unsupported command: {args.command}")


def read_audio(audio_path: Path) -> np.ndarray:
    data = audio_path.read_bytes()
    if not data:
        raise RuntimeError("audio payload is required")
    if len(data) % np.dtype("<f4").itemsize != 0:
        raise RuntimeError("audio payload must be raw float32le")
    samples = np.frombuffer(data, dtype="<f4")
    if samples.ndim != 1:
        raise RuntimeError("audio payload must be one-dimensional")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("audio payload contains non-finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise RuntimeError("audio payload samples must be normalized to [-1.0, 1.0]")
    return samples


def create_spotter(config: WorkerConfig):
    return sherpa_onnx.KeywordSpotter(
        tokens=str(config.tokens),
        encoder=str(config.encoder),
        decoder=str(config.decoder),
        joiner=str(config.joiner),
        num_threads=config.num_threads,
        max_active_paths=config.max_active_paths,
        num_trailing_blanks=config.num_trailing_blanks,
        keywords_file=str(config.keywords),
        keywords_score=config.keywords_score,
        keywords_threshold=config.keywords_threshold,
        provider=config.provider,
    )


def _result_keyword(result: object) -> str:
    if isinstance(result, str):
        return result.strip()
    return str(getattr(result, "keyword", "")).strip()


def _result_start_time(result: object) -> float:
    if isinstance(result, str):
        return 0.0
    return float(getattr(result, "start_time", 0.0))


def detect(config: DetectConfig) -> str:
    spotter = create_spotter(config.worker)
    stream = spotter.create_stream()
    samples = read_audio(config.audio_path)
    stream.accept_waveform(config.worker.sample_rate, samples)
    while spotter.is_ready(stream):
        spotter.decode_stream(stream)
    result = spotter.get_result(stream)
    keyword = _result_keyword(result)
    if not keyword:
        return "NO_DETECTION"
    score = 1.0
    start_time = _result_start_time(result)
    if not np.isfinite(start_time) or start_time < 0.0:
        raise RuntimeError("worker start_time must be finite and >= 0")
    return f"DETECTED\t{keyword}\t{score:.8f}\t{start_time:.8f}"


def _decode_inline_audio(value: str) -> np.ndarray:
    try:
        data = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise RuntimeError("stream audio must be base64 float32le") from exc
    if not data:
        raise RuntimeError("stream audio payload is required")
    if len(data) % np.dtype("<f4").itemsize != 0:
        raise RuntimeError("stream audio payload must be raw float32le")
    samples = np.frombuffer(data, dtype="<f4")
    if samples.ndim != 1:
        raise RuntimeError("stream audio payload must be one-dimensional")
    if not np.all(np.isfinite(samples)):
        raise RuntimeError("stream audio payload contains non-finite samples")
    if np.any(samples < -1.0) or np.any(samples > 1.0):
        raise RuntimeError("stream audio payload samples must be normalized to [-1.0, 1.0]")
    return samples


def _format_stream_result(result: object) -> str:
    keyword = _result_keyword(result)
    if not keyword:
        return "NO_DETECTION"
    start_time = _result_start_time(result)
    if not np.isfinite(start_time) or start_time < 0.0:
        raise RuntimeError("worker start_time must be finite and >= 0")
    return f"DETECTED\t{keyword}\t1.00000000\t{start_time:.8f}"


def stream_loop(config: WorkerConfig) -> int:
    spotter = create_spotter(config)
    stream = spotter.create_stream()
    print("STREAM_READY", flush=True)
    while True:
        line = input()
        if not line:
            continue
        if line == "EXIT":
            return 0
        if line == "RESET":
            spotter.reset_stream(stream)
            print("OK", flush=True)
            continue
        if not line.startswith("PUSH "):
            raise RuntimeError(f"unsupported stream command: {line.split(' ', 1)[0]}")
        parts = line.split(" ", 2)
        if len(parts) != 3:
            raise RuntimeError("PUSH command must be PUSH <sample_rate> <base64_audio>")
        sample_rate = int(parts[1])
        if sample_rate != config.sample_rate:
            raise RuntimeError(
                f"stream sample_rate must be {config.sample_rate}, got {sample_rate}"
            )
        samples = _decode_inline_audio(parts[2])
        stream.accept_waveform(sample_rate, samples)
        while spotter.is_ready(stream):
            spotter.decode_stream(stream)
        result = spotter.get_result(stream)
        output = _format_stream_result(result)
        print(output, flush=True)
        if output.startswith("DETECTED\t"):
            spotter.reset_stream(stream)


def main() -> int:
    config = parse_args()
    if isinstance(config, WorkerConfig):
        import sys
        if len(sys.argv) > 1 and sys.argv[1] == "stream":
            return stream_loop(config)
        create_spotter(config)
        print("health-ok")
        return 0
    print(detect(config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
