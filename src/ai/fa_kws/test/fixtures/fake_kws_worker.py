#!/usr/bin/env python3
from __future__ import annotations

import base64
import math
import os
import struct
import sys
from pathlib import Path


def _value(flag: str) -> str:
    for index, arg in enumerate(sys.argv):
        if arg == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    raise RuntimeError(f"missing flag: {flag}")


def _require_file(flag: str) -> Path:
    path = Path(_value(flag))
    if not path.is_file():
        raise RuntimeError(f"{flag} is not a file: {path}")
    return path


def _validate_audio(audio_bytes: bytes) -> None:
    if not audio_bytes:
        raise RuntimeError("audio payload is required")
    if len(audio_bytes) % 4 != 0:
        raise RuntimeError("audio payload must be raw float32le")
    samples = [value[0] for value in struct.iter_unpack("<f", audio_bytes)]
    if not all(math.isfinite(sample) for sample in samples):
        raise RuntimeError("audio payload contains non-finite samples")
    if any(sample < -1.0 or sample > 1.0 for sample in samples):
        raise RuntimeError("audio payload samples must be normalized to [-1.0, 1.0]")


def main() -> int:
    if len(sys.argv) < 2:
        raise RuntimeError("mode is required")
    mode = sys.argv[1]
    for flag in ("--encoder", "--decoder", "--joiner", "--tokens", "--keywords"):
        _require_file(flag)
    if not _value("--provider"):
        raise RuntimeError("provider is required")

    if mode == "health":
        print("health-ok")
        return 0

    if int(_value("--sample-rate")) <= 0:
        raise RuntimeError("sample rate must be positive")

    if mode == "stream":
        print("STREAM_READY", flush=True)
        calls = 0
        while True:
            line = sys.stdin.readline()
            if line == "":
                return 0
            line = line.strip()
            if line == "EXIT":
                return 0
            if line == "RESET":
                print("OK", flush=True)
                continue
            if not line.startswith("PUSH "):
                raise RuntimeError(f"unsupported stream command: {line}")
            parts = line.split(" ", 2)
            if len(parts) != 3:
                raise RuntimeError("bad PUSH command")
            audio_bytes = base64.b64decode(parts[2].encode("ascii"), validate=True)
            _validate_audio(audio_bytes)
            calls += 1
            fake_mode = os.environ.get("FA_KWS_FAKE_MODE", "detect")
            if fake_mode == "none":
                print("NO_DETECTION", flush=True)
            elif fake_mode == "bad":
                print("bad-output", flush=True)
            elif fake_mode == "detect_after_two" and calls < 2:
                print("NO_DETECTION", flush=True)
            else:
                print("DETECTED\thello_fluent\t0.875\t0.25", flush=True)
        return 0

    if mode != "detect":
        raise RuntimeError(f"unsupported mode: {mode}")

    audio_path = _require_file("--audio")
    audio_bytes = audio_path.read_bytes()
    _validate_audio(audio_bytes)

    fake_mode = os.environ.get("FA_KWS_FAKE_MODE", "detect")
    if fake_mode == "none":
        print("NO_DETECTION")
        return 0
    if fake_mode == "bad":
        print("bad-output")
        return 0
    print("DETECTED\thello_fluent\t0.875\t0.25")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
