#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", default="")
    parser.add_argument("--model", required=True)
    parser.add_argument("--provider", required=True)
    parser.add_argument("--health-check", action="store_true")
    args = parser.parse_args()

    model_path = Path(args.model)
    model_text = model_path.read_text(encoding="utf-8", errors="ignore")
    first_line = model_text.splitlines()[0].strip()
    if args.health_check:
        if first_line == "healthfail":
            return 2
        print("ok")
        return 0

    audio_path = Path(args.audio)
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        return 3
    print(first_line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
