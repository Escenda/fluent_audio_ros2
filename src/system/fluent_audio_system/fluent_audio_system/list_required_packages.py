from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys

from fluent_audio_system.config_schema import load_required_packages


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="List ROS2 packages required by a FluentAudio system config."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Comma-separated FluentAudio system YAML config path list.",
    )
    args = parser.parse_args(argv)

    try:
        packages = load_required_packages(args.config)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for package in packages:
        print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
