#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"

CONFIG_PATH="${1:-${BENCH_DIR}/config/reverberant_librichime5_prepare.yaml}"

echo "Config: ${CONFIG_PATH}"
python3 "${BENCH_DIR}/tools/prepare_reverberant_librichime5.py" \
  --root-dir "${ROOT_DIR}" \
  --config "${CONFIG_PATH}"

