#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/aec_challenge_2023"

DATA_DIR="${BENCH_DIR}/data"
RUN_ID="${1:-}"

if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 <run_id>"
  echo "example: $0 20260101_120000_aec_linear_v1"
  exit 2
fi

RUN_DIR="${BENCH_DIR}/runs/${RUN_ID}"
META_DIR="${RUN_DIR}/meta"
PARAMS_DIR="${META_DIR}/params"
ARTIFACTS_DIR="${RUN_DIR}/artifacts"
METRICS_DIR="${RUN_DIR}/metrics"

mkdir -p "${PARAMS_DIR}" "${ARTIFACTS_DIR}" "${METRICS_DIR}"

git -C "${ROOT_DIR}" rev-parse HEAD > "${META_DIR}/git_rev.txt" 2>/dev/null || true
date -u +"%Y-%m-%dT%H:%M:%SZ" > "${META_DIR}/started_at_utc.txt"
printf "%q " "$0" "$@" > "${META_DIR}/command.txt"
echo "" >> "${META_DIR}/command.txt"

# Copy YAML params (as of run)
cp -f "${ROOT_DIR}/src/dsp/"*/config/*.yaml "${PARAMS_DIR}/" 2>/dev/null || true

# Dataset manifest placeholder (fill in with actual dataset metadata / checksums as needed)
cat > "${META_DIR}/dataset_manifest.json" <<EOF
{
  "benchmark": "aec_challenge_2023",
  "data_dir": "${DATA_DIR}",
  "notes": "Place the AEC Challenge dataset under data_dir. If redistribution is restricted, keep it local and record checksums and acquisition details here."
}
EOF

echo "Run directory: ${RUN_DIR}"
echo "NOTE: This script is a scaffold. Plug in the official dataset layout + evaluation script here."

