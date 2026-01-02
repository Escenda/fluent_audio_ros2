#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"
DATA_DIR="${BENCH_DIR}/data/zenodo_10418311"

ZIP_NAME="CHiME-7-UDASE-evaluation-data.zip"
ZIP_PATH="${DATA_DIR}/${ZIP_NAME}"

# Zenodo record 10418311 (single ZIP file)
DOWNLOAD_URL="https://zenodo.org/api/records/10418311/files/${ZIP_NAME}/content"
EXPECTED_MD5="37b97f9f4ba95152725548fa2a1893ac"

EXTRACT="${1:-}"

mkdir -p "${DATA_DIR}"

echo "Downloading: ${DOWNLOAD_URL}"
echo "To: ${ZIP_PATH}"
echo "NOTE: This is a large file (~3.5GB). curl will resume if partially downloaded."

curl -L -C - -o "${ZIP_PATH}" "${DOWNLOAD_URL}"

actual_md5=""
if command -v md5sum >/dev/null 2>&1; then
  actual_md5="$(md5sum "${ZIP_PATH}" | awk '{print $1}')"
elif command -v md5 >/dev/null 2>&1; then
  actual_md5="$(md5 -q "${ZIP_PATH}")"
else
  echo "WARN: md5 tool not found; skipping checksum verification"
fi

if [[ -n "${actual_md5}" ]]; then
  if [[ "${actual_md5}" != "${EXPECTED_MD5}" ]]; then
    echo "ERROR: MD5 mismatch"
    echo "  expected: ${EXPECTED_MD5}"
    echo "  actual:   ${actual_md5}"
    exit 1
  fi
  echo "MD5 OK: ${actual_md5}"
fi

if [[ "${EXTRACT}" == "--extract" ]]; then
  if ! command -v unzip >/dev/null 2>&1; then
    echo "ERROR: unzip not found; cannot extract"
    exit 1
  fi
  OUT_DIR="${DATA_DIR}/extracted"
  mkdir -p "${OUT_DIR}"
  echo "Extracting to: ${OUT_DIR}"
  unzip -n "${ZIP_PATH}" -d "${OUT_DIR}"
fi

