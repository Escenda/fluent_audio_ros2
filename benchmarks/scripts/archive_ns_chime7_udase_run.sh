#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 <run_id>"
  exit 2
fi

RUNS_DIR="${BENCH_DIR}/runs"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
ARCHIVE_DIR="${BENCH_DIR}/archives"
ARCHIVE_PATH="${ARCHIVE_DIR}/${RUN_ID}.tar.xz"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "ERROR: run dir not found: ${RUN_DIR}" >&2
  exit 2
fi

mkdir -p "${ARCHIVE_DIR}"

if [[ -e "${ARCHIVE_PATH}" ]]; then
  echo "ERROR: archive already exists: ${ARCHIVE_PATH}" >&2
  exit 2
fi

echo "Archiving: ${RUN_DIR}"
echo "To: ${ARCHIVE_PATH}"
tar -cJf "${ARCHIVE_PATH}" -C "${RUNS_DIR}" "${RUN_ID}"

if [[ ! -s "${ARCHIVE_PATH}" ]]; then
  echo "ERROR: archive creation failed (empty file): ${ARCHIVE_PATH}" >&2
  exit 1
fi

SHA_PATH="${ARCHIVE_PATH}.sha256"
if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "${ARCHIVE_PATH}" > "${SHA_PATH}"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "${ARCHIVE_PATH}" > "${SHA_PATH}"
elif command -v openssl >/dev/null 2>&1; then
  openssl dgst -sha256 "${ARCHIVE_PATH}" > "${SHA_PATH}"
else
  echo "WARN: no sha256 tool found; skipping checksum output" >&2
fi

echo "Removing run dir: ${RUN_DIR}"
rm -rf "${RUN_DIR}"

echo "Archived: ${ARCHIVE_PATH}"

