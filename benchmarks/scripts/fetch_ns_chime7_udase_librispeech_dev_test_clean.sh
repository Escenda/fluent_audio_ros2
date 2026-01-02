#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"
ARCHIVE_DIR="${BENCH_DIR}/data/openslr_12"
LIBRISPEECH_DIR="${BENCH_DIR}/data/librispeech"

DEV_NAME="dev-clean.tar.gz"
TEST_NAME="test-clean.tar.gz"

DEV_MD5="42e2234ba48799c1f50f24a7926300a1"
TEST_MD5="32fa31d27d2e1cad72775fee3f4849a9"

DEV_URL="https://openslr.trmal.net/resources/12/${DEV_NAME}"
TEST_URL="https://openslr.trmal.net/resources/12/${TEST_NAME}"

DEV_PATH="${ARCHIVE_DIR}/${DEV_NAME}"
TEST_PATH="${ARCHIVE_DIR}/${TEST_NAME}"
MD5SUM_PATH="${ARCHIVE_DIR}/md5sum.txt"

EXTRACT="${1:-}"

mkdir -p "${ARCHIVE_DIR}" "${LIBRISPEECH_DIR}"

echo "Downloading md5sum.txt"
curl -L -o "${MD5SUM_PATH}" "https://openslr.trmal.net/resources/12/md5sum.txt"

echo "Downloading: ${DEV_URL}"
curl -L -C - -o "${DEV_PATH}" "${DEV_URL}"

echo "Downloading: ${TEST_URL}"
curl -L -C - -o "${TEST_PATH}" "${TEST_URL}"

check_md5() {
  local path="$1"
  local expected="$2"
  local actual=""
  if command -v md5sum >/dev/null 2>&1; then
    actual="$(md5sum "${path}" | awk '{print $1}')"
  elif command -v md5 >/dev/null 2>&1; then
    actual="$(md5 -q "${path}")"
  else
    echo "WARN: md5 tool not found; skipping checksum verification"
    return 0
  fi

  if [[ "${actual}" != "${expected}" ]]; then
    echo "ERROR: MD5 mismatch: ${path}"
    echo "  expected: ${expected}"
    echo "  actual:   ${actual}"
    exit 1
  fi
  echo "MD5 OK: ${actual} (${path})"
}

check_md5 "${DEV_PATH}" "${DEV_MD5}"
check_md5 "${TEST_PATH}" "${TEST_MD5}"

if [[ "${EXTRACT}" == "--extract" ]]; then
  echo "Extracting to: ${LIBRISPEECH_DIR}"
  # The official archives contain a top-level "LibriSpeech/" directory.
  # We strip it so that ${LIBRISPEECH_DIR} directly contains dev-clean/ and test-clean/.
  tar -xzf "${DEV_PATH}" -C "${LIBRISPEECH_DIR}" --strip-components=1
  tar -xzf "${TEST_PATH}" -C "${LIBRISPEECH_DIR}" --strip-components=1
fi
