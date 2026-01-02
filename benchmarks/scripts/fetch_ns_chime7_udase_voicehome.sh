#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"
DATA_DIR="${BENCH_DIR}/data/voicehome_1314196"

ZIP_KEY="voiceHome_corpus v0.2.zip"
PDF_KEY="voiceHome_corpus_french_documentation_v1.2.pdf"

ZIP_MD5="1a98cace6a3ceda9736e107185baf3a6"
PDF_MD5="da952914ea614cfe3f4ee08abc434459"

ZIP_PATH="${DATA_DIR}/voiceHome_corpus_v0.2.zip"
PDF_PATH="${DATA_DIR}/${PDF_KEY}"

EXTRACT="${1:-}"

mkdir -p "${DATA_DIR}"

urlencode() {
  python3 - <<'PY' "$1"
import sys
import urllib.parse

print(urllib.parse.quote(sys.argv[1]))
PY
}

download() {
  local key="$1"
  local out="$2"
  local encoded_key=""
  encoded_key="$(urlencode "${key}")"
  echo "Downloading: ${key}"
  echo "To: ${out}"
  curl -L -C - -o "${out}" "https://zenodo.org/api/records/1314196/files/${encoded_key}/content"
}

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

download "${ZIP_KEY}" "${ZIP_PATH}"
download "${PDF_KEY}" "${PDF_PATH}"

check_md5 "${ZIP_PATH}" "${ZIP_MD5}"
check_md5 "${PDF_PATH}" "${PDF_MD5}"

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
