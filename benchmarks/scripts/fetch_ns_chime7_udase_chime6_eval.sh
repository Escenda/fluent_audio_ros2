#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"
ARCHIVE_DIR="${BENCH_DIR}/data/openslr_150"

NAME="CHiME6_eval.tar.gz"
DEFAULT_URL="https://openslr.trmal.net/resources/150/${NAME}"

URL="${CHIME6_EVAL_URL:-${DEFAULT_URL}}"
TAR_PATH="${ARCHIVE_DIR}/${NAME}"
OUT_DIR="${ARCHIVE_DIR}/extracted"

EXTRACT=0
PRINT_ROOT=0

usage() {
  cat <<EOF
usage: $0 [--extract] [--print-root] [--url <url>]

examples:
  $0 --extract
  $0 --extract --print-root
  $0 --url https://openslr.elda.org/resources/150/CHiME6_eval.tar.gz --extract

notes:
  - CHiME-6 is provided by OpenSLR SLR150 (CC BY-SA 4.0): https://openslr.org/150/
  - This is a large file (~12GB). curl will resume if partially downloaded.
EOF
}

log() {
  echo "$@" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extract)
      EXTRACT=1
      shift 1
      ;;
    --print-root)
      PRINT_ROOT=1
      EXTRACT=1
      shift 1
      ;;
    --url)
      URL="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      log "ERROR: unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

mkdir -p "${ARCHIVE_DIR}"

log "Downloading: ${URL}"
log "To: ${TAR_PATH}"
curl -L -C - -o "${TAR_PATH}" "${URL}"

if [[ "${EXTRACT}" -eq 1 ]]; then
  mkdir -p "${OUT_DIR}"
  log "Extracting to: ${OUT_DIR}"
  tar -xzf "${TAR_PATH}" -C "${OUT_DIR}"
fi

if [[ "${PRINT_ROOT}" -eq 1 ]]; then
  # Find the dataset root that contains an 'audio' directory with wav files.
  roots=()
  while IFS= read -r audio_dir; do
    root="$(dirname "${audio_dir}")"
    if find "${audio_dir}" -maxdepth 3 -type f -name "*.wav" -print -quit 2>/dev/null | grep -q .; then
      roots+=("${root}")
    fi
  done < <(find "${OUT_DIR}" -maxdepth 4 -type d -name audio 2>/dev/null || true)

  if [[ "${#roots[@]}" -eq 0 ]]; then
    log "ERROR: Could not locate CHiME-6 dataset root under: ${OUT_DIR}"
    log "Expected a directory containing: audio/*/*.wav (e.g., audio/eval/S01_P01.wav)"
    exit 1
  fi

  if [[ "${#roots[@]}" -gt 1 ]]; then
    log "WARN: Multiple dataset roots found; using the first one:"
    for r in "${roots[@]}"; do
      log "  - ${r}"
    done
  fi

  (cd "${roots[0]}" && pwd)
fi
