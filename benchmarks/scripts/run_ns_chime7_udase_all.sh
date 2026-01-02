#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"
DATA_DIR="${BENCH_DIR}/data"

CHIME_ORIG_DIR="${CHIME_ORIG_DIR:-${CHIME5_ORIG_DIR:-}}"
USE_CHIME6_EVAL=0
SKIP_REVERB=0

usage() {
  cat <<EOF
usage: $0 <run_id> [--chime5-data-dir <path> | --chime6-eval | --chime6-eval-data-dir <path>] [--skip-reverberant]

examples:
  $0 20260102_120000_dtln_all --chime5-data-dir /path/to/CHiME5
  CHIME5_ORIG_DIR=/path/to/CHiME5 $0 20260102_120000_dtln_all
  $0 20260102_120000_dtln_all --chime6-eval

notes:
  - CHiME-5 is NOT auto-downloadable (license); you must obtain it manually.
  - CHiME-6 eval is auto-downloadable from OpenSLR (SLR150, CC BY-SA 4.0): https://openslr.org/150/
  - This script downloads large datasets (Zenodo/LibriSpeech/VoiceHome) and may take a long time.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

RUN_ID="${1:-}"
shift || true

if [[ -z "${RUN_ID}" ]]; then
  usage
  exit 2
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --chime5-data-dir)
      CHIME_ORIG_DIR="${2:-}"
      shift 2
      ;;
    --chime6-eval)
      USE_CHIME6_EVAL=1
      shift 1
      ;;
    --chime6-eval-data-dir)
      CHIME_ORIG_DIR="${2:-}"
      USE_CHIME6_EVAL=1
      shift 2
      ;;
    --skip-reverberant)
      SKIP_REVERB=1
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1"
      usage
      exit 2
      ;;
  esac
done

ZENODO_LISTENING_C0_DIR="${DATA_DIR}/zenodo_10418311/extracted/listening_test/data/C0"
if [[ ! -d "${ZENODO_LISTENING_C0_DIR}" ]]; then
  echo "[1/4] Fetching CHiME-7 UDASE evaluation data (Zenodo)..."
  "${BENCH_DIR}/../scripts/fetch_ns_chime7_udase_eval_data.sh" --extract
else
  echo "[1/4] Zenodo eval data already extracted."
fi

LIBRI_DIR="${DATA_DIR}/librispeech"
if [[ ! -d "${LIBRI_DIR}/dev-clean" || ! -d "${LIBRI_DIR}/test-clean" ]]; then
  echo "[2/4] Fetching LibriSpeech dev-clean/test-clean..."
  "${BENCH_DIR}/../scripts/fetch_ns_chime7_udase_librispeech_dev_test_clean.sh" --extract
else
  echo "[2/4] LibriSpeech already extracted."
fi

VOICEHOME_DIR="${DATA_DIR}/voicehome_1314196/extracted"
if [[ ! -d "${VOICEHOME_DIR}" ]]; then
  echo "[3/4] Fetching VoiceHome (Zenodo 1314196)..."
  "${BENCH_DIR}/../scripts/fetch_ns_chime7_udase_voicehome.sh" --extract
else
  echo "[3/4] VoiceHome already extracted."
fi

RUN_DIR="${BENCH_DIR}/runs/${RUN_ID}"
META_DIR="${RUN_DIR}/meta"
PARAMS_DIR="${META_DIR}/params"
mkdir -p "${PARAMS_DIR}"

if [[ "${SKIP_REVERB}" -eq 0 ]]; then
  if [[ "${USE_CHIME6_EVAL}" -eq 1 && -z "${CHIME_ORIG_DIR}" ]]; then
    echo "[4/4] Fetching CHiME-6 eval (OpenSLR SLR150)..."
    CHIME_ORIG_DIR="$("${BENCH_DIR}/../scripts/fetch_ns_chime7_udase_chime6_eval.sh" --extract --print-root)"
  fi

  if [[ -z "${CHIME_ORIG_DIR}" ]]; then
    echo "ERROR: CHiME original data dir is required for reverberant-LibriCHiME-5 generation."
    echo "Set via one of:"
    echo "  - --chime5-data-dir /path/to/CHiME5 (manual)"
    echo "  - --chime6-eval (auto-download)"
    echo "  - CHIME_ORIG_DIR=/path/to/CHiME* (env)"
    exit 2
  fi
  if [[ ! -d "${CHIME_ORIG_DIR}" ]]; then
    echo "ERROR: CHiME dir not found: ${CHIME_ORIG_DIR}"
    exit 2
  fi

  EFFECTIVE_PREP_CFG="${PARAMS_DIR}/reverberant_librichime5_prepare_effective.yaml"
  python3 - <<PY
import pathlib
import yaml

root_dir = pathlib.Path("${ROOT_DIR}").resolve()
template = root_dir / "benchmarks/ns_chime7_udase/config/reverberant_librichime5_prepare.yaml"
out = pathlib.Path("${EFFECTIVE_PREP_CFG}").resolve()

cfg = yaml.safe_load(template.read_text(encoding="utf-8"))
cfg["chime5"]["original_data_dir"] = "${CHIME_ORIG_DIR}"
out.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
print(f"Wrote: {out}")
PY

  REVERB_EVAL_DIR="${DATA_DIR}/reverberant_librichime5_audio/eval"
  if [[ ! -d "${REVERB_EVAL_DIR}" ]]; then
    echo "[4/4] Preparing reverberant-LibriCHiME-5 (eval)..."
    "${BENCH_DIR}/../scripts/prepare_ns_chime7_udase_reverberant_librichime5.sh" "${EFFECTIVE_PREP_CFG}"
  else
    echo "[4/4] reverberant-LibriCHiME-5 already exists: ${REVERB_EVAL_DIR}"
  fi
else
  echo "[4/4] Skip: reverberant-LibriCHiME-5 generation."
fi

echo "Running benchmark run: ${RUN_ID}"
"${BENCH_DIR}/../scripts/run_ns_chime7_udase.sh" "${RUN_ID}"
