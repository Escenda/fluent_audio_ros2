#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BENCH_DIR="${ROOT_DIR}/benchmarks/ns_chime7_udase"

RUN_ID="${1:-}"
if [[ -z "${RUN_ID}" ]]; then
  echo "usage: $0 <run_id>"
  echo "example: $0 20260101_120000_dtln_onnx_v1"
  exit 2
fi

DATA_DIR="${BENCH_DIR}/data"
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

{
  echo "uname: $(uname -a 2>/dev/null || true)"
  echo "date_utc: $(date -u +\"%Y-%m-%dT%H:%M:%SZ\")"
  echo "python: $(python3 --version 2>/dev/null || true)"
  python3 - <<'PY' 2>/dev/null || true
import importlib
import platform
import sys

print(f"python_executable: {sys.executable}")
print(f"platform: {platform.platform()}")
for m in ["numpy", "onnxruntime", "soundfile", "yaml", "scipy", "pesq", "pystoi", "tqdm"]:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, "__version__", "unknown")
        print(f"{m}_version: {ver}")
    except Exception as e:
        print(f"{m}_version: <missing> ({e})")
PY
} > "${META_DIR}/env.txt"

# Copy all YAML params (as of run)
cp -f "${ROOT_DIR}/src/dsp/"*/config/*.yaml "${PARAMS_DIR}/" 2>/dev/null || true
cp -f "${BENCH_DIR}/config/"*.yaml "${PARAMS_DIR}/" 2>/dev/null || true

# Dataset manifest (fill paths/checksums as the local datasets are prepared)
cat > "${META_DIR}/dataset_manifest.json" <<EOF
{
  "benchmark": "ns_chime7_udase_2402_01413",
  "paper": "https://arxiv.org/abs/2402.01413",
  "zenodo": {
    "record": "https://doi.org/10.5281/zenodo.10418311",
    "zip_name": "CHiME-7-UDASE-evaluation-data.zip",
    "expected_md5": "37b97f9f4ba95152725548fa2a1893ac",
    "local_path": "${DATA_DIR}/zenodo_10418311/CHiME-7-UDASE-evaluation-data.zip",
    "local_extracted_dir": "${DATA_DIR}/zenodo_10418311/extracted"
  },
  "listening_test": {
    "input_dir_C0": "${DATA_DIR}/zenodo_10418311/extracted/listening_test/data/C0"
  },
  "tools": {
    "segmented_chime5_scripts": "https://github.com/UDASE-CHiME2023/CHiME-5",
    "reverberant_librichime5_scripts": "https://github.com/UDASE-CHiME2023/reverberant-LibriCHiME-5"
  },
  "reverberant_librichime5_preparation": {
    "config": "${BENCH_DIR}/config/reverberant_librichime5_prepare.yaml",
    "scripts": [
      "benchmarks/scripts/fetch_ns_chime7_udase_librispeech_dev_test_clean.sh",
      "benchmarks/scripts/fetch_ns_chime7_udase_voicehome.sh",
      "benchmarks/scripts/fetch_ns_chime7_udase_chime6_eval.sh",
      "benchmarks/scripts/prepare_ns_chime7_udase_reverberant_librichime5.sh"
    ],
    "upstream_repos": [
      "https://github.com/UDASE-CHiME2023/CHiME-5",
      "https://github.com/UDASE-CHiME2023/reverberant-LibriCHiME-5"
    ],
    "librispeech": {
      "openslr_page": "https://www.openslr.org/12/",
      "md5sum_txt": "https://openslr.trmal.net/resources/12/md5sum.txt",
      "archives": [
        {"name": "dev-clean.tar.gz", "expected_md5": "42e2234ba48799c1f50f24a7926300a1"},
        {"name": "test-clean.tar.gz", "expected_md5": "32fa31d27d2e1cad72775fee3f4849a9"}
      ],
      "local_dir": "${DATA_DIR}/librispeech",
      "local_archive_dir": "${DATA_DIR}/openslr_12"
    },
    "voicehome": {
      "record": "https://doi.org/10.5281/zenodo.1314196",
      "zip_key": "voiceHome_corpus v0.2.zip",
      "expected_md5": "1a98cace6a3ceda9736e107185baf3a6",
      "local_zip_path": "${DATA_DIR}/voicehome_1314196/voiceHome_corpus_v0.2.zip",
      "local_extracted_dir": "${DATA_DIR}/voicehome_1314196/extracted"
    },
    "chime5": {
      "dataset_page": "https://www.chimechallenge.org/datasets/chime5",
      "notes": "Original CHiME-5 data must be obtained separately (license/restrictions). Alternatively, you can use CHiME-6 eval from OpenSLR SLR150."
    },
    "chime6": {
      "openslr_page": "https://openslr.org/150/",
      "license": "CC BY-SA 4.0",
      "archives": [
        {
          "name": "CHiME6_eval.tar.gz",
          "approx_size": "12G",
          "default_url": "https://openslr.trmal.net/resources/150/CHiME6_eval.tar.gz"
        }
      ],
      "local_archive_dir": "${DATA_DIR}/openslr_150",
      "local_extract_dir": "${DATA_DIR}/openslr_150/extracted"
    }
  },
  "local_datasets": {
    "segmented_chime5": "${DATA_DIR}/segmented_chime5",
    "reverberant_librichime5_audio": "${DATA_DIR}/reverberant_librichime5_audio"
  },
  "notes": "Store all inputs/outputs/metrics under runs/<run_id> per docs/fa_benchmark_policy.md. If datasets cannot be redistributed, keep them locally and record checksums + acquisition details here."
}
EOF

echo "Run directory: ${RUN_DIR}"

ZENODO_EXTRACTED_DIR="${DATA_DIR}/zenodo_10418311/extracted"
LISTENING_C0_DIR="${ZENODO_EXTRACTED_DIR}/listening_test/data/C0"

if [[ ! -d "${LISTENING_C0_DIR}" ]]; then
  echo "ERROR: listening_test data not found: ${LISTENING_C0_DIR}"
  echo "Run: benchmarks/scripts/fetch_ns_chime7_udase_eval_data.sh --extract"
  exit 1
fi

DTLN_CONFIG="${BENCH_DIR}/config/dtln_onnx_listening_test.yaml"
DTLN_OUT_DIR="${ARTIFACTS_DIR}/listening_test/dtln_onnx/C0"
DTLN_METRICS_CSV="${METRICS_DIR}/listening_test_dtln_onnx_C0.csv"
DTLN_SUMMARY_JSON="${METRICS_DIR}/summary_dtln_onnx_C0.json"
DTLN_INPUT_MANIFEST_JSON="${META_DIR}/input_manifest_listening_test_C0.json"

echo "Running DTLN(ONNX) on listening_test C0..."
python3 "${BENCH_DIR}/tools/run_listening_test_dtln_onnx.py" \
  --root-dir "${ROOT_DIR}" \
  --config "${DTLN_CONFIG}" \
  --input-dir "${LISTENING_C0_DIR}" \
  --output-dir "${DTLN_OUT_DIR}" \
  --metrics-csv "${DTLN_METRICS_CSV}" \
  --summary-json "${DTLN_SUMMARY_JSON}" \
  --manifest-json "${DTLN_INPUT_MANIFEST_JSON}" \
  --overwrite

DNSMOS_CONFIG="${BENCH_DIR}/config/dnsmos_sig_bak_ovr.yaml"
DNSMOS_IN_METRICS_CSV="${METRICS_DIR}/dnsmos_sig_bak_ovr_input_C0.csv"
DNSMOS_IN_SUMMARY_JSON="${METRICS_DIR}/dnsmos_sig_bak_ovr_input_C0_summary.json"
DNSMOS_IN_MANIFEST_JSON="${META_DIR}/dnsmos_sig_bak_ovr_input_C0_manifest.json"

DNSMOS_OUT_METRICS_CSV="${METRICS_DIR}/dnsmos_sig_bak_ovr_dtln_onnx_C0.csv"
DNSMOS_OUT_SUMMARY_JSON="${METRICS_DIR}/dnsmos_sig_bak_ovr_dtln_onnx_C0_summary.json"
DNSMOS_OUT_MANIFEST_JSON="${META_DIR}/dnsmos_sig_bak_ovr_dtln_onnx_C0_manifest.json"

echo "Scoring DNSMOS (SIG/BAK/OVR) for input C0..."
python3 "${BENCH_DIR}/tools/score_dnsmos_sig_bak_ovr.py" \
  --root-dir "${ROOT_DIR}" \
  --config "${DNSMOS_CONFIG}" \
  --input-dir "${LISTENING_C0_DIR}" \
  --metrics-csv "${DNSMOS_IN_METRICS_CSV}" \
  --summary-json "${DNSMOS_IN_SUMMARY_JSON}" \
  --manifest-json "${DNSMOS_IN_MANIFEST_JSON}" \
  --overwrite

echo "Scoring DNSMOS (SIG/BAK/OVR) for DTLN output..."
python3 "${BENCH_DIR}/tools/score_dnsmos_sig_bak_ovr.py" \
  --root-dir "${ROOT_DIR}" \
  --config "${DNSMOS_CONFIG}" \
  --input-dir "${DTLN_OUT_DIR}" \
  --metrics-csv "${DNSMOS_OUT_METRICS_CSV}" \
  --summary-json "${DNSMOS_OUT_SUMMARY_JSON}" \
  --manifest-json "${DNSMOS_OUT_MANIFEST_JSON}" \
  --overwrite

python3 - <<PY
import json
from pathlib import Path

in_summary = json.loads(Path("${DNSMOS_IN_SUMMARY_JSON}").read_text(encoding="utf-8"))
out_summary = json.loads(Path("${DNSMOS_OUT_SUMMARY_JSON}").read_text(encoding="utf-8"))

delta = {
  "input_summary": "${DNSMOS_IN_SUMMARY_JSON}",
  "output_summary": "${DNSMOS_OUT_SUMMARY_JSON}",
  "delta_avg_SIG_MOS": out_summary.get("avg_SIG_MOS", 0.0) - in_summary.get("avg_SIG_MOS", 0.0),
  "delta_avg_BAK_MOS": out_summary.get("avg_BAK_MOS", 0.0) - in_summary.get("avg_BAK_MOS", 0.0),
  "delta_avg_OVR_MOS": out_summary.get("avg_OVR_MOS", 0.0) - in_summary.get("avg_OVR_MOS", 0.0),
}

Path("${METRICS_DIR}/dnsmos_sig_bak_ovr_delta_input_C0_vs_dtln_onnx_C0.json").write_text(
  json.dumps(delta, ensure_ascii=False, indent=2),
  encoding="utf-8",
)
PY

REVERB_ROOT_DIR="${DATA_DIR}/reverberant_librichime5_audio"
REVERB_EVAL_ROOT_DIR="${REVERB_ROOT_DIR}/eval"

if [[ -d "${REVERB_EVAL_ROOT_DIR}" ]]; then
  echo "Running reverberant-LibriCHiME-5 (eval) processing + intrusive metrics..."

  DTLN_REVERB_CONFIG="${BENCH_DIR}/config/dtln_onnx_reverberant_librichime5_eval.yaml"
  INTRUSIVE_CONFIG="${BENCH_DIR}/config/reverberant_librichime5_intrusive_metrics.yaml"

  for group in 1 2 3; do
    GROUP_IN_DIR="${REVERB_EVAL_ROOT_DIR}/${group}"
    if [[ ! -d "${GROUP_IN_DIR}" ]]; then
      echo "Skip: ${GROUP_IN_DIR} not found"
      continue
    fi

    GROUP_OUT_DIR="${ARTIFACTS_DIR}/reverberant_librichime5/dtln_onnx/eval/${group}"
    GROUP_DTLN_METRICS_CSV="${METRICS_DIR}/reverberant_librichime5_dtln_onnx_eval_${group}.csv"
    GROUP_DTLN_SUMMARY_JSON="${METRICS_DIR}/reverberant_librichime5_dtln_onnx_eval_${group}_summary.json"
    GROUP_DTLN_MANIFEST_JSON="${META_DIR}/reverberant_librichime5_dtln_onnx_eval_${group}_input_manifest.json"

    echo "Running DTLN(ONNX) on reverberant-LibriCHiME-5 eval/${group}..."
    python3 "${BENCH_DIR}/tools/run_listening_test_dtln_onnx.py" \
      --root-dir "${ROOT_DIR}" \
      --config "${DTLN_REVERB_CONFIG}" \
      --input-dir "${GROUP_IN_DIR}" \
      --output-dir "${GROUP_OUT_DIR}" \
      --metrics-csv "${GROUP_DTLN_METRICS_CSV}" \
      --summary-json "${GROUP_DTLN_SUMMARY_JSON}" \
      --manifest-json "${GROUP_DTLN_MANIFEST_JSON}" \
      --overwrite

    GROUP_INTRUSIVE_CSV="${METRICS_DIR}/reverberant_librichime5_intrusive_eval_${group}.csv"
    GROUP_INTRUSIVE_SUMMARY_JSON="${METRICS_DIR}/reverberant_librichime5_intrusive_eval_${group}_summary.json"
    GROUP_INTRUSIVE_MANIFEST_JSON="${META_DIR}/reverberant_librichime5_intrusive_eval_${group}_manifest.json"

    echo "Evaluating intrusive metrics (PESQ/STOI/SI-SDR) for eval/${group}..."
    python3 "${BENCH_DIR}/tools/eval_reverberant_librichime5_intrusive.py" \
      --config "${INTRUSIVE_CONFIG}" \
      --dataset-root "${REVERB_ROOT_DIR}" \
      --subset "eval" \
      --group "${group}" \
      --pred-dir "${GROUP_OUT_DIR}" \
      --pred-method "dtln_onnx" \
      --out-csv "${GROUP_INTRUSIVE_CSV}" \
      --summary-json "${GROUP_INTRUSIVE_SUMMARY_JSON}" \
      --manifest-json "${GROUP_INTRUSIVE_MANIFEST_JSON}" \
      --overwrite
  done
else
  echo "NOTE: reverberant-LibriCHiME-5 data not found: ${REVERB_EVAL_ROOT_DIR}"
  echo "Prepare it first:"
  echo "  benchmarks/scripts/fetch_ns_chime7_udase_librispeech_dev_test_clean.sh --extract"
  echo "  benchmarks/scripts/fetch_ns_chime7_udase_voicehome.sh --extract"
  echo "  (optional) benchmarks/scripts/fetch_ns_chime7_udase_chime6_eval.sh --extract --print-root"
  echo "  edit: benchmarks/ns_chime7_udase/config/reverberant_librichime5_prepare.yaml"
  echo "  benchmarks/scripts/prepare_ns_chime7_udase_reverberant_librichime5.sh"
  echo "Or run the all-in-one script:"
  echo "  benchmarks/scripts/run_ns_chime7_udase_all.sh ${RUN_ID} --chime6-eval"
fi

date -u +"%Y-%m-%dT%H:%M:%SZ" > "${META_DIR}/finished_at_utc.txt"

echo "Archiving run and removing runs/<run_id>..."
"${BENCH_DIR}/../scripts/archive_ns_chime7_udase_run.sh" "${RUN_ID}"

echo "Done. (Archived under ${BENCH_DIR}/archives)"
