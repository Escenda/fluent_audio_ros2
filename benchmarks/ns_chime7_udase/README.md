# NS Benchmark: CHiME-7 UDASE (arXiv:2402.01413)

参照:
- 論文: https://arxiv.org/abs/2402.01413
- CHiME-7 UDASE evaluation data (Zenodo): https://doi.org/10.5281/zenodo.10418311

保存ポリシー:
- `docs/fa_benchmark_policy.md` に従い、使用データ/生成物/評価結果/評価に用いた全データを必ず保存します。

## ディレクトリ構成（標準）
- `benchmarks/ns_chime7_udase/data/`（git管理しない）
- `benchmarks/ns_chime7_udase/runs/<run_id>/`（一時。完走後はアーカイブして削除）
- `benchmarks/ns_chime7_udase/archives/<run_id>.tar.xz`（git管理しない。runsの退避先）
- `benchmarks/ns_chime7_udase/results/`（必要ならgit管理）

## データ取得（Zenodo）
- `benchmarks/scripts/fetch_ns_chime7_udase_eval_data.sh` で `CHiME-7-UDASE-evaluation-data.zip` を `data/zenodo_10418311/` に保存します。

```bash
benchmarks/scripts/fetch_ns_chime7_udase_eval_data.sh --extract
```

## 全自動（推奨）
CHiME-6(eval)（OpenSLR SLR150, CC BY-SA 4.0）を使うと、CHiME-5原本の手動入手なしで
（Zenodo/LibriSpeech/VoiceHomeのDL + reverberant-LibriCHiME-5生成 + 実行/評価）まで自動化できます。

```bash
benchmarks/scripts/run_ns_chime7_udase_all.sh 20260102_120000_dtln_all --chime6-eval
```

CHiME-5原本を使う場合:
```bash
benchmarks/scripts/run_ns_chime7_udase_all.sh 20260102_120000_dtln_all --chime5-data-dir /path/to/CHiME5
```

## 実行（DTLN/ONNX: listening_test C0）
このリポジトリでは Phase1 として、Zenodo ZIP に含まれる `listening_test/data/C0` を入力に、
DTLN(ONNX)で雑音抑圧した音声と実行メタ/処理時間指標、さらに DNSMOS（SIG/BAK/OVR）の品質指標を `runs/<run_id>/` に保存します。

前提（Python）:
- `onnxruntime`, `numpy`, `soundfile`, `pyyaml`

実行:
```bash
benchmarks/scripts/run_ns_chime7_udase.sh 20260101_120000_dtln_onnx_v1
```

出力:
- 生成物は完走後に `benchmarks/ns_chime7_udase/archives/<run_id>.tar.xz` に保存され、`runs/<run_id>` は削除されます。

設定:
- `benchmarks/ns_chime7_udase/config/dtln_onnx_listening_test.yaml`
- `benchmarks/ns_chime7_udase/config/dnsmos_sig_bak_ovr.yaml`

## close-to-in-domain データ（reverberant LibriCHiME-5）
論文で intrusive 指標（SI-SDR/PESQ/STOI）を算出するために使われるデータです。

参照（上流）:
- segmented CHiME-5: https://github.com/UDASE-CHiME2023/CHiME-5
- reverberant LibriCHiME-5: https://github.com/UDASE-CHiME2023/reverberant-LibriCHiME-5

### データ準備（ローカル生成）
注: CHiME-5 / VoiceHome 等のライセンス都合により、派生物を再配布できない場合があります。必ずローカルに生成・保存し、`runs/<run_id>/meta/dataset_manifest.json` に取得元とチェックサムを残します。

1) LibriSpeech (dev-clean/test-clean) を取得:
```bash
benchmarks/scripts/fetch_ns_chime7_udase_librispeech_dev_test_clean.sh --extract
```

2) VoiceHome を取得:
```bash
benchmarks/scripts/fetch_ns_chime7_udase_voicehome.sh --extract
```

3) CHiME-5 を入手してローカルに配置（要手動）:
- https://www.chimechallenge.org/datasets/chime5

代替（推奨: CHiME-6 eval / OpenSLR SLR150）:
```bash
benchmarks/scripts/fetch_ns_chime7_udase_chime6_eval.sh --extract --print-root
```
注: `reverberant_librichime5_prepare.yaml` が `subsets: [eval]` の場合、基本的には `CHiME6_eval.tar.gz` だけで足ります（devも生成する場合はCHiME6_devも必要です）。

4) 生成設定を編集:
- `benchmarks/ns_chime7_udase/config/reverberant_librichime5_prepare.yaml` の `chime5.original_data_dir` を自分の環境に合わせて変更

5) `reverberant_librichime5_audio` を生成:
```bash
benchmarks/scripts/prepare_ns_chime7_udase_reverberant_librichime5.sh
```

### 実行（侵襲指標: SI-SDR/PESQ/STOI）
前提（Python）:
- `scipy`, `pystoi`, `pesq`, `tqdm`（加えて上記のPhase1と同様に `onnxruntime` 等）

`data/reverberant_librichime5_audio/eval/{1,2,3}` が存在する場合、`run_ns_chime7_udase.sh` が自動で
DTLN出力生成と侵襲指標の算出まで行います:
```bash
benchmarks/scripts/run_ns_chime7_udase.sh 20260101_120000_dtln_onnx_v1
```

出力例:
- 生成物は完走後に `benchmarks/ns_chime7_udase/archives/<run_id>.tar.xz` に保存され、`runs/<run_id>` は削除されます。

### アーカイブの展開
```bash
mkdir -p benchmarks/ns_chime7_udase/runs
tar -xJf benchmarks/ns_chime7_udase/archives/<run_id>.tar.xz -C benchmarks/ns_chime7_udase/runs
```

設定:
- `benchmarks/ns_chime7_udase/config/reverberant_librichime5_prepare.yaml`
- `benchmarks/ns_chime7_udase/config/dtln_onnx_reverberant_librichime5_eval.yaml`
- `benchmarks/ns_chime7_udase/config/reverberant_librichime5_intrusive_metrics.yaml`
