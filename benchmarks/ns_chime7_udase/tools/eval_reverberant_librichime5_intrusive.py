#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml

from pesq import pesq as pesq_fn
from pystoi import stoi as stoi_fn


def _sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_bytes), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping")
    return data


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise KeyError(f"Missing required key: {key}")
    return mapping[key]


@dataclass
class AudioConfig:
    sample_rate_hz: int


@dataclass
class AlignmentConfig:
    mode: str  # strict|min
    pred_trim_start_samples: int
    pred_pad_end: bool


@dataclass
class SiSdrConfig:
    enabled: bool
    zero_mean: bool
    eps: float


@dataclass
class PesqConfig:
    enabled: bool
    mode: str  # nb|wb


@dataclass
class StoiConfig:
    enabled: bool
    extended: bool


@dataclass
class RunConfig:
    glob_mix: str
    sort: bool


def _parse_config(cfg: dict[str, Any]) -> tuple[AudioConfig, AlignmentConfig, SiSdrConfig, PesqConfig, StoiConfig, RunConfig]:
    audio = _require(cfg, "audio")
    alignment = _require(cfg, "alignment")
    si_sdr = _require(cfg, "si_sdr")
    pesq = _require(cfg, "pesq")
    stoi = _require(cfg, "stoi")
    run = _require(cfg, "run")

    if not all(isinstance(x, dict) for x in (audio, alignment, si_sdr, pesq, stoi, run)):
        raise ValueError("audio/alignment/si_sdr/pesq/stoi/run must be mappings")

    audio_cfg = AudioConfig(sample_rate_hz=int(_require(audio, "sample_rate_hz")))
    pred_trim_start_samples = int(alignment.get("pred_trim_start_samples", 0))
    pred_pad_end = bool(alignment.get("pred_pad_end", False))
    alignment_cfg = AlignmentConfig(
        mode=str(_require(alignment, "mode")),
        pred_trim_start_samples=pred_trim_start_samples,
        pred_pad_end=pred_pad_end,
    )
    si_sdr_cfg = SiSdrConfig(
        enabled=bool(_require(si_sdr, "enabled")),
        zero_mean=bool(_require(si_sdr, "zero_mean")),
        eps=float(_require(si_sdr, "eps")),
    )
    pesq_cfg = PesqConfig(enabled=bool(_require(pesq, "enabled")), mode=str(_require(pesq, "mode")))
    stoi_cfg = StoiConfig(enabled=bool(_require(stoi, "enabled")), extended=bool(_require(stoi, "extended")))
    run_cfg = RunConfig(glob_mix=str(_require(run, "glob_mix")), sort=bool(_require(run, "sort")))

    if audio_cfg.sample_rate_hz <= 0:
        raise ValueError("audio.sample_rate_hz must be > 0")
    if alignment_cfg.mode not in {"strict", "min"}:
        raise ValueError("alignment.mode must be strict|min")
    if alignment_cfg.pred_trim_start_samples < 0:
        raise ValueError("alignment.pred_trim_start_samples must be >= 0")
    if si_sdr_cfg.eps <= 0:
        raise ValueError("si_sdr.eps must be > 0")
    if pesq_cfg.mode not in {"nb", "wb"}:
        raise ValueError("pesq.mode must be nb|wb")

    return audio_cfg, alignment_cfg, si_sdr_cfg, pesq_cfg, stoi_cfg, run_cfg


def _read_audio(path: Path, expected_sr: int) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise RuntimeError(f"Sample rate mismatch: {path} {sr} != {expected_sr}")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        if x.shape[1] != 1:
            raise RuntimeError(f"Expected mono audio: {path} shape={x.shape}")
        x = x[:, 0]
    if x.ndim != 1:
        raise RuntimeError(f"Unexpected audio ndim: {path} ndim={x.ndim}")
    return x


def _align_multi(mode: str, *signals: np.ndarray) -> tuple[np.ndarray, ...]:
    if mode == "strict":
        if not signals:
            raise ValueError("No signals to align")
        n0 = int(signals[0].shape[0])
        for sig in signals[1:]:
            if int(sig.shape[0]) != n0:
                raise RuntimeError(f"Length mismatch: expected={n0} got={sig.shape[0]}")
        return signals
    if mode == "min":
        n = min(int(sig.shape[0]) for sig in signals)
        return tuple(sig[:n] for sig in signals)
    raise ValueError(f"Unknown alignment mode: {mode}")


def _si_sdr(ref: np.ndarray, est: np.ndarray, zero_mean: bool, eps: float) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    est = np.asarray(est, dtype=np.float64)
    if zero_mean:
        ref = ref - np.mean(ref)
        est = est - np.mean(est)
    ref_energy = float(np.sum(ref * ref))
    if ref_energy <= eps:
        return float("-inf")
    alpha = float(np.sum(est * ref) / (ref_energy + eps))
    target = alpha * ref
    noise = est - target
    target_energy = float(np.sum(target * target))
    noise_energy = float(np.sum(noise * noise))
    return 10.0 * float(np.log10((target_energy + eps) / (noise_energy + eps)))


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--subset", required=True)  # eval/dev
    ap.add_argument("--group", required=True, type=int)  # 1/2/3
    ap.add_argument("--pred-dir", default="")
    ap.add_argument("--pred-method", default="dtln_onnx")
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = _load_yaml(cfg_path)
    audio_cfg, alignment_cfg, si_sdr_cfg, pesq_cfg, stoi_cfg, run_cfg = _parse_config(cfg)

    dataset_root = Path(args.dataset_root).resolve()
    subset = str(args.subset)
    group = int(args.group)
    group_dir = (dataset_root / subset / str(group)).resolve()
    if not group_dir.exists():
        raise FileNotFoundError(str(group_dir))

    pred_dir = Path(args.pred_dir).resolve() if args.pred_dir else None
    out_csv = Path(args.out_csv).resolve()
    summary_json = Path(args.summary_json).resolve()
    manifest_json = Path(args.manifest_json).resolve()
    for p in (out_csv, summary_json, manifest_json):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists() and not args.overwrite:
            raise FileExistsError(str(p))

    mix_paths = list(group_dir.glob(run_cfg.glob_mix))
    if run_cfg.sort:
        mix_paths = sorted(mix_paths)
    if args.limit and args.limit > 0:
        mix_paths = mix_paths[: args.limit]
    if not mix_paths:
        raise RuntimeError(f"No mix files found: {group_dir}/{run_cfg.glob_mix}")

    dataset_name = "reverberant-LibriCHiME-5"
    subset_name = f"{subset}/{group}"

    rows: list[dict[str, Any]] = []
    sig_sdr_in: list[float] = []
    sig_sdr_out: list[float] = []
    pesq_in: list[float] = []
    pesq_out: list[float] = []
    stoi_in: list[float] = []
    stoi_out: list[float] = []

    t0 = time.perf_counter()
    for mix_path in mix_paths:
        if not mix_path.name.endswith("_mix.wav"):
            continue
        speech_path = mix_path.with_name(mix_path.name.replace("_mix.wav", "_speech.wav"))
        if not speech_path.exists():
            raise FileNotFoundError(str(speech_path))

        mix_sig = _read_audio(mix_path, expected_sr=audio_cfg.sample_rate_hz)
        ref_sig = _read_audio(speech_path, expected_sr=audio_cfg.sample_rate_hz)

        pred_sig = None
        if pred_dir is not None:
            pred_path = (pred_dir / mix_path.name).resolve()
            if not pred_path.exists():
                raise FileNotFoundError(str(pred_path))
            pred_sig = _read_audio(pred_path, expected_sr=audio_cfg.sample_rate_hz)
            if alignment_cfg.pred_trim_start_samples:
                if alignment_cfg.pred_trim_start_samples >= int(pred_sig.shape[0]):
                    raise RuntimeError(
                        f"pred_trim_start_samples too large: {alignment_cfg.pred_trim_start_samples} >= {pred_sig.shape[0]} ({pred_path})"
                    )
                pred_sig = pred_sig[alignment_cfg.pred_trim_start_samples :]
                if alignment_cfg.pred_pad_end:
                    pred_sig = np.pad(pred_sig, (0, alignment_cfg.pred_trim_start_samples), mode="constant")

        if pred_sig is not None:
            ref_eval, mix_eval, pred_eval = _align_multi(alignment_cfg.mode, ref_sig, mix_sig, pred_sig)
        else:
            ref_eval, mix_eval = _align_multi(alignment_cfg.mode, ref_sig, mix_sig)
            pred_eval = None

        methods: list[tuple[str, np.ndarray]] = [("input", mix_eval)]
        if pred_eval is not None:
            methods.append((args.pred_method, pred_eval))

        for method, deg_sig in methods:
            if method == "input":
                r = ref_eval
                d = deg_sig
            else:
                r = ref_eval
                d = deg_sig

            if si_sdr_cfg.enabled:
                val = _si_sdr(r, d, zero_mean=si_sdr_cfg.zero_mean, eps=si_sdr_cfg.eps)
                rows.append(
                    {
                        "dataset": dataset_name,
                        "subset": subset_name,
                        "sample": mix_path.name,
                        "metric": "SI-SDR",
                        "method": method,
                        "result": val,
                    }
                )
                (sig_sdr_in if method == "input" else sig_sdr_out).append(val)

            if pesq_cfg.enabled:
                val = float(pesq_fn(audio_cfg.sample_rate_hz, r, d, mode=pesq_cfg.mode))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "subset": subset_name,
                        "sample": mix_path.name,
                        "metric": "PESQ",
                        "method": method,
                        "result": val,
                    }
                )
                (pesq_in if method == "input" else pesq_out).append(val)

            if stoi_cfg.enabled:
                val = float(stoi_fn(r, d, audio_cfg.sample_rate_hz, extended=stoi_cfg.extended))
                rows.append(
                    {
                        "dataset": dataset_name,
                        "subset": subset_name,
                        "sample": mix_path.name,
                        "metric": "STOI",
                        "method": method,
                        "result": val,
                    }
                )
                (stoi_in if method == "input" else stoi_out).append(val)

    t1 = time.perf_counter()

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "subset", "sample", "metric", "method", "result"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "tool": "eval_reverberant_librichime5_intrusive",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "dataset_root": str(dataset_root),
        "subset": subset_name,
        "file_count": len(mix_paths),
        "total_proc_sec": float(t1 - t0),
        "methods": {
            "input": {
                "avg_SI-SDR": _mean(sig_sdr_in) if si_sdr_cfg.enabled else None,
                "avg_PESQ": _mean(pesq_in) if pesq_cfg.enabled else None,
                "avg_STOI": _mean(stoi_in) if stoi_cfg.enabled else None,
            },
        },
    }
    if pred_dir is not None:
        summary["methods"][args.pred_method] = {
            "avg_SI-SDR": _mean(sig_sdr_out) if si_sdr_cfg.enabled else None,
            "avg_PESQ": _mean(pesq_out) if pesq_cfg.enabled else None,
            "avg_STOI": _mean(stoi_out) if stoi_cfg.enabled else None,
        }
        summary["delta"] = {
            "delta_avg_SI-SDR": (_mean(sig_sdr_out) - _mean(sig_sdr_in)) if si_sdr_cfg.enabled else None,
            "delta_avg_PESQ": (_mean(pesq_out) - _mean(pesq_in)) if pesq_cfg.enabled else None,
            "delta_avg_STOI": (_mean(stoi_out) - _mean(stoi_in)) if stoi_cfg.enabled else None,
        }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(cfg_path),
        "input_dir": str(group_dir),
        "pred_dir": str(pred_dir) if pred_dir is not None else "",
        "files": [],
    }
    for mix_path in mix_paths:
        speech_path = mix_path.with_name(mix_path.name.replace("_mix.wav", "_speech.wav"))
        entry = {
            "mix": {"path": str(mix_path), "sha256": _sha256_file(mix_path)},
            "speech": {"path": str(speech_path), "sha256": _sha256_file(speech_path)},
        }
        if pred_dir is not None:
            pred_path = (pred_dir / mix_path.name).resolve()
            entry["pred"] = {"path": str(pred_path), "sha256": _sha256_file(pred_path)}
        manifest["files"].append(entry)

    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
