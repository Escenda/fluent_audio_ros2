#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import soundfile as sf
import yaml


@dataclass
class DnsMosConfig:
    model_path: Path
    sample_rate_hz: int
    input_length_sec: float
    hop_sec: float
    repeat_short_audio: bool
    polyfit_sig: list[float]
    polyfit_bak: list[float]
    polyfit_ovr: list[float]


@dataclass
class AudioConfig:
    expected_channels: int
    multi_channel_mode: str


@dataclass
class RunConfig:
    glob: str
    sort: bool


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


def _parse_config(cfg: dict[str, Any], root_dir: Path) -> tuple[DnsMosConfig, AudioConfig, RunConfig]:
    dnsmos = _require(cfg, "dnsmos")
    audio = _require(cfg, "audio")
    run = _require(cfg, "run")
    if not isinstance(dnsmos, dict) or not isinstance(audio, dict) or not isinstance(run, dict):
        raise ValueError("dnsmos/audio/run must be mappings")

    polyfit = _require(dnsmos, "polyfit")
    if not isinstance(polyfit, dict):
        raise ValueError("dnsmos.polyfit must be a mapping")

    dnsmos_cfg = DnsMosConfig(
        model_path=(root_dir / str(_require(dnsmos, "model_path"))).resolve(),
        sample_rate_hz=int(_require(dnsmos, "sample_rate_hz")),
        input_length_sec=float(_require(dnsmos, "input_length_sec")),
        hop_sec=float(_require(dnsmos, "hop_sec")),
        repeat_short_audio=bool(_require(dnsmos, "repeat_short_audio")),
        polyfit_sig=[float(x) for x in _require(polyfit, "sig")],
        polyfit_bak=[float(x) for x in _require(polyfit, "bak")],
        polyfit_ovr=[float(x) for x in _require(polyfit, "ovr")],
    )

    audio_cfg = AudioConfig(
        expected_channels=int(_require(audio, "expected_channels")),
        multi_channel_mode=str(_require(audio, "multi_channel_mode")),
    )

    run_cfg = RunConfig(
        glob=str(_require(run, "glob")),
        sort=bool(_require(run, "sort")),
    )

    if dnsmos_cfg.sample_rate_hz <= 0:
        raise ValueError("dnsmos.sample_rate_hz must be > 0")
    if dnsmos_cfg.input_length_sec <= 0:
        raise ValueError("dnsmos.input_length_sec must be > 0")
    if dnsmos_cfg.hop_sec <= 0:
        raise ValueError("dnsmos.hop_sec must be > 0")
    if audio_cfg.multi_channel_mode not in {"error", "first", "average"}:
        raise ValueError("audio.multi_channel_mode must be one of: error, first, average")
    if not dnsmos_cfg.polyfit_sig or not dnsmos_cfg.polyfit_bak or not dnsmos_cfg.polyfit_ovr:
        raise ValueError("dnsmos.polyfit.* must be non-empty coefficient lists")

    return dnsmos_cfg, audio_cfg, run_cfg


def _audio_to_mono(x: np.ndarray, mode: str) -> np.ndarray:
    if x.ndim == 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"Unsupported audio array ndim={x.ndim}")
    if mode == "error":
        raise ValueError(f"Expected mono input, got shape={x.shape}")
    if mode == "first":
        return x[:, 0]
    if mode == "average":
        return np.mean(x, axis=1)
    raise ValueError(f"Unknown multi_channel_mode: {mode}")


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _std(values: list[float], mean: float) -> float:
    if not values:
        return float("nan")
    v = sum((x - mean) ** 2 for x in values) / len(values)
    return float(math.sqrt(v))


class DnsMosSigBakOvr:
    def __init__(self, cfg: DnsMosConfig):
        self.cfg = cfg
        if not self.cfg.model_path.exists():
            raise FileNotFoundError(str(self.cfg.model_path))

        self.session = ort.InferenceSession(str(self.cfg.model_path), providers=["CPUExecutionProvider"])
        in0 = self.session.get_inputs()[0]
        self.input_name = in0.name

        # Validate expected waveform length from ONNX.
        shape = in0.shape
        if len(shape) != 2:
            raise RuntimeError(f"Unexpected input rank: {shape}")
        expected_len = shape[1]
        if not isinstance(expected_len, int):
            raise RuntimeError(f"Unexpected input length dim: {shape}")

        cfg_len = int(self.cfg.input_length_sec * self.cfg.sample_rate_hz)
        if cfg_len != expected_len:
            raise RuntimeError(f"Config input length mismatch: cfg={cfg_len} onnx={expected_len}")

        out0 = self.session.get_outputs()[0]
        self.output_name = out0.name

    def score_clip(self, x: np.ndarray) -> dict[str, Any]:
        if x.ndim != 1:
            raise ValueError("Input must be mono (1D)")
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        fs = self.cfg.sample_rate_hz
        input_len_samples = int(self.cfg.input_length_sec * fs)
        hop_len_samples = int(self.cfg.hop_sec * fs)
        if hop_len_samples <= 0:
            raise ValueError("hop_len_samples must be > 0")

        audio = x
        actual_audio_len = int(audio.shape[0])

        if self.cfg.repeat_short_audio:
            while audio.shape[0] < input_len_samples:
                audio = np.append(audio, audio)
        else:
            if audio.shape[0] < input_len_samples:
                audio = np.pad(audio, (0, input_len_samples - audio.shape[0]), mode="constant", constant_values=0.0)

        num_hops = int(np.floor(audio.shape[0] / fs) - self.cfg.input_length_sec) + 1
        if num_hops <= 0:
            num_hops = 1

        sig_raw_list: list[float] = []
        bak_raw_list: list[float] = []
        ovr_raw_list: list[float] = []
        sig_list: list[float] = []
        bak_list: list[float] = []
        ovr_list: list[float] = []

        for idx in range(num_hops):
            start = int(idx * hop_len_samples)
            end = int((idx + self.cfg.input_length_sec) * hop_len_samples)
            seg = audio[start:end]
            if seg.shape[0] < input_len_samples:
                continue

            inp = seg.astype(np.float32, copy=False)[None, :]
            out = self.session.run([self.output_name], {self.input_name: inp})[0]
            sig_raw, bak_raw, ovr_raw = (float(out[0][0]), float(out[0][1]), float(out[0][2]))

            sig = float(np.polyval(self.cfg.polyfit_sig, sig_raw))
            bak = float(np.polyval(self.cfg.polyfit_bak, bak_raw))
            ovr = float(np.polyval(self.cfg.polyfit_ovr, ovr_raw))

            sig_raw_list.append(sig_raw)
            bak_raw_list.append(bak_raw)
            ovr_raw_list.append(ovr_raw)
            sig_list.append(sig)
            bak_list.append(bak)
            ovr_list.append(ovr)

        return {
            "len_in_sec": actual_audio_len / fs,
            "num_hops": int(num_hops),
            "SIG_raw": _mean(sig_raw_list),
            "BAK_raw": _mean(bak_raw_list),
            "OVR_raw": _mean(ovr_raw_list),
            "SIG_MOS": _mean(sig_list),
            "BAK_MOS": _mean(bak_list),
            "OVR_MOS": _mean(ovr_list),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    cfg_path = Path(args.config).resolve()
    input_dir = Path(args.input_dir).resolve()
    metrics_csv = Path(args.metrics_csv).resolve()
    summary_json = Path(args.summary_json).resolve()
    manifest_json = Path(args.manifest_json).resolve()

    cfg = _load_yaml(cfg_path)
    dnsmos_cfg, audio_cfg, run_cfg = _parse_config(cfg, root_dir=root_dir)

    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    scorer = DnsMosSigBakOvr(dnsmos_cfg)

    wav_paths = list(input_dir.glob(run_cfg.glob))
    if run_cfg.sort:
        wav_paths = sorted(wav_paths)
    if args.limit and args.limit > 0:
        wav_paths = wav_paths[: args.limit]
    if not wav_paths:
        raise RuntimeError(f"No input files found: {input_dir}/{run_cfg.glob}")

    rows: list[dict[str, Any]] = []

    t0 = time.perf_counter()
    for wav_path in wav_paths:
        info = sf.info(str(wav_path))
        if info.samplerate != dnsmos_cfg.sample_rate_hz:
            raise RuntimeError(f"Sample rate mismatch: {wav_path} {info.samplerate} != {dnsmos_cfg.sample_rate_hz}")

        x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sr != dnsmos_cfg.sample_rate_hz:
            raise RuntimeError(f"Sample rate mismatch (read): {wav_path} {sr} != {dnsmos_cfg.sample_rate_hz}")

        x = _audio_to_mono(np.asarray(x), mode=audio_cfg.multi_channel_mode).astype(np.float32, copy=False)
        if audio_cfg.expected_channels != 1:
            raise RuntimeError("Only mono scoring is currently supported (expected_channels must be 1)")

        scores = scorer.score_clip(x)
        row = {
            "file": wav_path.name,
            "input_path": str(wav_path),
            "duration_sec": float(scores["len_in_sec"]),
            "num_hops": int(scores["num_hops"]),
            "SIG_MOS": float(scores["SIG_MOS"]),
            "BAK_MOS": float(scores["BAK_MOS"]),
            "OVR_MOS": float(scores["OVR_MOS"]),
            "SIG_raw": float(scores["SIG_raw"]),
            "BAK_raw": float(scores["BAK_raw"]),
            "OVR_raw": float(scores["OVR_raw"]),
        }
        rows.append(row)

    t1 = time.perf_counter()

    if metrics_csv.exists() and not args.overwrite:
        raise FileExistsError(str(metrics_csv))
    if summary_json.exists() and not args.overwrite:
        raise FileExistsError(str(summary_json))
    if manifest_json.exists() and not args.overwrite:
        raise FileExistsError(str(manifest_json))

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    sig = [r["SIG_MOS"] for r in rows]
    bak = [r["BAK_MOS"] for r in rows]
    ovr = [r["OVR_MOS"] for r in rows]

    sig_mean = _mean(sig)
    bak_mean = _mean(bak)
    ovr_mean = _mean(ovr)

    summary = {
        "tool": "score_dnsmos_sig_bak_ovr",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "soundfile_version": sf.__version__,
        "config_path": str(cfg_path),
        "input_dir": str(input_dir),
        "file_count": len(rows),
        "total_proc_sec": float(t1 - t0),
        "avg_SIG_MOS": sig_mean,
        "avg_BAK_MOS": bak_mean,
        "avg_OVR_MOS": ovr_mean,
        "std_SIG_MOS": _std(sig, sig_mean),
        "std_BAK_MOS": _std(bak, bak_mean),
        "std_OVR_MOS": _std(ovr, ovr_mean),
        "dnsmos": {
            "model_path": str(dnsmos_cfg.model_path),
            "model_sha256": _sha256_file(dnsmos_cfg.model_path),
            "sample_rate_hz": dnsmos_cfg.sample_rate_hz,
            "input_length_sec": dnsmos_cfg.input_length_sec,
            "hop_sec": dnsmos_cfg.hop_sec,
            "repeat_short_audio": dnsmos_cfg.repeat_short_audio,
            "polyfit": {
                "sig": dnsmos_cfg.polyfit_sig,
                "bak": dnsmos_cfg.polyfit_bak,
                "ovr": dnsmos_cfg.polyfit_ovr,
            },
        },
    }

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(cfg_path),
        "dnsmos_model": {"path": str(dnsmos_cfg.model_path), "sha256": _sha256_file(dnsmos_cfg.model_path)},
        "input_files": [],
    }

    for wav_path in wav_paths:
        info = sf.info(str(wav_path))
        manifest["input_files"].append(
            {
                "path": str(wav_path.resolve()),
                "sha256": _sha256_file(wav_path),
                "bytes": wav_path.stat().st_size,
                "samplerate_hz": int(info.samplerate),
                "channels": int(info.channels),
                "subtype": str(info.subtype),
                "duration_sec": float(info.frames) / float(info.samplerate) if info.samplerate else 0.0,
            }
        )

    with manifest_json.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

