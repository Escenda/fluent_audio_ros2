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
import onnxruntime as ort
import soundfile as sf
import yaml


@dataclass
class GtcrnConfig:
    model_path: Path
    fft_size: int
    hop_length: int
    win_length: int
    window: str
    intra_op_num_threads: int
    inter_op_num_threads: int
    enable_optimizations: bool


@dataclass
class AudioConfig:
    expected_sample_rate_hz: int
    expected_channels: int
    multi_channel_mode: str
    output_format: str
    output_subtype: str
    output_clip_enabled: bool
    output_clip_min: float
    output_clip_max: float


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


def _parse_config(cfg: dict[str, Any], root_dir: Path) -> tuple[GtcrnConfig, AudioConfig, RunConfig]:
    gtcrn = _require(cfg, "gtcrn")
    audio = _require(cfg, "audio")
    run = _require(cfg, "run")
    if not isinstance(gtcrn, dict) or not isinstance(audio, dict) or not isinstance(run, dict):
        raise ValueError("gtcrn/audio/run must be mappings")

    ort_cfg = _require(gtcrn, "ort")
    if not isinstance(ort_cfg, dict):
        raise ValueError("gtcrn.ort must be a mapping")

    clip_cfg = _require(audio, "output_clip")
    if not isinstance(clip_cfg, dict):
        raise ValueError("audio.output_clip must be a mapping")

    gtcrn_config = GtcrnConfig(
        model_path=(root_dir / str(_require(gtcrn, "model_path"))).resolve(),
        fft_size=int(_require(gtcrn, "fft_size")),
        hop_length=int(_require(gtcrn, "hop_length")),
        win_length=int(_require(gtcrn, "win_length")),
        window=str(_require(gtcrn, "window")),
        intra_op_num_threads=int(_require(ort_cfg, "intra_op_num_threads")),
        inter_op_num_threads=int(_require(ort_cfg, "inter_op_num_threads")),
        enable_optimizations=bool(_require(ort_cfg, "enable_optimizations")),
    )

    audio_config = AudioConfig(
        expected_sample_rate_hz=int(_require(audio, "expected_sample_rate_hz")),
        expected_channels=int(_require(audio, "expected_channels")),
        multi_channel_mode=str(_require(audio, "multi_channel_mode")),
        output_format=str(_require(audio, "output_format")),
        output_subtype=str(_require(audio, "output_subtype")),
        output_clip_enabled=bool(_require(clip_cfg, "enabled")),
        output_clip_min=float(_require(clip_cfg, "min")),
        output_clip_max=float(_require(clip_cfg, "max")),
    )

    run_config = RunConfig(
        glob=str(_require(run, "glob")),
        sort=bool(_require(run, "sort")),
    )

    if gtcrn_config.fft_size <= 0 or gtcrn_config.hop_length <= 0 or gtcrn_config.win_length <= 0:
        raise ValueError("gtcrn.fft_size/hop_length/win_length must be > 0")
    if gtcrn_config.hop_length > gtcrn_config.win_length:
        raise ValueError("gtcrn.hop_length must be <= gtcrn.win_length")
    if gtcrn_config.win_length != gtcrn_config.fft_size:
        raise ValueError("gtcrn.win_length must match gtcrn.fft_size (no zero-padding)")
    if audio_config.multi_channel_mode not in {"error", "first", "average"}:
        raise ValueError("audio.multi_channel_mode must be one of: error, first, average")
    if audio_config.output_clip_min >= audio_config.output_clip_max:
        raise ValueError("audio.output_clip.min must be < audio.output_clip.max")

    return gtcrn_config, audio_config, run_config


def _sqrt_hann_periodic(win_length: int) -> np.ndarray:
    n = np.arange(win_length, dtype=np.float64)
    hann = 0.5 - 0.5 * np.cos((2.0 * np.pi * n) / float(win_length))
    return np.sqrt(np.maximum(hann, 0.0)).astype(np.float32)


class GtcrnOnnx:
    def __init__(self, config: GtcrnConfig):
        self.config = config

        if not self.config.model_path.exists():
            raise FileNotFoundError(str(self.config.model_path))
        if self.config.window != "sqrt_hann_periodic":
            raise ValueError(f"Unsupported window: {self.config.window}")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, self.config.intra_op_num_threads)
        opts.inter_op_num_threads = max(1, self.config.inter_op_num_threads)
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.config.enable_optimizations
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

        self.sess = ort.InferenceSession(str(self.config.model_path), sess_options=opts, providers=["CPUExecutionProvider"])

        ins = self.sess.get_inputs()
        outs = self.sess.get_outputs()
        if len(ins) < 4 or len(outs) < 4:
            raise RuntimeError("gtcrn.onnx requires 4 inputs and 4 outputs")

        in_names = {i.name for i in ins}
        out_names = {o.name for o in outs}
        for name in ("mix", "conv_cache", "tra_cache", "inter_cache"):
            if name not in in_names:
                raise RuntimeError(f"Missing required input: {name}")
        for name in ("enh", "conv_cache_out", "tra_cache_out", "inter_cache_out"):
            if name not in out_names:
                raise RuntimeError(f"Missing required output: {name}")

        self.mix_name = "mix"
        self.conv_cache_name = "conv_cache"
        self.tra_cache_name = "tra_cache"
        self.inter_cache_name = "inter_cache"

        self.enh_name = "enh"
        self.conv_cache_out_name = "conv_cache_out"
        self.tra_cache_out_name = "tra_cache_out"
        self.inter_cache_out_name = "inter_cache_out"

        mix_shape = next(i.shape for i in ins if i.name == "mix")
        if mix_shape != [1, (self.config.fft_size // 2) + 1, 1, 2]:
            raise RuntimeError(f"Unexpected mix shape: {mix_shape}")

        self.bins = int((self.config.fft_size // 2) + 1)
        self.window = _sqrt_hann_periodic(self.config.win_length)

        self.conv_cache = np.zeros(next(i.shape for i in ins if i.name == "conv_cache"), dtype=np.float32)
        self.tra_cache = np.zeros(next(i.shape for i in ins if i.name == "tra_cache"), dtype=np.float32)
        self.inter_cache = np.zeros(next(i.shape for i in ins if i.name == "inter_cache"), dtype=np.float32)

        self.in_buffer = np.zeros((self.config.win_length,), dtype=np.float32)
        self.out_buffer = np.zeros((self.config.win_length,), dtype=np.float32)

    def reset(self) -> None:
        self.conv_cache.fill(0.0)
        self.tra_cache.fill(0.0)
        self.inter_cache.fill(0.0)
        self.in_buffer.fill(0.0)
        self.out_buffer.fill(0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError("Input must be mono (1D)")
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        hop = self.config.hop_length
        n = int(x.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        padded_len = ((n + hop - 1) // hop) * hop
        if padded_len != n:
            x = np.pad(x, (0, padded_len - n), mode="constant", constant_values=0.0)

        y = np.zeros((padded_len,), dtype=np.float32)

        for idx in range(0, padded_len, hop):
            hop_in = x[idx : idx + hop]

            self.in_buffer[:-hop] = self.in_buffer[hop:]
            self.in_buffer[-hop:] = hop_in

            frame = self.in_buffer * self.window
            spec = np.fft.rfft(frame, n=self.config.fft_size)

            mix = np.zeros((1, self.bins, 1, 2), dtype=np.float32)
            mix[0, :, 0, 0] = spec.real.astype(np.float32, copy=False)
            mix[0, :, 0, 1] = spec.imag.astype(np.float32, copy=False)

            enh, conv_out, tra_out, inter_out = self.sess.run(
                [self.enh_name, self.conv_cache_out_name, self.tra_cache_out_name, self.inter_cache_out_name],
                {
                    self.mix_name: mix,
                    self.conv_cache_name: self.conv_cache,
                    self.tra_cache_name: self.tra_cache,
                    self.inter_cache_name: self.inter_cache,
                },
            )

            self.conv_cache = conv_out.astype(np.float32, copy=False)
            self.tra_cache = tra_out.astype(np.float32, copy=False)
            self.inter_cache = inter_out.astype(np.float32, copy=False)

            enh = enh.astype(np.float32, copy=False)
            enh_spec = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
            out_frame = np.fft.irfft(enh_spec, n=self.config.fft_size).astype(np.float32, copy=False)
            out_block = out_frame * self.window

            self.out_buffer[:-hop] = self.out_buffer[hop:]
            self.out_buffer[-hop:] = 0.0
            self.out_buffer += out_block

            y[idx : idx + hop] = self.out_buffer[:hop]

        return y[:n]


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


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64), dtype=np.float64)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--metrics-csv", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--manifest-json", required=True)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    cfg_path = Path(args.config).resolve()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    metrics_csv = Path(args.metrics_csv).resolve()
    summary_json = Path(args.summary_json).resolve()
    manifest_json = Path(args.manifest_json).resolve()

    cfg = _load_yaml(cfg_path)
    gtcrn_cfg, audio_cfg, run_cfg = _parse_config(cfg, root_dir=root_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    gtcrn = GtcrnOnnx(gtcrn_cfg)

    wav_paths = list(input_dir.glob(run_cfg.glob))
    if run_cfg.sort:
        wav_paths = sorted(wav_paths)
    if args.limit and args.limit > 0:
        wav_paths = wav_paths[: args.limit]
    if not wav_paths:
        raise RuntimeError(f"No input files found: {input_dir}/{run_cfg.glob}")

    total_input_sec = 0.0
    total_proc_sec = 0.0
    max_rtf = 0.0

    rows: list[dict[str, Any]] = []

    for wav_path in wav_paths:
        info = sf.info(str(wav_path))
        if info.samplerate != audio_cfg.expected_sample_rate_hz:
            raise RuntimeError(f"Sample rate mismatch: {wav_path} {info.samplerate} != {audio_cfg.expected_sample_rate_hz}")
        if info.channels != audio_cfg.expected_channels:
            if audio_cfg.expected_channels == 1:
                pass
            else:
                raise RuntimeError(f"Channel mismatch: {wav_path} {info.channels} != {audio_cfg.expected_channels}")

        x, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        if sr != audio_cfg.expected_sample_rate_hz:
            raise RuntimeError(f"Sample rate mismatch (read): {wav_path} {sr} != {audio_cfg.expected_sample_rate_hz}")
        x = _audio_to_mono(np.asarray(x), mode=audio_cfg.multi_channel_mode).astype(np.float32, copy=False)

        gtcrn.reset()
        t0 = time.perf_counter()
        y = gtcrn.process(x)
        t1 = time.perf_counter()
        proc_sec = float(t1 - t0)

        if audio_cfg.output_clip_enabled:
            y = np.clip(y, audio_cfg.output_clip_min, audio_cfg.output_clip_max)

        out_path = output_dir / wav_path.name
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(str(out_path))

        sf.write(
            str(out_path),
            y,
            audio_cfg.expected_sample_rate_hz,
            format=audio_cfg.output_format,
            subtype=audio_cfg.output_subtype,
        )

        duration_sec = float(y.shape[0]) / float(audio_cfg.expected_sample_rate_hz)
        rtf = proc_sec / duration_sec if duration_sec > 0 else 0.0

        total_input_sec += duration_sec
        total_proc_sec += proc_sec
        max_rtf = max(max_rtf, rtf)

        rows.append(
            {
                "file": wav_path.name,
                "input_path": str(wav_path),
                "output_path": str(out_path),
                "sr_hz": audio_cfg.expected_sample_rate_hz,
                "samples": int(y.shape[0]),
                "duration_sec": duration_sec,
                "proc_sec": proc_sec,
                "rtf": rtf,
                "in_rms": _rms(x),
                "out_rms": _rms(y),
                "in_peak": float(np.max(np.abs(x))) if x.size else 0.0,
                "out_peak": float(np.max(np.abs(y))) if y.size else 0.0,
            }
        )

    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "tool": "run_listening_test_gtcrn_onnx",
        "onnxruntime_version": ort.__version__,
        "numpy_version": np.__version__,
        "soundfile_version": sf.__version__,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "file_count": len(rows),
        "total_input_sec": total_input_sec,
        "total_proc_sec": total_proc_sec,
        "avg_rtf": (total_proc_sec / total_input_sec) if total_input_sec > 0 else 0.0,
        "max_rtf": max_rtf,
        "gtcrn": {
            "model_path": str(gtcrn_cfg.model_path),
            "fft_size": gtcrn_cfg.fft_size,
            "hop_length": gtcrn_cfg.hop_length,
            "win_length": gtcrn_cfg.win_length,
            "window": gtcrn_cfg.window,
            "ort": {
                "intra_op_num_threads": gtcrn_cfg.intra_op_num_threads,
                "inter_op_num_threads": gtcrn_cfg.inter_op_num_threads,
                "enable_optimizations": gtcrn_cfg.enable_optimizations,
            },
        },
        "audio": {
            "expected_sample_rate_hz": audio_cfg.expected_sample_rate_hz,
            "expected_channels": audio_cfg.expected_channels,
            "multi_channel_mode": audio_cfg.multi_channel_mode,
            "output_format": audio_cfg.output_format,
            "output_subtype": audio_cfg.output_subtype,
            "output_clip": {
                "enabled": audio_cfg.output_clip_enabled,
                "min": audio_cfg.output_clip_min,
                "max": audio_cfg.output_clip_max,
            },
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(cfg_path),
        "gtcrn_model": {"path": str(gtcrn_cfg.model_path), "sha256": _sha256_file(gtcrn_cfg.model_path)},
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

    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

