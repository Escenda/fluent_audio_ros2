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
class DtlnConfig:
    model_1_path: Path
    model_2_path: Path
    block_len: int
    block_shift: int
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


def _parse_config(cfg: dict[str, Any], root_dir: Path) -> tuple[DtlnConfig, AudioConfig, RunConfig]:
    dtln = _require(cfg, "dtln")
    audio = _require(cfg, "audio")
    run = _require(cfg, "run")
    if not isinstance(dtln, dict) or not isinstance(audio, dict) or not isinstance(run, dict):
        raise ValueError("dtln/audio/run must be mappings")

    ort_cfg = _require(dtln, "ort")
    if not isinstance(ort_cfg, dict):
        raise ValueError("dtln.ort must be a mapping")

    clip_cfg = _require(audio, "output_clip")
    if not isinstance(clip_cfg, dict):
        raise ValueError("audio.output_clip must be a mapping")

    dtln_config = DtlnConfig(
        model_1_path=(root_dir / str(_require(dtln, "model_1_path"))).resolve(),
        model_2_path=(root_dir / str(_require(dtln, "model_2_path"))).resolve(),
        block_len=int(_require(dtln, "block_len")),
        block_shift=int(_require(dtln, "block_shift")),
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

    if dtln_config.block_len <= 0 or dtln_config.block_shift <= 0:
        raise ValueError("dtln.block_len/block_shift must be > 0")
    if dtln_config.block_shift > dtln_config.block_len:
        raise ValueError("dtln.block_shift must be <= dtln.block_len")
    if audio_config.multi_channel_mode not in {"error", "first", "average"}:
        raise ValueError("audio.multi_channel_mode must be one of: error, first, average")
    if audio_config.output_clip_min >= audio_config.output_clip_max:
        raise ValueError("audio.output_clip.min must be < audio.output_clip.max")

    return dtln_config, audio_config, run_config


class DtlnOnnx:
    def __init__(self, config: DtlnConfig):
        self.config = config

        if not self.config.model_1_path.exists():
            raise FileNotFoundError(str(self.config.model_1_path))
        if not self.config.model_2_path.exists():
            raise FileNotFoundError(str(self.config.model_2_path))

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, self.config.intra_op_num_threads)
        opts.inter_op_num_threads = max(1, self.config.inter_op_num_threads)
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.config.enable_optimizations
            else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )

        self.sess1 = ort.InferenceSession(str(self.config.model_1_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.sess2 = ort.InferenceSession(str(self.config.model_2_path), sess_options=opts, providers=["CPUExecutionProvider"])

        # Model 1
        in1 = self.sess1.get_inputs()
        if len(in1) < 2:
            raise RuntimeError("model_1 requires 2 inputs: [mag, state]")
        self.mag_name = in1[0].name
        self.state1_name = in1[1].name

        out1 = self.sess1.get_outputs()
        if len(out1) < 2:
            raise RuntimeError("model_1 requires 2 outputs: [mask, state]")
        self.mask_name = out1[0].name
        self.state1_out_name = out1[1].name

        # Model 2
        in2 = self.sess2.get_inputs()
        if len(in2) < 2:
            raise RuntimeError("model_2 requires 2 inputs: [time, state]")
        self.time_name = in2[0].name
        self.state2_name = in2[1].name

        out2 = self.sess2.get_outputs()
        if len(out2) < 2:
            raise RuntimeError("model_2 requires 2 outputs: [time, state]")
        self.time_out_name = out2[0].name
        self.state2_out_name = out2[1].name

        # Validate shapes against params (no hardcoded dims).
        mag_shape = in1[0].shape
        time_shape = in2[0].shape
        state_shape = in1[1].shape
        if mag_shape != [1, 1, (self.config.block_len // 2) + 1]:
            raise RuntimeError(f"Unexpected model_1 mag shape: {mag_shape}")
        if time_shape != [1, 1, self.config.block_len]:
            raise RuntimeError(f"Unexpected model_2 time shape: {time_shape}")
        if state_shape != in2[1].shape:
            raise RuntimeError("model_1/model_2 state shapes differ")

        self.state1 = np.zeros(state_shape, dtype=np.float32)
        self.state2 = np.zeros(state_shape, dtype=np.float32)

        self.in_buffer = np.zeros((self.config.block_len,), dtype=np.float32)
        self.out_buffer = np.zeros((self.config.block_len,), dtype=np.float32)

    def reset(self) -> None:
        self.state1.fill(0.0)
        self.state2.fill(0.0)
        self.in_buffer.fill(0.0)
        self.out_buffer.fill(0.0)

    def process(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 1:
            raise ValueError("Input must be mono (1D)")
        if x.dtype != np.float32:
            x = x.astype(np.float32, copy=False)

        hop = self.config.block_shift
        n = int(x.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=np.float32)

        padded_len = ((n + hop - 1) // hop) * hop
        if padded_len != n:
            x = np.pad(x, (0, padded_len - n), mode="constant", constant_values=0.0)

        y = np.zeros((padded_len,), dtype=np.float32)

        for idx in range(0, padded_len, hop):
            hop_in = x[idx : idx + hop]

            # shift input buffer and append hop
            self.in_buffer[:-hop] = self.in_buffer[hop:]
            self.in_buffer[-hop:] = hop_in

            # FFT -> magnitude
            in_fft = np.fft.rfft(self.in_buffer)
            mag = np.abs(in_fft).astype(np.float32)[None, None, :]

            mask, self.state1 = self.sess1.run(
                [self.mask_name, self.state1_out_name],
                {self.mag_name: mag, self.state1_name: self.state1},
            )
            mask = mask.astype(np.float32, copy=False).reshape(-1)
            est_fft = in_fft * mask
            time_in = np.fft.irfft(est_fft, n=self.config.block_len).astype(np.float32, copy=False)[None, None, :]

            out_block, self.state2 = self.sess2.run(
                [self.time_out_name, self.state2_out_name],
                {self.time_name: time_in, self.state2_name: self.state2},
            )
            out_block = out_block.astype(np.float32, copy=False).reshape(-1)

            # overlap-add buffer update
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
    dtln_cfg, audio_cfg, run_cfg = _parse_config(cfg, root_dir=root_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)

    dtln = DtlnOnnx(dtln_cfg)

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

        dtln.reset()
        t0 = time.perf_counter()
        y = dtln.process(x)
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
        "tool": "run_listening_test_dtln_onnx",
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
        "dtln": {
            "model_1_path": str(dtln_cfg.model_1_path),
            "model_2_path": str(dtln_cfg.model_2_path),
            "block_len": dtln_cfg.block_len,
            "block_shift": dtln_cfg.block_shift,
            "ort": {
                "intra_op_num_threads": dtln_cfg.intra_op_num_threads,
                "inter_op_num_threads": dtln_cfg.inter_op_num_threads,
                "enable_optimizations": dtln_cfg.enable_optimizations,
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

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_path": str(cfg_path),
        "dtln_models": {
            "model_1": {"path": str(dtln_cfg.model_1_path), "sha256": _sha256_file(dtln_cfg.model_1_path)},
            "model_2": {"path": str(dtln_cfg.model_2_path), "sha256": _sha256_file(dtln_cfg.model_2_path)},
        },
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

