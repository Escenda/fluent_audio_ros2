#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import yaml
from scipy import signal
from tqdm import tqdm


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
class Chime5ExtractorConfig:
    original_data_dir: Path
    segments_output_dir: Path
    json_dir: Path
    extract_stereo: bool
    eval_only: bool


@dataclass
class DatasetPathsConfig:
    librispeech_dir: Path
    voicehome_dir: Path


@dataclass
class ReverberantLibriChime5Config:
    metadata_dir: Path
    output_dir: Path
    subsets: list[str]


@dataclass
class AudioConfig:
    sample_rate_hz: int
    chime5_noise_channel_index: int
    clipping_threshold: float
    clipping_max_amp: float
    output_format: str
    output_subtype: str


@dataclass
class SnrConfig:
    verify_isclose: bool
    atol: float
    rtol: float


@dataclass
class RunConfig:
    overwrite: bool
    limit: int


def _parse_config(cfg: dict[str, Any], root_dir: Path) -> tuple[
    Chime5ExtractorConfig, DatasetPathsConfig, ReverberantLibriChime5Config, AudioConfig, SnrConfig, RunConfig
]:
    chime5 = _require(cfg, "chime5")
    datasets = _require(cfg, "datasets")
    rev = _require(cfg, "reverberant_librichime5")
    audio = _require(cfg, "audio")
    snr = _require(cfg, "snr")
    run = _require(cfg, "run")

    if not isinstance(chime5, dict) or not isinstance(datasets, dict) or not isinstance(rev, dict):
        raise ValueError("chime5/datasets/reverberant_librichime5 must be mappings")
    if not isinstance(audio, dict) or not isinstance(snr, dict) or not isinstance(run, dict):
        raise ValueError("audio/snr/run must be mappings")

    extractor = _require(chime5, "extractor")
    if not isinstance(extractor, dict):
        raise ValueError("chime5.extractor must be a mapping")

    clipping = _require(audio, "clipping")
    output = _require(audio, "output")
    if not isinstance(clipping, dict) or not isinstance(output, dict):
        raise ValueError("audio.clipping/audio.output must be mappings")

    chime5_cfg = Chime5ExtractorConfig(
        original_data_dir=Path(str(_require(chime5, "original_data_dir"))).expanduser(),
        segments_output_dir=(root_dir / str(_require(chime5, "segments_output_dir"))).resolve(),
        json_dir=(root_dir / str(_require(chime5, "json_dir"))).resolve(),
        extract_stereo=bool(_require(extractor, "extract_stereo")),
        eval_only=bool(_require(extractor, "eval_only")),
    )

    datasets_cfg = DatasetPathsConfig(
        librispeech_dir=(root_dir / str(_require(datasets, "librispeech_dir"))).resolve(),
        voicehome_dir=(root_dir / str(_require(datasets, "voicehome_dir"))).resolve(),
    )

    rev_cfg = ReverberantLibriChime5Config(
        metadata_dir=(root_dir / str(_require(rev, "metadata_dir"))).resolve(),
        output_dir=(root_dir / str(_require(rev, "output_dir"))).resolve(),
        subsets=[str(x) for x in _require(rev, "subsets")],
    )

    audio_cfg = AudioConfig(
        sample_rate_hz=int(_require(_require(cfg, "audio"), "sample_rate_hz")),
        chime5_noise_channel_index=int(_require(audio, "chime5_noise_channel_index")),
        clipping_threshold=float(_require(clipping, "threshold")),
        clipping_max_amp=float(_require(clipping, "max_amp")),
        output_format=str(_require(output, "format")),
        output_subtype=str(_require(output, "subtype")),
    )

    snr_cfg = SnrConfig(
        verify_isclose=bool(_require(snr, "verify_isclose")),
        atol=float(_require(snr, "atol")),
        rtol=float(_require(snr, "rtol")),
    )

    run_cfg = RunConfig(
        overwrite=bool(_require(run, "overwrite")),
        limit=int(_require(run, "limit")),
    )

    if audio_cfg.sample_rate_hz <= 0:
        raise ValueError("audio.sample_rate_hz must be > 0")
    if not rev_cfg.subsets:
        raise ValueError("reverberant_librichime5.subsets must be non-empty")
    if audio_cfg.clipping_threshold <= 0:
        raise ValueError("audio.clipping.threshold must be > 0")
    if audio_cfg.clipping_max_amp <= 0 or audio_cfg.clipping_max_amp > audio_cfg.clipping_threshold:
        raise ValueError("audio.clipping.max_amp must be in (0, threshold]")

    return chime5_cfg, datasets_cfg, rev_cfg, audio_cfg, snr_cfg, run_cfg


def _compute_loudness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x - float(np.mean(x))
    e = float(np.sum(np.square(x)))
    if e <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(e)


def _read_mono(path: Path, expected_sr: int, channel_index_if_multi: int) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if sr != expected_sr:
        raise RuntimeError(f"Sample rate mismatch: {path} {sr} != {expected_sr}")
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        return x
    if x.ndim == 2:
        if x.shape[1] == 1:
            return x[:, 0]
        if channel_index_if_multi < 0 or channel_index_if_multi >= x.shape[1]:
            raise ValueError(f"Invalid channel index {channel_index_if_multi} for {path} shape={x.shape}")
        return x[:, channel_index_if_multi]
    raise ValueError(f"Unsupported audio ndim={x.ndim} ({path})")


def _create_reverberant_speech(
    mix_infos: dict[str, Any],
    dtype: np.dtype,
    mix_len: int,
    sr: int,
    voicehome_dir: Path,
    librispeech_dir: Path,
) -> np.ndarray:
    speakers = [k for k in mix_infos.keys() if k.startswith("speaker_")]
    speech_sigs = np.zeros((mix_len, len(speakers)), dtype=dtype)

    for spk_ind, spk_key in enumerate(speakers):
        spk_infos = mix_infos[spk_key]
        utterances = spk_infos["utterances"]
        rir_infos = spk_infos["RIR"]
        rir_file = str(rir_infos["file"])
        rir_channel = int(rir_infos["channel"])

        rir_path = (voicehome_dir / rir_file).resolve()
        rir_sig, rir_sr = sf.read(str(rir_path), dtype="float32", always_2d=False)
        if rir_sr != sr:
            raise RuntimeError(f"RIR sample rate mismatch: {rir_path} {rir_sr} != {sr}")
        rir_sig = np.asarray(rir_sig, dtype=np.float32)
        if rir_sig.ndim != 2:
            raise RuntimeError(f"Unexpected RIR shape: {rir_path} {rir_sig.shape}")
        if rir_channel < 0 or rir_channel >= rir_sig.shape[1]:
            raise RuntimeError(f"Invalid RIR channel: {rir_path} ch={rir_channel} shape={rir_sig.shape}")
        rir_mono = rir_sig[:, rir_channel]

        for utt in utterances:
            utt_file = str(utt["file"])
            start_librispeech = int(utt["start_librispeech"])
            end_librispeech = int(utt["end_librispeech"])
            start_mix = int(utt["start_mix"])
            end_mix = int(utt["end_mix"])

            utt_len = end_mix - start_mix
            if utt_len <= 0:
                continue

            speech_path = (librispeech_dir / utt_file).resolve()
            speech_sig, speech_sr = sf.read(str(speech_path), dtype="float32", always_2d=False)
            if speech_sr != sr:
                raise RuntimeError(f"Speech sample rate mismatch: {speech_path} {speech_sr} != {sr}")
            speech_sig = np.asarray(speech_sig, dtype=np.float32)
            if speech_sig.ndim != 1:
                raise RuntimeError(f"Expected mono LibriSpeech audio: {speech_path} shape={speech_sig.shape}")

            if start_librispeech < 0 or end_librispeech > speech_sig.shape[0] or start_librispeech >= end_librispeech:
                raise RuntimeError(f"Invalid LibriSpeech slice: {speech_path} [{start_librispeech}, {end_librispeech})")

            speech_sig_cut = speech_sig[start_librispeech:end_librispeech]
            rev_speech_sig = signal.fftconvolve(speech_sig_cut, rir_mono, mode="full").astype(np.float32, copy=False)

            if start_mix == 0 and end_mix == mix_len:
                rev_speech_sig = rev_speech_sig[:utt_len]
                speech_sigs[start_mix:end_mix, spk_ind] = rev_speech_sig.astype(dtype, copy=False)
            elif start_mix == 0 and end_mix != mix_len:
                rev_speech_sig = rev_speech_sig[-utt_len:]
                speech_sigs[start_mix:end_mix, spk_ind] = rev_speech_sig.astype(dtype, copy=False)
            elif start_mix != 0 and end_mix == mix_len:
                rev_speech_sig = rev_speech_sig[:utt_len]
                speech_sigs[start_mix:end_mix, spk_ind] = rev_speech_sig.astype(dtype, copy=False)
            else:
                rir_len = int(rir_mono.shape[0])
                new_end_mix = end_mix + rir_len - 1
                if new_end_mix <= mix_len:
                    speech_sigs[start_mix:new_end_mix, spk_ind] = rev_speech_sig.astype(dtype, copy=False)
                else:
                    speech_sigs[start_mix:mix_len, spk_ind] = rev_speech_sig[: mix_len - start_mix].astype(dtype, copy=False)

    return speech_sigs


def _run_chime5_extractor(cfg: Chime5ExtractorConfig, root_dir: Path) -> None:
    if not cfg.original_data_dir.exists():
        raise FileNotFoundError(f"CHiME-5 original_data_dir not found: {cfg.original_data_dir}")
    if not cfg.json_dir.exists():
        raise FileNotFoundError(f"CHiME-5 json_dir not found: {cfg.json_dir}")
    cfg.segments_output_dir.mkdir(parents=True, exist_ok=True)

    script_path = (root_dir / "benchmarks/ns_chime7_udase/third_party/udase_chime5/create_audio_segments.py").resolve()
    if not script_path.exists():
        raise FileNotFoundError(str(script_path))

    cmd = [
        "python3",
        str(script_path),
        str(cfg.original_data_dir),
        str(cfg.json_dir),
        str(cfg.segments_output_dir),
    ]
    if cfg.extract_stereo:
        cmd.append("--extract_stereo")
    if cfg.eval_only:
        cmd.append("--eval_only")

    subprocess.run(cmd, check=True)


def _load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    with metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Metadata must be a list: {metadata_path}")
    return data


def _maybe_skip(path: Path, overwrite: bool) -> bool:
    if path.exists():
        return not overwrite
    return False


def _generate_subset(
    subset: str,
    metadata_dir: Path,
    output_dir: Path,
    chime5_segments_dir: Path,
    librispeech_dir: Path,
    voicehome_dir: Path,
    audio_cfg: AudioConfig,
    snr_cfg: SnrConfig,
    run_cfg: RunConfig,
) -> dict[str, Any]:
    metadata_path = (metadata_dir / f"{subset}.json").resolve()
    dataset = _load_metadata(metadata_path)
    if run_cfg.limit and run_cfg.limit > 0:
        dataset = dataset[: run_cfg.limit]

    out_subset_dir = (output_dir / subset).resolve()
    out_subset_dir.mkdir(parents=True, exist_ok=True)

    wrote = 0
    skipped = 0

    for mix_infos in tqdm(dataset, total=len(dataset), desc=f"generate_{subset}"):
        mix_name = str(mix_infos["name"])
        mix_len = int(mix_infos["length"])
        mix_max_n_spk = int(mix_infos["max_num_sim_active_speakers"])
        speakers = [k for k in mix_infos.keys() if k.startswith("speaker_")]

        noise_infos = mix_infos["noise"]
        noise_subset = str(noise_infos.get("subset", subset))
        noise_file = str(noise_infos["filename"])
        noise_path = (chime5_segments_dir / noise_subset / "0" / f"{noise_file}.wav").resolve()

        if not noise_path.exists():
            raise FileNotFoundError(str(noise_path))

        noise_sig = _read_mono(noise_path, expected_sr=audio_cfg.sample_rate_hz, channel_index_if_multi=audio_cfg.chime5_noise_channel_index)
        if int(noise_sig.shape[0]) != mix_len:
            raise RuntimeError(f"Noise length mismatch: {noise_path} {noise_sig.shape[0]} != {mix_len}")

        noise_loudness = _compute_loudness(noise_sig)
        if math.isinf(noise_loudness):
            raise RuntimeError(f"Noise loudness invalid: {noise_path}")

        speech_sigs = _create_reverberant_speech(
            mix_infos=mix_infos,
            dtype=noise_sig.dtype,
            mix_len=mix_len,
            sr=audio_cfg.sample_rate_hz,
            voicehome_dir=voicehome_dir,
            librispeech_dir=librispeech_dir,
        )

        speech_mix_sig = np.zeros((mix_len,), dtype=noise_sig.dtype)
        for spk_ind, spk_key in enumerate(speakers):
            spk_infos = mix_infos[spk_key]
            snr_spk = float(spk_infos["SNR"])
            speech_sig = speech_sigs[:, spk_ind]

            speech_loudness = _compute_loudness(speech_sig)
            if math.isinf(speech_loudness):
                raise RuntimeError(f"Speech loudness invalid: {mix_name} {spk_key}")

            orig_snr = speech_loudness - noise_loudness
            speech_gain = 10.0 ** ((snr_spk - orig_snr) / 20.0)

            scaled_speech_sig = (speech_sig * speech_gain).astype(noise_sig.dtype, copy=False)

            if snr_cfg.verify_isclose:
                speech_loudness_new = _compute_loudness(scaled_speech_sig)
                new_snr = speech_loudness_new - noise_loudness
                if not math.isclose(new_snr, snr_spk, rel_tol=snr_cfg.rtol, abs_tol=snr_cfg.atol):
                    raise RuntimeError(f"SNR verify failed: {mix_name} {spk_key} expected={snr_spk} actual={new_snr}")

            speech_mix_sig += scaled_speech_sig

        mix_sig = noise_sig + speech_mix_sig

        max_abs_mix = float(np.max(np.abs(mix_sig))) if mix_sig.size else 0.0
        max_abs_speech = float(np.max(np.abs(speech_mix_sig))) if speech_mix_sig.size else 0.0

        if max_abs_mix > audio_cfg.clipping_threshold or max_abs_speech > audio_cfg.clipping_threshold:
            denom = max(max_abs_mix, max_abs_speech)
            if denom > 0:
                scale = audio_cfg.clipping_max_amp / denom
                mix_sig = (mix_sig * scale).astype(noise_sig.dtype, copy=False)
                speech_mix_sig = (speech_mix_sig * scale).astype(noise_sig.dtype, copy=False)
                noise_sig = (noise_sig * scale).astype(noise_sig.dtype, copy=False)

        group_dir = (out_subset_dir / str(mix_max_n_spk)).resolve()
        group_dir.mkdir(parents=True, exist_ok=True)

        out_mix = group_dir / f"{mix_name}_mix.wav"
        out_speech = group_dir / f"{mix_name}_speech.wav"
        out_noise = group_dir / f"{mix_name}_noise.wav"

        if _maybe_skip(out_mix, overwrite=run_cfg.overwrite) and _maybe_skip(out_speech, overwrite=run_cfg.overwrite) and _maybe_skip(out_noise, overwrite=run_cfg.overwrite):
            skipped += 1
            continue

        sf.write(str(out_mix), mix_sig, audio_cfg.sample_rate_hz, format=audio_cfg.output_format, subtype=audio_cfg.output_subtype)
        sf.write(str(out_speech), speech_mix_sig, audio_cfg.sample_rate_hz, format=audio_cfg.output_format, subtype=audio_cfg.output_subtype)
        sf.write(str(out_noise), noise_sig, audio_cfg.sample_rate_hz, format=audio_cfg.output_format, subtype=audio_cfg.output_subtype)
        wrote += 1

    return {
        "subset": subset,
        "metadata_path": str(metadata_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "mix_count": int(len(dataset)),
        "wrote": wrote,
        "skipped": skipped,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--skip-chime5-extract", action="store_true")
    args = ap.parse_args()

    root_dir = Path(args.root_dir).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = _load_yaml(cfg_path)

    chime5_cfg, datasets_cfg, rev_cfg, audio_cfg, snr_cfg, run_cfg = _parse_config(cfg, root_dir=root_dir)

    if not args.skip_chime5_extract:
        _run_chime5_extractor(chime5_cfg, root_dir=root_dir)

    if not datasets_cfg.librispeech_dir.exists():
        raise FileNotFoundError(f"LibriSpeech dir not found: {datasets_cfg.librispeech_dir}")
    if not (datasets_cfg.librispeech_dir / "dev-clean").exists() or not (datasets_cfg.librispeech_dir / "test-clean").exists():
        nested = datasets_cfg.librispeech_dir / "LibriSpeech"
        if (nested / "dev-clean").exists() and (nested / "test-clean").exists():
            datasets_cfg.librispeech_dir = nested
        else:
            raise FileNotFoundError(f"LibriSpeech must contain dev-clean/ and test-clean/: {datasets_cfg.librispeech_dir}")
    if not datasets_cfg.voicehome_dir.exists():
        raise FileNotFoundError(f"VoiceHome dir not found: {datasets_cfg.voicehome_dir}")
    if not (datasets_cfg.voicehome_dir / "audio" / "rirs").exists():
        raise FileNotFoundError(f"VoiceHome must contain audio/rirs/: {datasets_cfg.voicehome_dir}")

    rev_cfg.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    for subset in rev_cfg.subsets:
        results.append(
            _generate_subset(
                subset=subset,
                metadata_dir=rev_cfg.metadata_dir,
                output_dir=rev_cfg.output_dir,
                chime5_segments_dir=chime5_cfg.segments_output_dir,
                librispeech_dir=datasets_cfg.librispeech_dir,
                voicehome_dir=datasets_cfg.voicehome_dir,
                audio_cfg=audio_cfg,
                snr_cfg=snr_cfg,
                run_cfg=run_cfg,
            )
        )

    manifest = {
        "tool": "prepare_reverberant_librichime5",
        "created_at_utc": started,
        "config_path": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "chime5": {
            "original_data_dir": str(chime5_cfg.original_data_dir),
            "segments_output_dir": str(chime5_cfg.segments_output_dir),
            "json_dir": str(chime5_cfg.json_dir),
            "extractor": {"extract_stereo": chime5_cfg.extract_stereo, "eval_only": chime5_cfg.eval_only},
            "third_party_script": {
                "path": "benchmarks/ns_chime7_udase/third_party/udase_chime5/create_audio_segments.py",
                "sha256": _sha256_file(root_dir / "benchmarks/ns_chime7_udase/third_party/udase_chime5/create_audio_segments.py"),
            },
        },
        "datasets": {
            "librispeech_dir": str(datasets_cfg.librispeech_dir),
            "voicehome_dir": str(datasets_cfg.voicehome_dir),
        },
        "reverberant_librichime5": {
            "metadata_dir": str(rev_cfg.metadata_dir),
            "output_dir": str(rev_cfg.output_dir),
            "subsets": rev_cfg.subsets,
        },
        "results": results,
    }

    out_manifest = rev_cfg.output_dir / "_generation_manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
