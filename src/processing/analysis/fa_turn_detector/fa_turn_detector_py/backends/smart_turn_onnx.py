from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort

from fa_turn_detector_py.backends.base import TurnDetectionResult


class SmartTurnOnnxBackend:
    """Smart Turn v3 ONNX inference backend."""

    name = "smart_turn_onnx"
    sample_rate = 16000
    n_fft = 400
    hop_length = 160
    n_mels = 80
    n_frames = 800
    min_samples = sample_rate
    max_samples = hop_length * n_frames + n_fft

    def __init__(self, *, model_path: Path, threshold: float, execution_provider: str) -> None:
        if threshold < 0.0 or threshold > 1.0:
            raise RuntimeError("backend.threshold must be between 0.0 and 1.0")
        if not execution_provider.strip():
            raise RuntimeError("backend.execution_provider is required")
        available_providers = ort.get_available_providers()
        if execution_provider not in available_providers:
            raise RuntimeError(
                "ONNX Runtime execution provider is not available: "
                f"{execution_provider}; available={available_providers}"
            )
        self._threshold = float(threshold)
        self._validate_model_file(model_path)
        self._session = ort.InferenceSession(
            str(model_path),
            providers=[execution_provider],
        )
        self._mel_filters_cache = self._mel_filterbank()
        self.model_path = model_path

    def detect(self, audio: np.ndarray) -> TurnDetectionResult:
        if audio.size == 0:
            raise ValueError("audio is empty")

        mel = self._compute_mel_spectrogram(audio)
        n_frames = mel.shape[1]
        if n_frames < self.n_frames:
            pad_width = self.n_frames - n_frames
            mel = np.pad(
                mel,
                ((0, 0), (pad_width, 0)),
                mode="constant",
                constant_values=-4.0,
            )
        elif n_frames > self.n_frames:
            mel = mel[:, -self.n_frames :]

        input_features = mel[np.newaxis, :, :].astype(np.float32)
        outputs = self._session.run(None, {"input_features": input_features})
        logits = outputs[0][0, 0]
        probability = 1.0 / (1.0 + np.exp(-float(logits)))
        return TurnDetectionResult(
            probability=float(probability),
            is_end=bool(probability >= self._threshold),
        )

    @staticmethod
    def _validate_model_file(model_path: Path) -> None:
        if not model_path.is_file():
            raise RuntimeError(f"Smart Turn model not found: {model_path}")
        if model_path.stat().st_size < 1024:
            header = model_path.read_text(encoding="utf-8", errors="ignore")[:128]
            if header.startswith("version https://git-lfs.github.com/spec"):
                raise RuntimeError(
                    f"Smart Turn model is a Git LFS pointer, not an ONNX file: {model_path}"
                )
            raise RuntimeError(f"Smart Turn model file is too small: {model_path}")

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 1.0:
            audio = audio / max_abs

        padded = np.pad(audio, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
        n_frames = (len(padded) - self.n_fft) // self.hop_length + 1
        if n_frames <= 0:
            raise ValueError("audio is too short for STFT")

        shape = (n_frames, self.n_fft)
        strides = (padded.strides[0] * self.hop_length, padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
        windowed = frames * window
        stft_matrix = np.fft.rfft(windowed, axis=1).T
        magnitudes = np.abs(stft_matrix) ** 2
        mel = self._mel_filters_cache @ magnitudes
        log_mel = np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
        log_mel = (log_mel + 4.0) / 4.0
        return log_mel.astype(np.float32)

    def _mel_filterbank(self) -> np.ndarray:
        def hz_to_mel(hz: np.ndarray) -> np.ndarray:
            return 2595.0 * np.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel: np.ndarray) -> np.ndarray:
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        n_fft_bins = self.n_fft // 2 + 1
        fmin = 0.0
        fmax = self.sample_rate / 2.0
        mel_min = hz_to_mel(np.array([fmin]))[0]
        mel_max = hz_to_mel(np.array([fmax]))[0]
        mel_points = np.linspace(mel_min, mel_max, self.n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        fft_freqs = np.linspace(0, self.sample_rate / 2, n_fft_bins)
        filters = np.zeros((self.n_mels, n_fft_bins), dtype=np.float32)

        for i in range(self.n_mels):
            left_hz = hz_points[i]
            center_hz = hz_points[i + 1]
            right_hz = hz_points[i + 2]
            left_slope = (fft_freqs - left_hz) / (center_hz - left_hz + 1e-10)
            right_slope = (right_hz - fft_freqs) / (right_hz - center_hz + 1e-10)
            filters[i] = np.maximum(0, np.minimum(left_slope, right_slope))

        enorm = 2.0 / (hz_points[2 : self.n_mels + 2] - hz_points[: self.n_mels])
        filters *= enorm[:, np.newaxis]
        return filters
