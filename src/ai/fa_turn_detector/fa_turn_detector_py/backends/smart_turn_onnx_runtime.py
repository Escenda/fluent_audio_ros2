from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import onnxruntime as ort


class SmartTurnOnnxRuntime:
    """In-worker Smart Turn v3 ONNX inference runtime."""

    sample_rate = 16000
    n_fft = 400
    hop_length = 160
    n_mels = 80
    n_frames = 800

    def __init__(self, *, model_path: Path, execution_provider: str) -> None:
        available_providers = ort.get_available_providers()
        if execution_provider not in available_providers:
            raise RuntimeError(
                "ONNX Runtime execution provider is not available: "
                f"{execution_provider}; available={available_providers}"
            )
        self._session = ort.InferenceSession(
            str(model_path),
            providers=[execution_provider],
        )
        self._validate_session_contract()
        self._mel_filters_cache = self._mel_filterbank()

    def detect_probability(self, audio: np.ndarray) -> float:
        self._validate_audio(audio)
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
        output = np.asarray(outputs[0], dtype=np.float32)
        if output.shape != (1, 1):
            raise RuntimeError(f"Smart Turn output shape must be (1, 1), got {output.shape}")
        logit = float(output[0, 0])
        if not math.isfinite(logit):
            raise RuntimeError("Smart Turn output logit must be finite")
        probability = 1.0 / (1.0 + np.exp(-logit))
        if not math.isfinite(probability):
            raise RuntimeError("Smart Turn probability must be finite")
        if probability < 0.0 or probability > 1.0:
            raise RuntimeError("Smart Turn probability must be in [0.0, 1.0]")
        return float(probability)

    def _validate_session_contract(self) -> None:
        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if len(inputs) != 1:
            raise RuntimeError(f"Smart Turn model must have 1 input, got {len(inputs)}")
        if len(outputs) != 1:
            raise RuntimeError(f"Smart Turn model must have 1 output, got {len(outputs)}")

        input_meta = inputs[0]
        if input_meta.name != "input_features":
            raise RuntimeError(
                "Smart Turn model input must be named input_features, "
                f"got {input_meta.name}"
            )
        if input_meta.type != "tensor(float)":
            raise RuntimeError(f"Smart Turn input type must be tensor(float), got {input_meta.type}")
        input_shape = tuple(input_meta.shape)
        if len(input_shape) != 3:
            raise RuntimeError(f"Smart Turn input rank must be 3, got {input_shape}")
        if input_shape[1] != self.n_mels:
            raise RuntimeError(
                f"Smart Turn input mel dimension must be {self.n_mels}, got {input_shape}"
            )
        if input_shape[2] != self.n_frames:
            raise RuntimeError(
                f"Smart Turn input frame dimension must be {self.n_frames}, got {input_shape}"
            )

        output_meta = outputs[0]
        if output_meta.type != "tensor(float)":
            raise RuntimeError(
                f"Smart Turn output type must be tensor(float), got {output_meta.type}"
            )
        output_shape = tuple(output_meta.shape)
        if len(output_shape) != 2:
            raise RuntimeError(f"Smart Turn output rank must be 2, got {output_shape}")
        if output_shape[1] != 1:
            raise RuntimeError(f"Smart Turn output second dimension must be 1, got {output_shape}")

    @staticmethod
    def _validate_audio(audio: np.ndarray) -> None:
        if audio.dtype != np.float32:
            raise RuntimeError("Smart Turn worker audio must be float32")
        if audio.ndim != 1:
            raise RuntimeError("Smart Turn worker audio must be one-dimensional")
        if audio.size == 0:
            raise RuntimeError("Smart Turn worker audio is required")
        if not np.all(np.isfinite(audio)):
            raise RuntimeError("Smart Turn worker audio contains non-finite samples")
        if np.any(audio < -1.0) or np.any(audio > 1.0):
            raise RuntimeError("Smart Turn worker audio samples must be normalized to [-1.0, 1.0]")

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        padded = np.pad(audio, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        window = np.hanning(self.n_fft + 1)[:-1].astype(np.float32)
        n_frames = (len(padded) - self.n_fft) // self.hop_length + 1
        if n_frames <= 0:
            raise RuntimeError("audio is too short for STFT")

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
