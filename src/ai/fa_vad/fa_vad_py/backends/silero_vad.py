from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from fa_vad_py.backends.base import VadBackendSettings


_PROVIDER_ALIASES = {
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
}


class SileroVadOnnxBackend:
    """ROS-free Silero VAD ONNX probability backend."""

    name = "silero_vad"

    def __init__(self, settings: VadBackendSettings) -> None:
        if settings.sample_rate != 16000:
            raise RuntimeError("Silero VAD backend currently requires sample_rate=16000")
        model_path = Path(settings.model_path).expanduser()
        if not model_path.is_file():
            raise RuntimeError(f"Silero VAD model_path does not exist: {model_path}")
        if settings.inter_op_num_threads <= 0:
            raise RuntimeError("backend.inter_op_num_threads must be > 0")
        if settings.intra_op_num_threads <= 0:
            raise RuntimeError("backend.intra_op_num_threads must be > 0")

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("onnxruntime is required for Silero VAD ONNX backend") from exc

        session_options = ort.SessionOptions()
        session_options.inter_op_num_threads = settings.inter_op_num_threads
        session_options.intra_op_num_threads = settings.intra_op_num_threads
        provider = self._execution_provider(settings.execution_provider)
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=[provider],
        )
        self.sample_rate = settings.sample_rate
        self.window_size_samples = 512
        self.context_size_samples = 64
        self._configure_io()
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros(self._state_shape, dtype=np.float32)
        self._context = np.zeros((1, self.context_size_samples), dtype=np.float32)

    def predict_probability(self, samples: np.ndarray) -> float:
        chunk = self._validate_samples(samples)
        audio = np.concatenate((self._context, chunk.reshape(1, -1)), axis=1)
        outputs = self.session.run(
            None,
            {
                self._audio_input_name: audio.astype(np.float32, copy=False),
                self._state_input_name: self._state,
                self._sample_rate_input_name: np.array(self.sample_rate, dtype=np.int64),
            },
        )
        probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        self._state = np.asarray(outputs[1], dtype=np.float32)
        self._context = audio[:, -self.context_size_samples :]
        return max(0.0, min(1.0, probability))

    def _configure_io(self) -> None:
        inputs = {item.name: item for item in self.session.get_inputs()}
        self._audio_input_name = self._required_input(inputs, "input")
        self._state_input_name = self._required_input(inputs, "state")
        self._sample_rate_input_name = self._required_input(inputs, "sr")
        state_shape = inputs[self._state_input_name].shape
        if len(state_shape) != 3:
            raise RuntimeError("Silero VAD state input must be rank 3")
        self._state_shape = tuple(self._shape_dim(dim, default=1) for dim in state_shape)
        if self._state_shape[0] != 2:
            raise RuntimeError("Silero VAD state first dimension must be 2")

    @staticmethod
    def _execution_provider(value: str) -> str:
        provider = value.strip()
        if not provider:
            raise RuntimeError("backend.execution_provider is required")
        return _PROVIDER_ALIASES.get(provider.lower(), provider)

    @staticmethod
    def _required_input(inputs: dict[str, Any], name: str) -> str:
        if name not in inputs:
            available = ", ".join(sorted(inputs))
            raise RuntimeError(f"Silero VAD ONNX input '{name}' is required; got {available}")
        return name

    @staticmethod
    def _shape_dim(value: Any, *, default: int) -> int:
        if isinstance(value, int) and value > 0:
            return value
        return default

    def _validate_samples(self, samples: np.ndarray) -> np.ndarray:
        if not isinstance(samples, np.ndarray):
            raise TypeError("samples must be a numpy.ndarray")
        if samples.ndim != 1:
            raise ValueError("samples must be mono 1-D float32 PCM")
        if samples.size != self.window_size_samples:
            raise ValueError(
                "Silero VAD expects exactly "
                f"{self.window_size_samples} samples per inference window"
            )
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples contain non-finite values")
        if np.any(samples < -1.0) or np.any(samples > 1.0):
            raise ValueError("samples must be normalized to [-1.0, 1.0]")
        return samples
