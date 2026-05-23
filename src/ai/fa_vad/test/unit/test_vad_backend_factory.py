from __future__ import annotations

import pytest

from fa_vad_py.backends.base import VadBackendSettings
from fa_vad_py.backends.factory import build_vad_backend
from fa_vad_py.backends.silero_vad import SileroVadOnnxBackend


def test_unknown_backend_name_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="unsupported VAD backend"):
        build_vad_backend(
            VadBackendSettings(
                name="",
                model_path="/missing/model.onnx",
                sample_rate=16000,
                execution_provider="CPUExecutionProvider",
                inter_op_num_threads=1,
                intra_op_num_threads=1,
            )
        )


def test_silero_provider_aliases_match_container_env_values() -> None:
    assert SileroVadOnnxBackend._execution_provider("cpu") == "CPUExecutionProvider"
    assert SileroVadOnnxBackend._execution_provider("cuda") == "CUDAExecutionProvider"
    assert (
        SileroVadOnnxBackend._execution_provider("CPUExecutionProvider")
        == "CPUExecutionProvider"
    )
