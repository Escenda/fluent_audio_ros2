from pathlib import Path
import sys

import numpy as np

from fa_turn_detector_py.backends.smart_turn_onnx import SmartTurnOnnxBackend


PACKAGE_ROOT = Path(__file__).parents[2]


def _write_fake_model(path: Path, probability: str) -> None:
    path.write_text(probability + "\n" + ("x" * 2048), encoding="utf-8")


def test_smart_turn_backend_uses_external_worker_payload_contract(tmp_path: Path) -> None:
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path, "0.75")
    worker_path = PACKAGE_ROOT / "test" / "fixtures" / "fake_turn_worker.py"
    backend = SmartTurnOnnxBackend(
        model_path=str(model_path),
        threshold=0.5,
        execution_provider="CPUExecutionProvider",
        command=sys.executable,
        args=(
            str(worker_path),
            "--audio",
            "{audio}",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
        ),
        health_args=(
            str(worker_path),
            "--model",
            "{model}",
            "--provider",
            "{provider}",
            "--health-check",
        ),
        timeout_sec=1.0,
        workspace_dir=str(tmp_path / "workspace"),
        cleanup_audio_files=True,
    )

    result = backend.detect(np.zeros(backend.min_samples, dtype=np.float32))

    assert result.probability == 0.75
    assert result.is_end is True
    assert not tuple((tmp_path / "workspace").glob("*_turn_audio.npy"))


def test_smart_turn_ros_adapter_does_not_import_onnxruntime_or_ros_messages() -> None:
    adapter_text = (
        PACKAGE_ROOT / "fa_turn_detector_py" / "backends" / "smart_turn_onnx.py"
    ).read_text(encoding="utf-8")
    worker_text = (
        PACKAGE_ROOT
        / "fa_turn_detector_py"
        / "backends"
        / "smart_turn_onnx_worker.py"
    ).read_text(encoding="utf-8")
    runtime_text = (
        PACKAGE_ROOT
        / "fa_turn_detector_py"
        / "backends"
        / "smart_turn_onnx_runtime.py"
    ).read_text(encoding="utf-8")

    assert "onnxruntime" not in adapter_text
    assert "onnxruntime" not in worker_text
    assert "onnxruntime" in runtime_text
    forbidden_ros_tokens = (
        "rclpy",
        "fa_interfaces",
        "AudioFrame",
        "VadState",
        "TurnContext",
        "TurnEnd",
    )
    backend_files = tuple((PACKAGE_ROOT / "fa_turn_detector_py" / "backends").glob("*.py"))
    assert backend_files
    for backend_file in backend_files:
        source = backend_file.read_text(encoding="utf-8")
        for token in forbidden_ros_tokens:
            assert token not in source
