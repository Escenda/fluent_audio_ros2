from collections import deque
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml


class _BackendError(Exception):
    pass


class _FakeLogger:
    def __init__(self) -> None:
        self.fatal_records: list[tuple[str, Exception]] = []

    def fatal(self, message: str, exc: Exception) -> None:
        self.fatal_records.append((message, exc))


class _FakeNode:
    def get_logger(self) -> _FakeLogger:
        return self._logger


class _FakeQoSProfile:
    def __init__(self, *, depth: int) -> None:
        self.depth = depth
        self.reliability: str | None = None
        self.history: str | None = None


class _FakeReliabilityPolicy:
    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"


class _FakeHistoryPolicy:
    KEEP_LAST = "keep_last"


class _FakeAudioFrame:
    pass


class _FakeTurnContext:
    pass


class _FakeTurnEnd:
    pass


class _FakeVadState:
    pass


class _FakeInferenceSession:
    def __init__(self, model_path: str, *, providers: list[str]) -> None:
        self.model_path = model_path
        self.providers = providers


class _FailingBackend:
    name = "failing"
    sample_rate = 16000
    min_samples = 1
    max_samples = 4
    model_path = Path("/tmp/failing.onnx")

    def detect(self, audio: np.ndarray) -> None:
        raise _BackendError("backend down")


def _install_turn_detector_import_fakes(
    monkeypatch: pytest.MonkeyPatch,
    shutdown_calls: list[bool],
) -> None:
    rclpy_module = ModuleType("rclpy")

    def shutdown() -> None:
        shutdown_calls.append(True)

    def init(args: list[str] | None = None) -> None:
        del args

    def spin(node: _FakeNode) -> None:
        del node

    rclpy_module.shutdown = shutdown
    rclpy_module.init = init
    rclpy_module.spin = spin

    node_module = ModuleType("rclpy.node")
    node_module.Node = _FakeNode

    qos_module = ModuleType("rclpy.qos")
    qos_module.HistoryPolicy = _FakeHistoryPolicy
    qos_module.QoSProfile = _FakeQoSProfile
    qos_module.ReliabilityPolicy = _FakeReliabilityPolicy

    fa_interfaces_module = ModuleType("fa_interfaces")
    fa_interfaces_msg_module = ModuleType("fa_interfaces.msg")
    fa_interfaces_msg_module.AudioFrame = _FakeAudioFrame
    fa_interfaces_msg_module.TurnContext = _FakeTurnContext
    fa_interfaces_msg_module.TurnEnd = _FakeTurnEnd
    fa_interfaces_msg_module.VadState = _FakeVadState

    ort_module = ModuleType("onnxruntime")

    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    ort_module.get_available_providers = get_available_providers
    ort_module.InferenceSession = _FakeInferenceSession

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.msg", fa_interfaces_msg_module)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)


def test_default_config_requires_explicit_model_and_provider() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["backend.name"] == "smart_turn_onnx"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""


def test_turn_detector_node_rejects_non_canonical_audio_frames() -> None:
    package_root = Path(__file__).parents[2]
    source = (
        package_root / "fa_turn_detector_py" / "turn_detector_node.py"
    ).read_text(encoding="utf-8")

    assert "_resample_linear" not in source
    assert "_to_mono" not in source
    assert "np.frombuffer(bytes(msg.data), dtype=np.int16)" not in source
    assert "AudioFrame channels must be 1" in source
    assert "AudioFrame source_id and stream_id are required" in source
    assert "AudioFrame layout must be interleaved" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame sample_rate must match backend sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


def test_turn_detector_backend_runtime_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_turn_detector_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_turn_detector_py.backends.smart_turn_onnx", None)
    sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)

    try:
        module = importlib.import_module("fa_turn_detector_py.turn_detector_node")
        node = module.FaTurnDetectorNode.__new__(module.FaTurnDetectorNode)
        node._logger = _FakeLogger()
        node.backend = _FailingBackend()
        node.audio_buffer = deque([0.1, 0.2], maxlen=10)

        with pytest.raises(_BackendError):
            node._detect_turn_end()

        assert shutdown_calls == [True]
        assert len(node._logger.fatal_records) == 1
        fatal_message, fatal_exception = node._logger.fatal_records[0]
        assert fatal_message == "Turn detection backend failed: %s"
        assert isinstance(fatal_exception, _BackendError)
        assert str(fatal_exception) == "backend down"
    finally:
        sys.modules.pop("fa_turn_detector_py.backends.smart_turn_onnx", None)
        sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)


def test_smart_turn_backend_rejects_out_of_range_audio() -> None:
    package_root = Path(__file__).parents[2]
    source = (
        package_root / "fa_turn_detector_py" / "backends" / "smart_turn_onnx.py"
    ).read_text(encoding="utf-8")

    assert "audio = audio / max_abs" not in source
    assert "audio samples must be normalized to [-1.0, 1.0]" in source
