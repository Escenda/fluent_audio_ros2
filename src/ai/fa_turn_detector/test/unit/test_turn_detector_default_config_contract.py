from collections import deque
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml

from fa_turn_detector_py.backends.smart_turn_onnx import SmartTurnOnnxBackend


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


class _FakeParameter:
    class Type:
        STRING_ARRAY = "string_array"


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

    parameter_module = ModuleType("rclpy.parameter")
    parameter_module.Parameter = _FakeParameter

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

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.parameter", parameter_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.msg", fa_interfaces_msg_module)


def _write_fake_model(path: Path, *, probability: str = "0.75") -> None:
    path.write_text(probability + "\n" + ("x" * 2048), encoding="utf-8")


def _smart_turn_backend(tmp_path: Path, *, probability: str = "0.75") -> SmartTurnOnnxBackend:
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path, probability=probability)
    worker = Path(__file__).parents[1] / "fixtures" / "fake_turn_worker.py"
    return SmartTurnOnnxBackend(
        model_path=str(model_path),
        threshold=0.5,
        execution_provider="CPUExecutionProvider",
        command=sys.executable,
        args=(
            str(worker),
            "--audio",
            "{audio}",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
        ),
        health_args=(
            str(worker),
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


def _install_onnxruntime_fake(
    monkeypatch: pytest.MonkeyPatch,
    *,
    input_name: str = "input_features",
) -> None:
    ort_module = ModuleType("onnxruntime")

    class _FakeMeta:
        def __init__(
            self,
            *,
            name: str,
            type_name: str,
            shape: tuple[int | str | None, ...],
        ) -> None:
            self.name = name
            self.type = type_name
            self.shape = shape

    class _RuntimeInferenceSession:
        def __init__(self, model_path: str, *, providers: list[str]) -> None:
            self.model_path = model_path
            self.providers = providers

        def get_inputs(self) -> list[_FakeMeta]:
            return [
                _FakeMeta(
                    name=input_name,
                    type_name="tensor(float)",
                    shape=(1, 80, 800),
                )
            ]

        def get_outputs(self) -> list[_FakeMeta]:
            return [_FakeMeta(name="logits", type_name="tensor(float)", shape=(1, 1))]

        def run(
            self,
            output_names: None,
            feeds: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            del output_names, feeds
            return [np.array([[0.0]], dtype=np.float32)]

    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    ort_module.get_available_providers = get_available_providers
    ort_module.InferenceSession = _RuntimeInferenceSession
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_module)
    sys.modules.pop("fa_turn_detector_py.backends.smart_turn_onnx_runtime", None)


def test_default_config_requires_explicit_model_and_provider() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = (
        Path(__file__).parents[2] / "fa_turn_detector_py" / "turn_detector_node.py"
    ).read_text(encoding="utf-8")

    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["backend.name"] == "smart_turn_onnx"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert 'declare_parameter("backend.args", Parameter.Type.STRING_ARRAY)' in source
    assert 'declare_parameter("backend.health_args", Parameter.Type.STRING_ARRAY)' in source
    assert "tuple(str(item) for item in value)" not in source


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
    assert "AudioFrame encoding must be FLOAT32LE" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame data is required" in source
    assert "AudioFrame sample_rate must match backend sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


def test_frame_to_float_rejects_pcm32_payload_before_float_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_turn_detector_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)

    module = importlib.import_module("fa_turn_detector_py.turn_detector_node")
    msg = _FakeAudioFrame()
    msg.source_id = "mic"
    msg.stream_id = "audio/frame"
    msg.layout = "interleaved"
    msg.channels = 1
    msg.encoding = "PCM32LE"
    msg.bit_depth = 32
    msg.data = np.array([0.0, 0.5], dtype=np.float32).tobytes()

    with pytest.raises(ValueError, match="AudioFrame encoding must be FLOAT32LE"):
        module.FaTurnDetectorNode._frame_to_float(msg)


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
    assert "turn detector audio samples must be normalized to [-1.0, 1.0]" in source


def test_turn_detector_node_does_not_import_onnxruntime() -> None:
    package_root = Path(__file__).parents[2]
    node_source = (
        package_root / "fa_turn_detector_py" / "turn_detector_node.py"
    ).read_text(encoding="utf-8")
    backend_source = (
        package_root / "fa_turn_detector_py" / "backends" / "smart_turn_onnx.py"
    ).read_text(encoding="utf-8")
    init_source = (
        package_root / "fa_turn_detector_py" / "backends" / "__init__.py"
    ).read_text(encoding="utf-8")

    assert "onnxruntime" not in node_source
    assert "onnxruntime" not in backend_source
    assert "onnxruntime" not in init_source
    assert "subprocess.run" in backend_source


def test_smart_turn_backend_external_worker_contract(tmp_path: Path) -> None:
    backend = _smart_turn_backend(tmp_path)
    audio = np.full(backend.sample_rate, 0.1, dtype=np.float32)

    result = backend.detect(audio)

    assert result.probability == 0.75
    assert result.is_end is True
    assert list((tmp_path / "workspace").iterdir()) == []


def test_smart_turn_backend_rejects_invalid_worker_probability(tmp_path: Path) -> None:
    backend = _smart_turn_backend(tmp_path, probability="nan")
    audio = np.full(backend.sample_rate, 0.1, dtype=np.float32)

    with pytest.raises(RuntimeError, match="probability must be finite"):
        backend.detect(audio)


def test_smart_turn_backend_rejects_missing_command_args(tmp_path: Path) -> None:
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path)
    worker = Path(__file__).parents[1] / "fixtures" / "fake_turn_worker.py"

    with pytest.raises(RuntimeError, match="backend.args must include placeholders"):
        SmartTurnOnnxBackend(
            model_path=str(model_path),
            threshold=0.5,
            execution_provider="CPUExecutionProvider",
            command=sys.executable,
            args=(str(worker), "--model", "{model}", "--provider", "{provider}"),
            health_args=(
                str(worker),
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


def test_smart_turn_backend_rejects_unknown_arg_placeholder(tmp_path: Path) -> None:
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path)
    worker = Path(__file__).parents[1] / "fixtures" / "fake_turn_worker.py"

    with pytest.raises(RuntimeError, match="unsupported backend.args placeholder: threshold"):
        SmartTurnOnnxBackend(
            model_path=str(model_path),
            threshold=0.5,
            execution_provider="CPUExecutionProvider",
            command=sys.executable,
            args=(
                str(worker),
                "--audio",
                "{audio}",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
                "{threshold}",
            ),
            health_args=(
                str(worker),
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


def test_smart_turn_backend_health_check_failure_is_startup_failure(tmp_path: Path) -> None:
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path, probability="healthfail")
    worker = Path(__file__).parents[1] / "fixtures" / "fake_turn_worker.py"

    with pytest.raises(RuntimeError, match="health check failed"):
        SmartTurnOnnxBackend(
            model_path=str(model_path),
            threshold=0.5,
            execution_provider="CPUExecutionProvider",
            command=sys.executable,
            args=(
                str(worker),
                "--audio",
                "{audio}",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
            ),
            health_args=(
                str(worker),
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


def test_smart_turn_runtime_validates_model_io_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_onnxruntime_fake(monkeypatch)
    runtime_module = importlib.import_module(
        "fa_turn_detector_py.backends.smart_turn_onnx_runtime"
    )
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path)

    runtime = runtime_module.SmartTurnOnnxRuntime(
        model_path=model_path,
        execution_provider="CPUExecutionProvider",
    )
    probability = runtime.detect_probability(np.full(16000, 0.1, dtype=np.float32))

    assert probability == 0.5


def test_smart_turn_runtime_rejects_wrong_input_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_onnxruntime_fake(monkeypatch, input_name="wrong_input")
    runtime_module = importlib.import_module(
        "fa_turn_detector_py.backends.smart_turn_onnx_runtime"
    )
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path)

    with pytest.raises(RuntimeError, match="input must be named input_features"):
        runtime_module.SmartTurnOnnxRuntime(
            model_path=model_path,
            execution_provider="CPUExecutionProvider",
        )
