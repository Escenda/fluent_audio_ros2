from collections import deque
import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml

from fa_turn_detector_py.backends.smart_turn_onnx import SmartTurnOnnxBackend
from fa_turn_detector_py.backends.factory import (
    TurnDetectorBackendSettings,
    build_turn_detector_backend,
)


class _BackendError(Exception):
    pass


class _FakeParameterUninitializedException(Exception):
    pass


class _FakeLogger:
    def __init__(self) -> None:
        self.fatal_records: list[str] = []
        self.error_records: list[str] = []

    def fatal(self, message: str) -> None:
        self.fatal_records.append(message)

    def error(self, message: str) -> None:
        self.error_records.append(message)


class _FakeNode:
    def get_logger(self) -> _FakeLogger:
        return self._logger


class _FakeParameter:
    class Type:
        STRING = "string"
        BOOL = "bool"
        INTEGER = "integer"
        DOUBLE = "double"
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


class _FakeParameterValue:
    def __init__(self, string_array_value: tuple[str, ...]) -> None:
        self.string_array_value = string_array_value


class _TypedParameter:
    def __init__(self, type_value: str, value: str | bool | float | int | tuple[str, ...]) -> None:
        self.type_ = type_value
        self.value = value

    def get_parameter_value(self) -> _FakeParameterValue:
        if self.type_ != _FakeParameter.Type.STRING_ARRAY:
            raise RuntimeError("test parameter is not a string array")
        if not isinstance(self.value, tuple):
            raise RuntimeError("test string array parameter value must be tuple")
        return _FakeParameterValue(self.value)


class _TypedNode:
    def __init__(self, parameter: _TypedParameter) -> None:
        self._parameter = parameter

    def get_parameter(self, name: str) -> _TypedParameter:
        del name
        return self._parameter


class _ParameterMapNode:
    def __init__(self, parameters: dict[str, _TypedParameter]) -> None:
        self._parameters = parameters

    def get_parameter(self, name: str) -> _TypedParameter:
        return self._parameters[name]


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

    exceptions_module = ModuleType("rclpy.exceptions")
    exceptions_module.ParameterUninitializedException = _FakeParameterUninitializedException

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
    monkeypatch.setitem(sys.modules, "rclpy.exceptions", exceptions_module)
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
            "detect",
            "--audio",
            "{audio}",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
        ),
        health_args=(
            str(worker),
            "health",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
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


def test_default_config_requires_explicit_backend_model_and_provider() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["audio_topic"] == "audio/frame"
    assert params["expected_stream_id"] == "audio/raw/mic"
    assert params["audio_topic"] != params["expected_stream_id"]
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert params["expected_source_id"] == ""
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] == 10
    assert params["vad.qos.reliable"] is False
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
    assert params["output.qos.depth"] == 10
    assert params["output.qos.reliable"] is True


def test_turn_detector_backend_factory_rejects_missing_and_unknown_backend() -> None:
    settings = TurnDetectorBackendSettings(
        name="",
        model_path="",
        threshold=0.5,
        execution_provider="",
        command="",
        args=(),
        health_args=(),
        timeout_sec=1.0,
        workspace_dir="/tmp/fa_turn_detector_test",
        cleanup_audio_files=True,
    )

    with pytest.raises(RuntimeError, match="backend.name is required"):
        build_turn_detector_backend(settings)

    unknown_backend_settings = TurnDetectorBackendSettings(
        name="bogus",
        model_path="",
        threshold=0.5,
        execution_provider="",
        command="",
        args=(),
        health_args=(),
        timeout_sec=1.0,
        workspace_dir="/tmp/fa_turn_detector_test",
        cleanup_audio_files=True,
    )
    with pytest.raises(RuntimeError, match="unsupported turn detector backend.name: bogus"):
        build_turn_detector_backend(unknown_backend_settings)


def test_smart_turn_backend_rejects_unsupported_execution_provider_before_worker(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path)

    with pytest.raises(
        RuntimeError,
        match=(
            "backend.execution_provider must be one of: "
            "CPUExecutionProvider, CUDAExecutionProvider, TensorrtExecutionProvider; got BogusProvider"
        ),
    ):
        SmartTurnOnnxBackend(
            model_path=str(model_path),
            threshold=0.5,
            execution_provider="BogusProvider",
            command=sys.executable,
            args=(
                "detect",
                "--audio",
                "{audio}",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
            ),
            health_args=("health", "--model", "{model}", "--provider", "{provider}"),
            timeout_sec=1.0,
            workspace_dir=str(tmp_path / "workspace"),
            cleanup_audio_files=True,
        )


def test_smart_turn_backend_rejects_missing_execution_provider() -> None:
    with pytest.raises(RuntimeError, match="backend.execution_provider is required"):
        SmartTurnOnnxBackend._validate_execution_provider("")


def test_smart_turn_backend_rejects_missing_or_invalid_model_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        SmartTurnOnnxBackend._validate_model_file("")

    with pytest.raises(RuntimeError, match="Smart Turn model not found"):
        SmartTurnOnnxBackend._validate_model_file(str(tmp_path / "missing.onnx"))

    too_small_path = tmp_path / "too-small.onnx"
    too_small_path.write_bytes(b"onnx")
    with pytest.raises(RuntimeError, match="Smart Turn model file is too small"):
        SmartTurnOnnxBackend._validate_model_file(str(too_small_path))

    lfs_pointer_path = tmp_path / "lfs-pointer.onnx"
    lfs_pointer_path.write_text(
        "version https://git-lfs.github.com/spec/v1\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        SmartTurnOnnxBackend._validate_model_file(str(lfs_pointer_path))


def test_smart_turn_backend_rejects_missing_or_invalid_command(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.command is required"):
        SmartTurnOnnxBackend._validate_command("")

    with pytest.raises(RuntimeError, match="backend.command not found on PATH"):
        SmartTurnOnnxBackend._validate_command("fa-turn-detector-missing-worker")

    worker_path = tmp_path / "worker.py"
    worker_path.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    worker_path.chmod(0o644)
    with pytest.raises(RuntimeError, match="backend.command is not executable"):
        SmartTurnOnnxBackend._validate_command(str(worker_path))


def test_smart_turn_backend_rejects_invalid_arg_format_contract() -> None:
    allowed_fields = frozenset(("audio", "model", "provider"))
    required_fields = frozenset(("audio", "model", "provider"))

    with pytest.raises(RuntimeError, match="backend.args must be a tuple of non-empty strings"):
        SmartTurnOnnxBackend._validate_args(
            args=["{audio}", "{model}", "{provider}"],
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )
    with pytest.raises(RuntimeError, match="backend.args must be a tuple of non-empty strings"):
        SmartTurnOnnxBackend._validate_args(
            args=("{audio}", 3, "{provider}"),
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )
    with pytest.raises(RuntimeError, match="backend.args must be a tuple of non-empty strings"):
        SmartTurnOnnxBackend._validate_args(
            args=("{audio}", "", "{provider}"),
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )
    with pytest.raises(RuntimeError, match="backend.args must not be empty"):
        SmartTurnOnnxBackend._validate_args(
            args=(),
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )
    with pytest.raises(RuntimeError, match="backend.args contains malformed format string"):
        SmartTurnOnnxBackend._validate_args(
            args=("{audio", "{model}", "{provider}"),
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )
    with pytest.raises(RuntimeError, match="placeholders must not use conversion or format spec"):
        SmartTurnOnnxBackend._validate_args(
            args=("{audio!r}", "{model}", "{provider}"),
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )
    with pytest.raises(RuntimeError, match="placeholders must not use conversion or format spec"):
        SmartTurnOnnxBackend._validate_args(
            args=("{audio:>8}", "{model}", "{provider}"),
            allowed_fields=allowed_fields,
            required_fields=required_fields,
            field_label="backend.args",
        )


def test_turn_detector_node_parameter_helpers_reject_wrong_ros_parameter_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_turn_detector_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)

    try:
        module = importlib.import_module("fa_turn_detector_py.turn_detector_node")

        assert module.FaTurnDetectorNode._string_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "audio/frame")),
            "audio_topic",
        ) == "audio/frame"
        assert module.FaTurnDetectorNode._bool_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.BOOL, False)),
            "backend.cleanup_audio_files",
        ) is False
        assert module.FaTurnDetectorNode._integer_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 10)),
            "audio.qos.depth",
        ) == 10
        assert module.FaTurnDetectorNode._positive_integer_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 10)),
            "audio.qos.depth",
        ) == 10
        assert module.FaTurnDetectorNode._double_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 0.5)),
            "backend.threshold",
        ) == 0.5
        assert module.FaTurnDetectorNode._string_tuple_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING_ARRAY, ("--audio", "{audio}"))),
            "backend.args",
        ) == ("--audio", "{audio}")
        qos = module.FaTurnDetectorNode._qos_profile(
            _ParameterMapNode(
                {
                    "audio.qos.depth": _TypedParameter(_FakeParameter.Type.INTEGER, 10),
                    "audio.qos.reliable": _TypedParameter(_FakeParameter.Type.BOOL, False),
                }
            ),
            depth_parameter="audio.qos.depth",
            reliable_parameter="audio.qos.reliable",
        )
        assert qos.depth == 10
        assert qos.history == _FakeHistoryPolicy.KEEP_LAST
        assert qos.reliability == _FakeReliabilityPolicy.BEST_EFFORT

        with pytest.raises(RuntimeError, match="audio_topic must be a string"):
            module.FaTurnDetectorNode._string_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 1.0)),
                "audio_topic",
            )
        with pytest.raises(RuntimeError, match="backend.cleanup_audio_files must be a bool"):
            module.FaTurnDetectorNode._bool_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "false")),
                "backend.cleanup_audio_files",
            )
        with pytest.raises(RuntimeError, match="audio.qos.depth must be an integer"):
            module.FaTurnDetectorNode._integer_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "10")),
                "audio.qos.depth",
            )
        with pytest.raises(RuntimeError, match="audio.qos.depth must be greater than zero"):
            module.FaTurnDetectorNode._positive_integer_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 0)),
                "audio.qos.depth",
            )
        with pytest.raises(RuntimeError, match="audio.qos.reliable must be a bool"):
            module.FaTurnDetectorNode._qos_profile(
                _ParameterMapNode(
                    {
                        "audio.qos.depth": _TypedParameter(_FakeParameter.Type.INTEGER, 10),
                        "audio.qos.reliable": _TypedParameter(
                            _FakeParameter.Type.STRING,
                            "false",
                        ),
                    }
                ),
                depth_parameter="audio.qos.depth",
                reliable_parameter="audio.qos.reliable",
            )
        with pytest.raises(RuntimeError, match="backend.threshold must be a double"):
            module.FaTurnDetectorNode._double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "0.5")),
                "backend.threshold",
            )
        with pytest.raises(RuntimeError, match="backend.timeout_sec must be a double"):
            module.FaTurnDetectorNode._double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 5)),
                "backend.timeout_sec",
            )
        with pytest.raises(RuntimeError, match="backend.args must be a string array"):
            module.FaTurnDetectorNode._string_tuple_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "--audio")),
                "backend.args",
            )
    finally:
        sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)


def test_turn_detector_rejects_unbound_vad_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_turn_detector_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)

    try:
        module = importlib.import_module("fa_turn_detector_py.turn_detector_node")
        msg = _FakeVadState()
        msg.source_id = "mic-a"
        msg.stream_id = "audio/raw/mic"

        module.FaTurnDetectorNode._validate_vad_identity(
            msg,
            expected_source_id="mic-a",
            expected_stream_id="audio/raw/mic",
        )

        with pytest.raises(ValueError, match="VadState source_id must match expected_source_id"):
            module.FaTurnDetectorNode._validate_vad_identity(
                msg,
                expected_source_id="mic-b",
                expected_stream_id="audio/raw/mic",
            )

        with pytest.raises(ValueError, match="VadState stream_id must match expected_stream_id"):
            module.FaTurnDetectorNode._validate_vad_identity(
                msg,
                expected_source_id="mic-a",
                expected_stream_id="audio/other",
            )
    finally:
        sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)


def test_turn_context_replacement_clears_audio_buffer_and_vad_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_turn_detector_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)

    try:
        module = importlib.import_module("fa_turn_detector_py.turn_detector_node")
        node = module.FaTurnDetectorNode.__new__(module.FaTurnDetectorNode)
        node.audio_buffer = deque([0.1, 0.2, 0.3], maxlen=10)
        node.is_speech = True
        node._active_session_id = "session-a"
        node._active_user_turn_id = 1
        node._context_active = True

        same_turn = _FakeTurnContext()
        same_turn.active = True
        same_turn.session_id = "session-a"
        same_turn.user_turn_id = 1
        node.on_turn_context(same_turn)

        assert list(node.audio_buffer) == [0.1, 0.2, 0.3]
        assert node.is_speech is True

        next_turn = _FakeTurnContext()
        next_turn.active = True
        next_turn.session_id = "session-a"
        next_turn.user_turn_id = 2
        node.on_turn_context(next_turn)

        assert list(node.audio_buffer) == []
        assert node.is_speech is False
        assert node._active_session_id == "session-a"
        assert node._active_user_turn_id == 2
        assert node._context_active is True
    finally:
        sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)


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
    msg.stream_id = "audio/raw/mic"
    msg.layout = "interleaved"
    msg.channels = 1
    msg.encoding = "PCM32LE"
    msg.bit_depth = 32
    msg.data = np.array([0.0, 0.5], dtype=np.float32).tobytes()

    with pytest.raises(ValueError, match="AudioFrame encoding must be FLOAT32LE"):
        module.FaTurnDetectorNode._frame_to_float(
            msg,
            expected_source_id="mic",
            expected_stream_id="audio/raw/mic",
        )


def test_frame_to_float_rejects_unbound_source_or_stream_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_turn_detector_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)

    module = importlib.import_module("fa_turn_detector_py.turn_detector_node")
    msg = _FakeAudioFrame()
    msg.source_id = "mic-a"
    msg.stream_id = "audio/raw/mic"
    msg.layout = "interleaved"
    msg.channels = 1
    msg.encoding = "FLOAT32LE"
    msg.bit_depth = 32
    msg.data = np.array([0.0, 0.5], dtype=np.float32).tobytes()

    with pytest.raises(ValueError, match="AudioFrame source_id must match expected_source_id"):
        module.FaTurnDetectorNode._frame_to_float(
            msg,
            expected_source_id="mic-b",
            expected_stream_id="audio/raw/mic",
        )

    with pytest.raises(ValueError, match="AudioFrame stream_id must match expected_stream_id"):
        module.FaTurnDetectorNode._frame_to_float(
            msg,
            expected_source_id="mic-a",
            expected_stream_id="audio/other",
        )


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
        assert node._logger.fatal_records[0] == "Turn detection backend failed: backend down"
    finally:
        sys.modules.pop("fa_turn_detector_py.backends.smart_turn_onnx", None)
        sys.modules.pop("fa_turn_detector_py.turn_detector_node", None)


def test_smart_turn_backend_config_validation_rejects_type_coercion() -> None:
    with pytest.raises(RuntimeError, match="backend.threshold must be a double"):
        SmartTurnOnnxBackend._validate_threshold("0.5")
    with pytest.raises(RuntimeError, match="backend.threshold must be a double"):
        SmartTurnOnnxBackend._validate_threshold(1)
    with pytest.raises(RuntimeError, match="backend.timeout_sec must be a double"):
        SmartTurnOnnxBackend._validate_timeout("5.0")
    with pytest.raises(RuntimeError, match="backend.timeout_sec must be a double"):
        SmartTurnOnnxBackend._validate_timeout(5)
    with pytest.raises(RuntimeError, match="backend.timeout_sec must be greater than zero"):
        SmartTurnOnnxBackend._validate_timeout(0.0)
    with pytest.raises(RuntimeError, match="backend.cleanup_audio_files must be a bool"):
        SmartTurnOnnxBackend._validate_cleanup_audio_files("false")
    with pytest.raises(RuntimeError, match="backend.workspace_dir is required"):
        SmartTurnOnnxBackend._validate_workspace_dir("")


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
                "health",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
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
                "detect",
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
                "health",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
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
                "detect",
                "--audio",
                "{audio}",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
            ),
            health_args=(
                str(worker),
                "health",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
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


def test_smart_turn_runtime_feature_window_contract(
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

    class _CaptureSession:
        def __init__(self) -> None:
            self.input_features: list[np.ndarray] = []

        def run(
            self,
            output_names: None,
            feeds: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            del output_names
            self.input_features.append(feeds["input_features"].copy())
            return [np.array([[0.0]], dtype=np.float32)]

    capture_session = _CaptureSession()
    runtime._session = capture_session

    short_mel = np.full((80, 3), 2.0, dtype=np.float32)
    monkeypatch.setattr(runtime, "_compute_mel_spectrogram", lambda audio: short_mel)
    runtime.detect_probability(np.full(16000, 0.1, dtype=np.float32))
    padded_features = capture_session.input_features[-1]

    assert padded_features.shape == (1, 80, 800)
    assert np.all(padded_features[:, :, :797] == -4.0)
    assert np.all(padded_features[:, :, 797:] == 2.0)

    long_mel = np.tile(np.arange(805, dtype=np.float32), (80, 1))
    monkeypatch.setattr(runtime, "_compute_mel_spectrogram", lambda audio: long_mel)
    runtime.detect_probability(np.full(16000, 0.1, dtype=np.float32))
    truncated_features = capture_session.input_features[-1]

    assert truncated_features.shape == (1, 80, 800)
    assert np.all(truncated_features[:, :, 0] == 5.0)
    assert np.all(truncated_features[:, :, -1] == 804.0)


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
