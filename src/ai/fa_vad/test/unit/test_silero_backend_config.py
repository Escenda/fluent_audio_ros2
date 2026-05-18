import importlib
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml

from fa_vad_py.backends.base import Float32MonoWindow
from fa_vad_py.backends.silero import SileroVAD
from fa_vad_py.backends.silero_worker import parse_args
from fa_vad_py.contracts import (
    audio_frame_to_float_samples,
    validate_node_config,
    validate_qos_depth,
)


DEFAULT_ARGS = (
    "--audio",
    "{audio}",
    "--model",
    "{model}",
    "--provider",
    "{provider}",
    "--sample-rate",
    "{sample_rate}",
)


class _BackendCrash(Exception):
    pass


class _FakeLogger:
    def __init__(self) -> None:
        self.fatal_records: list[str] = []

    def fatal(self, message: str) -> None:
        self.fatal_records.append(message)

    def error(self, message: str) -> None:
        del message


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


class _FakeBool:
    def __init__(self) -> None:
        self.data = False


class _FakeFloat32:
    def __init__(self) -> None:
        self.data = 0.0


class _FakeAudioFrame:
    def __init__(self, data: bytes) -> None:
        self.header = None
        self.data = data
        self.sample_rate = 16000
        self.source_id = "mic0"
        self.stream_id = "stream0"
        self.encoding = "FLOAT32LE"
        self.layout = "interleaved"
        self.channels = 1
        self.bit_depth = 32


class _FakeVadState:
    pass


class _FakeParameterValue:
    def __init__(self, string_array_value: tuple[str, ...]) -> None:
        self.string_array_value = string_array_value


class _TypedParameter:
    def __init__(self, type_value: str, value: str | bool | int | float | tuple[str, ...]) -> None:
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


class _FailingVadBackend:
    def update(self, window: Float32MonoWindow) -> tuple[float, bool, bool, bool]:
        del window
        raise _BackendCrash("vad backend down")


def _install_vad_node_import_fakes(
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
    qos_module.QoSProfile = _FakeQoSProfile
    qos_module.ReliabilityPolicy = _FakeReliabilityPolicy
    qos_module.HistoryPolicy = _FakeHistoryPolicy

    std_msgs_module = ModuleType("std_msgs")
    std_msgs_msg_module = ModuleType("std_msgs.msg")
    std_msgs_msg_module.Bool = _FakeBool
    std_msgs_msg_module.Float32 = _FakeFloat32

    fa_interfaces_module = ModuleType("fa_interfaces")
    fa_interfaces_msg_module = ModuleType("fa_interfaces.msg")
    fa_interfaces_msg_module.AudioFrame = _FakeAudioFrame
    fa_interfaces_msg_module.VadState = _FakeVadState

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.parameter", parameter_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "std_msgs", std_msgs_module)
    monkeypatch.setitem(sys.modules, "std_msgs.msg", std_msgs_msg_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.msg", fa_interfaces_msg_module)


def _silero_backend(
    *,
    model_path: str,
    execution_provider: str = "cpu",
    command: str = "python3",
    args: tuple[str, ...] = DEFAULT_ARGS,
    timeout_sec: float = 1.0,
    workspace_dir: str = "/tmp/fluent_audio_fa_vad_test",
    sample_rate: int = 16000,
    frame_ms: int = 20,
    threshold_start: float = 0.5,
    threshold_end: float = 0.1,
    hangover_ms: int = 300,
) -> SileroVAD:
    return SileroVAD(
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        threshold_start=threshold_start,
        threshold_end=threshold_end,
        hangover_ms=hangover_ms,
        model_path=model_path,
        execution_provider=execution_provider,
        command=command,
        args=args,
        timeout_sec=timeout_sec,
        workspace_dir=workspace_dir,
        cleanup_audio_files=True,
    )


def _float32_window(
    *, sample_rate: int = 16000, sample_count: int = 512
) -> Float32MonoWindow:
    return Float32MonoWindow(
        sample_rate=sample_rate,
        data=np.zeros(sample_count, dtype="<f4").tobytes(),
    )


def _write_raw_float32(path: Path, *, sample_count: int) -> None:
    path.write_bytes(np.zeros(sample_count, dtype="<f4").tobytes())


def _write_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)


def test_default_config_requires_explicit_silero_model_path() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = (Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py").read_text(
        encoding="utf-8"
    )

    params = config["fa_vad_node"]["ros__parameters"]

    assert params["backend.name"] == "silero"
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["qos.depth"] == 10
    assert params["qos.reliable"] is False
    assert "{audio}" in params["backend.args"]
    assert "{model}" in params["backend.args"]
    assert "{provider}" in params["backend.args"]
    assert "{sample_rate}" in params["backend.args"]
    assert "silero" not in params
    assert '"backend.args",' in source
    assert "Parameter.Type.STRING_ARRAY" in source
    assert "tuple(str(item) for item in self.get_parameter" not in source


def test_vad_node_parameter_helpers_reject_wrong_ros_parameter_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_vad_node_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_vad_py.vad_node", None)

    try:
        module = importlib.import_module("fa_vad_py.vad_node")

        assert module.FaVadNode._string_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "audio/frame")),
            "input_topic",
        ) == "audio/frame"
        assert module.FaVadNode._bool_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.BOOL, False)),
            "publish_probability",
        ) is False
        assert module.FaVadNode._integer_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 16000)),
            "target_sample_rate",
        ) == 16000
        assert module.FaVadNode._double_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 0.5)),
            "threshold_start",
        ) == 0.5
        assert module.FaVadNode._string_array_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING_ARRAY, ("--audio", "{audio}"))),
            "backend.args",
        ) == ("--audio", "{audio}")

        with pytest.raises(RuntimeError, match="publish_probability must be a bool"):
            module.FaVadNode._bool_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "false")),
                "publish_probability",
            )
        with pytest.raises(RuntimeError, match="target_sample_rate must be an integer"):
            module.FaVadNode._integer_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "16000")),
                "target_sample_rate",
            )
        with pytest.raises(RuntimeError, match="threshold_start must be a double"):
            module.FaVadNode._double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 1)),
                "threshold_start",
            )
    finally:
        sys.modules.pop("fa_vad_py.vad_node", None)


def test_vad_frame_contract_accepts_canonical_float32_mono() -> None:
    expected = np.array([-1.0, -0.25, 0.0, 0.5, 1.0], dtype=np.float32)

    samples = audio_frame_to_float_samples(
        data=expected.astype("<f4").tobytes(),
        source_id="mic0",
        stream_id="stream0",
        expected_stream_id="stream0",
        encoding="FLOAT32LE",
        layout="interleaved",
        channels=1,
        bit_depth=32,
    )

    np.testing.assert_allclose(samples, expected)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"data": b""}, "AudioFrame data is required"),
        ({"source_id": ""}, "AudioFrame source_id and stream_id are required"),
        ({"stream_id": ""}, "AudioFrame source_id and stream_id are required"),
        ({"stream_id": "wrong_stream"}, "AudioFrame stream_id must match input_topic"),
        ({"layout": "planar"}, "AudioFrame layout must be interleaved"),
        ({"channels": 2}, "AudioFrame channels must be 1"),
        ({"encoding": "PCM32LE"}, "AudioFrame encoding must be FLOAT32LE"),
        ({"bit_depth": 16}, "AudioFrame bit_depth must be 32"),
        ({"data": b"1"}, "AudioFrame float32 data length is not byte-aligned"),
        (
            {"data": np.array([np.nan], dtype=np.float32).tobytes()},
            "AudioFrame contains non-finite samples",
        ),
        (
            {"data": np.array([1.25], dtype=np.float32).tobytes()},
            "AudioFrame samples must be normalized to \\[-1.0, 1.0\\]",
        ),
    ),
)
def test_vad_frame_contract_rejects_non_canonical_audio(
    kwargs: dict[str, bytes | str | int],
    message: str,
) -> None:
    base_kwargs = {
        "data": np.array([0.0], dtype=np.float32).tobytes(),
        "source_id": "mic0",
        "stream_id": "stream0",
        "expected_stream_id": "stream0",
        "encoding": "FLOAT32LE",
        "layout": "interleaved",
        "channels": 1,
        "bit_depth": 32,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(ValueError, match=message):
        audio_frame_to_float_samples(**base_kwargs)


def test_vad_node_source_does_not_hide_format_conversion() -> None:
    source_path = Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "bool(self.get_parameter" not in source
    assert "int(self.get_parameter" not in source
    assert "float(self.get_parameter" not in source
    assert "str(self.get_parameter" not in source
    assert "_resample_linear" not in source
    assert "_convert_to_mono" not in source
    assert "np.clip" not in source
    assert "_float_to_pcm16" not in source
    assert "astype(np.int16)" not in source
    assert "AudioFrame sample_rate must match target_sample_rate" in source
    assert "expected_stream_id=self._input_topic" in source
    assert '"FA VAD (Silero): "' in source
    assert "Dropping invalid AudioFrame: %s" not in source
    assert "VAD backend failed: %s" not in source
    assert "Speech START (prob=%.2f)" not in source


def test_vad_node_backend_runtime_failure_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package_root = Path(__file__).parents[2]
    shutdown_calls: list[bool] = []
    _install_vad_node_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(package_root))
    sys.modules.pop("fa_vad_py.vad_node", None)

    try:
        module = importlib.import_module("fa_vad_py.vad_node")
        node = module.FaVadNode.__new__(module.FaVadNode)
        node._logger = _FakeLogger()
        node._target_sample_rate = 16000
        node._input_topic = "stream0"
        node._vad = _FailingVadBackend()

        frame = _FakeAudioFrame(np.array([0.1], dtype=np.float32).tobytes())
        with pytest.raises(_BackendCrash):
            node._on_audio_frame(frame)

        assert shutdown_calls == [True]
        assert len(node._logger.fatal_records) == 1
        assert node._logger.fatal_records[0] == "VAD backend failed: vad backend down"
    finally:
        sys.modules.pop("fa_vad_py.vad_node", None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"target_sample_rate": 0}, "target_sample_rate must be > 0"),
        ({"threshold_start": -0.1}, "threshold_start must be in \\[0.0, 1.0\\]"),
        ({"threshold_start": 1.1}, "threshold_start must be in \\[0.0, 1.0\\]"),
        ({"threshold_end": -0.1}, "threshold_end must be in \\[0.0, 1.0\\]"),
        ({"threshold_end": 1.1}, "threshold_end must be in \\[0.0, 1.0\\]"),
        ({"threshold_start": 0.2, "threshold_end": 0.8}, "threshold_end must be <= threshold_start"),
        ({"hangover_ms": 0}, "hangover_ms must be > 0"),
    ),
)
def test_vad_node_rejects_invalid_runtime_config_without_fallback(
    kwargs: dict[str, int | float],
    message: str,
) -> None:
    base_kwargs = {
        "target_sample_rate": 16000,
        "threshold_start": 0.5,
        "threshold_end": 0.1,
        "hangover_ms": 300,
    }
    base_kwargs.update(kwargs)

    with pytest.raises(RuntimeError, match=message):
        validate_node_config(**base_kwargs)


def test_vad_node_rejects_invalid_qos_depth_without_fallback() -> None:
    assert validate_qos_depth(1) == 1
    with pytest.raises(RuntimeError, match="qos.depth must be > 0"):
        validate_qos_depth(0)
    with pytest.raises(RuntimeError, match="qos.depth must be > 0"):
        validate_qos_depth(-1)


def test_silero_backend_rejects_missing_model_path() -> None:
    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        _silero_backend(model_path="")


def test_silero_backend_rejects_missing_model_path_directory() -> None:
    missing_repo = "/tmp/fluent_audio_missing_silero_repo"

    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        _silero_backend(model_path=missing_repo)


def test_silero_backend_rejects_missing_execution_provider(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.execution_provider is required"):
        _silero_backend(model_path=str(tmp_path), execution_provider="")


def test_silero_backend_rejects_unknown_execution_provider(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported backend.execution_provider"):
        _silero_backend(model_path=str(tmp_path), execution_provider="tpu")


def test_silero_backend_rejects_missing_command(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.command is required"):
        _silero_backend(model_path=str(tmp_path), command="")


def test_silero_backend_rejects_missing_command_placeholders(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.args must include placeholders"):
        _silero_backend(model_path=str(tmp_path), args=("--model", "{model}"))


def test_silero_backend_rejects_unsupported_arg_placeholder(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unsupported backend.args placeholder: device"):
        _silero_backend(
            model_path=str(tmp_path),
            args=DEFAULT_ARGS + ("{device}",),
        )


def test_silero_backend_rejects_malformed_arg_placeholder(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="backend.args contains malformed format string"):
        _silero_backend(
            model_path=str(tmp_path),
            args=DEFAULT_ARGS + ("{audio",),
        )


def test_silero_backend_rejects_non_finite_probability() -> None:
    with pytest.raises(RuntimeError, match="probability must be finite"):
        SileroVAD._parse_probability("nan")


def test_silero_backend_rejects_arg_format_spec(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="placeholders must not use conversion or format spec"):
        _silero_backend(
            model_path=str(tmp_path),
            args=DEFAULT_ARGS + ("{audio!r}",),
        )


def test_silero_backend_resolves_model_and_command_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    command_path = tmp_path / "worker"
    _write_executable(command_path)

    backend = _silero_backend(model_path=str(model_dir), command=str(command_path))

    assert backend._model_path == model_dir.resolve()
    assert backend._command == str(command_path.resolve())


def test_silero_backend_rejects_non_executable_command(tmp_path: Path) -> None:
    command_path = tmp_path / "worker"
    command_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.command is not executable"):
        _silero_backend(model_path=str(tmp_path), command=str(command_path))


def test_silero_backend_rejects_invalid_sample_rate(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="sample_rate must be 8000 or 16000"):
        SileroVAD(
            sample_rate=44100,
            frame_ms=20,
            hangover_ms=300,
            threshold_start=0.5,
            threshold_end=0.1,
            model_path=str(tmp_path),
            execution_provider="cpu",
            command="python3",
            args=DEFAULT_ARGS,
            timeout_sec=1.0,
            workspace_dir="/tmp/fluent_audio_fa_vad_test",
            cleanup_audio_files=True,
        )


def test_silero_backend_rejects_invalid_thresholds(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="threshold_end must be <= threshold_start"):
        _silero_backend(
            model_path=str(tmp_path),
            threshold_start=0.2,
            threshold_end=0.8,
        )


def test_silero_backend_rejects_invalid_hangover(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hangover_ms must be > 0"):
        _silero_backend(model_path=str(tmp_path), hangover_ms=0)


def test_silero_backend_rejects_implicit_hangover_rounding(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="hangover_ms must be >= frame_ms"):
        _silero_backend(model_path=str(tmp_path), hangover_ms=10, frame_ms=20)
    with pytest.raises(RuntimeError, match="hangover_ms must be divisible by frame_ms"):
        _silero_backend(model_path=str(tmp_path), hangover_ms=25, frame_ms=20)


def test_silero_backend_constructor_requires_explicit_vad_config() -> None:
    backend_path = Path(__file__).parents[2] / "fa_vad_py" / "backends" / "silero.py"
    source = backend_path.read_text(encoding="utf-8")

    assert "sample_rate: int = 16000" not in source
    assert "frame_ms: int = 20" not in source
    assert "hangover_ms: int = 250" not in source
    assert "threshold_start: float | None" not in source
    assert "threshold_end: float | None" not in source
    assert "VAD_THRESHOLD_START" not in source
    assert "VAD_THRESHOLD_END" not in source


def test_silero_backend_is_external_process_boundary() -> None:
    backend_path = Path(__file__).parents[2] / "fa_vad_py" / "backends" / "silero.py"
    node_path = Path(__file__).parents[2] / "fa_vad_py" / "vad_node.py"
    backend_source = backend_path.read_text(encoding="utf-8")
    node_source = node_path.read_text(encoding="utf-8")

    assert "import torch" not in backend_source
    assert "torch.hub" not in backend_source
    assert "subprocess.run" in backend_source
    assert 'declare_parameter("backend.command", "")' in node_source
    assert "VAD backend failed" in node_source


def test_silero_backend_runs_external_worker_contract(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    workspace_dir = tmp_path / "workspace"
    worker = Path(__file__).parents[1] / "fixtures" / "fake_vad_worker.py"
    backend = _silero_backend(
        model_path=str(model_dir),
        command=sys.executable,
        args=(
            str(worker),
            "--audio",
            "{audio}",
            "--model",
            "{model}",
            "--provider",
            "{provider}",
            "--sample-rate",
            "{sample_rate}",
        ),
        workspace_dir=str(workspace_dir),
    )

    result = backend.update(_float32_window())

    assert result is not None
    assert result.probability == 0.75
    assert result.is_speech is True
    assert result.start is True
    assert result.end is False
    assert list(workspace_dir.iterdir()) == []


def test_silero_backend_returns_explicit_no_decision_for_short_window(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend = _silero_backend(model_path=str(model_dir))

    assert backend.update(_float32_window(sample_count=128)) is None


def test_silero_backend_rejects_window_sample_rate_mismatch(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    backend = _silero_backend(model_path=str(model_dir), sample_rate=16000)

    with pytest.raises(ValueError, match="sample_rate must match backend sample_rate"):
        backend.update(_float32_window(sample_rate=8000, sample_count=256))


def test_float32_window_rejects_invalid_boundary_values() -> None:
    with pytest.raises(ValueError, match="sample_rate must be 8000 or 16000"):
        Float32MonoWindow(sample_rate=44100, data=np.zeros(1, dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="data is required"):
        Float32MonoWindow(sample_rate=16000, data=b"")
    with pytest.raises(ValueError, match="float32 byte-aligned"):
        Float32MonoWindow(sample_rate=16000, data=b"\x00")
    with pytest.raises(ValueError, match="non-finite"):
        Float32MonoWindow(sample_rate=16000, data=np.array([np.nan], dtype="<f4").tobytes())
    with pytest.raises(ValueError, match="normalized"):
        Float32MonoWindow(sample_rate=16000, data=np.array([1.5], dtype="<f4").tobytes())


def test_silero_worker_rejects_unsupported_sample_rate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    audio_path = tmp_path / "audio.f32"
    _write_raw_float32(audio_path, sample_count=512)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "silero_vad_worker",
            "--audio",
            str(audio_path),
            "--model",
            str(model_dir),
            "--provider",
            "cpu",
            "--sample-rate",
            "44100",
        ],
    )

    with pytest.raises(RuntimeError, match="sample-rate must be 8000 or 16000"):
        parse_args()


def test_silero_worker_is_installed_by_cmake() -> None:
    cmake_path = Path(__file__).parents[2] / "CMakeLists.txt"
    worker_path = Path(__file__).parents[2] / "scripts" / "silero_vad_worker"

    assert worker_path.is_file()
    assert "scripts/silero_vad_worker" in cmake_path.read_text(encoding="utf-8")
