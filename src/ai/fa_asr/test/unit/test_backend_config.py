import threading
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml

from fa_asr_py.backends.base import (
    AsrRequest,
    AsrTranscript,
    AsrTranscriptSegment,
    asr_transcript_text,
)
from fa_asr_py.backends.factory import AsrBackendSettings, build_asr_backend
from fa_asr_py.backends.local_command import LocalCommandAsrBackend, load_local_command_config
from fa_asr_py.backends.openai_realtime import (
    OpenAiRealtimeAsrBackend,
    load_openai_realtime_config,
)
from fa_asr_py.backends.openai_transcriptions import (
    OpenAiTranscriptionsAsrBackend,
    load_openai_transcriptions_config,
)
from fa_asr_py.backends.parakeet_worker import (
    ParakeetWorkerAsrBackend,
    load_parakeet_worker_config,
)
from fa_asr_py.backends.whisper_cpp import WhisperCppAsrBackend, load_whisper_cpp_config


PACKAGE_ROOT = Path(__file__).parents[2]


def _write_executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _write_openai_env_probe_worker(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import os
import sys
from pathlib import Path


def _flag_value(flag: str) -> str:
    for index, value in enumerate(sys.argv):
        if value == flag and index + 1 < len(sys.argv):
            return sys.argv[index + 1]
    raise RuntimeError(f"missing argv flag: {flag}")


expected_key = os.environ.get("EXPECTED_OPENAI_API_KEY_FOR_TEST", "")
if not expected_key:
    raise RuntimeError("EXPECTED_OPENAI_API_KEY_FOR_TEST is required")
if os.environ.get("OPENAI_API_KEY", "") != expected_key:
    raise RuntimeError("OPENAI_API_KEY was not forwarded to worker")
if any(expected_key in value for value in sys.argv):
    raise RuntimeError("OPENAI_API_KEY value leaked into argv")
if "--health-check" in sys.argv:
    if not _flag_value("--model"):
        raise RuntimeError("model is required")
    print("openai-health-ok")
    raise SystemExit(0)
audio_path = Path(_flag_value("--audio"))
if not audio_path.is_file() or not audio_path.read_bytes():
    raise RuntimeError("audio payload is required")
if not _flag_value("--model"):
    raise RuntimeError("model is required")
if int(_flag_value("--sample-rate")) <= 0:
    raise RuntimeError("sample rate must be positive")
print("openai-env-ok")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


def _write_health_fail_worker(path: Path) -> Path:
    path.write_text(
        """#!/usr/bin/env python3
import sys

if "--health-check" in sys.argv:
    print("health failed", file=sys.stderr)
    raise SystemExit(7)
print("transcript")
""",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)
    return path


class _FakeNode:
    def get_logger(self) -> "_FakeLogger":
        return self._logger


class _FakeLogger:
    def __init__(self) -> None:
        self.error_records: list[str] = []
        self.fatal_records: list[str] = []

    def error(self, message: str) -> None:
        self.error_records.append(message)

    def fatal(self, message: str) -> None:
        self.fatal_records.append(message)


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


class _FakeAsrResult:
    STATUS_FINAL = 1
    STATUS_TIMEOUT = 2
    STATUS_ERROR = 3


class _FakeAudioFrame:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.source_id = "mic0"
        self.stream_id = "stream0"
        self.layout = "interleaved"
        self.channels = 1
        self.encoding = "FLOAT32LE"
        self.bit_depth = 32
        self.sample_rate = 16000


class _FakeAudioModelRef:
    pass


class _FakeAudioWindowRef:
    pass


class _FakeResolvedTimeRange:
    CLOCK_AGENT = "agent"
    CLOCK_ROBOT = "robot"
    CLOCK_MEDIA = "media"


class _FakeTranscriptSegment:
    pass


class _FakeTurnContext:
    pass


class _FakeVadState:
    pass


class _FakeTime:
    pass


class _FakeMutuallyExclusiveCallbackGroup:
    pass


class _FakeMultiThreadedExecutor:
    instances: list["_FakeMultiThreadedExecutor"] = []

    def __init__(self, *, num_threads: int) -> None:
        self.num_threads = num_threads
        self.nodes: list[_FakeNode] = []
        self.spin_called = False
        self.shutdown_called = False
        _FakeMultiThreadedExecutor.instances.append(self)

    def add_node(self, node: _FakeNode) -> None:
        self.nodes.append(node)

    def spin(self) -> None:
        self.spin_called = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeTranscribeAudioResponse:
    ERROR_NONE = "none"
    ERROR_TIME_RANGE_UNRESOLVED = "time_range_unresolved"
    ERROR_WINDOW_NOT_FOUND = "window_not_found"
    ERROR_RANGE_OUTSIDE_WINDOW = "range_outside_window"
    ERROR_UNSUPPORTED_AUDIO_SCOPE = "unsupported_audio_scope"
    ERROR_TRANSCRIBE_FAILED = "transcribe_failed"


class _FakeTranscribeAudio:
    class Request:
        pass

    Response = _FakeTranscribeAudioResponse


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


class _ParameterMapNode:
    def __init__(self, parameters: dict[str, _TypedParameter]) -> None:
        self._parameters = parameters

    def get_parameter(self, name: str) -> _TypedParameter:
        return self._parameters[name]


class _BackendCrash(Exception):
    pass


class _FakeParameterUninitializedException(Exception):
    pass


class _FailingAsrBackend:
    name = "failing"

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        raise _BackendCrash("asr backend crashed")


class _TimeoutAsrBackend:
    name = "timeout"

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        del request
        raise TimeoutError("asr backend timed out")


class _SegmentAsrBackend:
    name = "segments"

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        del request
        return AsrTranscript(
            segments=(
                AsrTranscriptSegment(start_sample=0, end_sample=1, text="hello"),
                AsrTranscriptSegment(start_sample=1, end_sample=2, text="world"),
            )
        )


def _install_asr_node_import_fakes(
    monkeypatch: pytest.MonkeyPatch,
    shutdown_calls: list[bool] | None = None,
) -> None:
    rclpy_module = ModuleType("rclpy")

    def shutdown() -> None:
        if shutdown_calls is not None:
            shutdown_calls.append(True)

    def init(args: list[str] | None = None) -> None:
        del args

    def spin(node: _FakeNode) -> None:
        del node

    rclpy_module.shutdown = shutdown
    rclpy_module.init = init
    rclpy_module.spin = spin
    _FakeMultiThreadedExecutor.instances = []

    node_module = ModuleType("rclpy.node")
    node_module.Node = _FakeNode

    callback_groups_module = ModuleType("rclpy.callback_groups")
    callback_groups_module.MutuallyExclusiveCallbackGroup = _FakeMutuallyExclusiveCallbackGroup

    executors_module = ModuleType("rclpy.executors")
    executors_module.MultiThreadedExecutor = _FakeMultiThreadedExecutor

    parameter_module = ModuleType("rclpy.parameter")
    parameter_module.Parameter = _FakeParameter

    exceptions_module = ModuleType("rclpy.exceptions")
    exceptions_module.ParameterUninitializedException = _FakeParameterUninitializedException

    qos_module = ModuleType("rclpy.qos")
    qos_module.HistoryPolicy = _FakeHistoryPolicy
    qos_module.QoSProfile = _FakeQoSProfile
    qos_module.ReliabilityPolicy = _FakeReliabilityPolicy

    builtin_interfaces_module = ModuleType("builtin_interfaces")
    builtin_interfaces_msg_module = ModuleType("builtin_interfaces.msg")
    builtin_interfaces_msg_module.Time = _FakeTime

    fa_interfaces_module = ModuleType("fa_interfaces")
    fa_interfaces_msg_module = ModuleType("fa_interfaces.msg")
    fa_interfaces_msg_module.AsrResult = _FakeAsrResult
    fa_interfaces_msg_module.AudioFrame = _FakeAudioFrame
    fa_interfaces_msg_module.AudioModelRef = _FakeAudioModelRef
    fa_interfaces_msg_module.AudioWindowRef = _FakeAudioWindowRef
    fa_interfaces_msg_module.ResolvedTimeRange = _FakeResolvedTimeRange
    fa_interfaces_msg_module.TranscriptSegment = _FakeTranscriptSegment
    fa_interfaces_msg_module.TurnContext = _FakeTurnContext
    fa_interfaces_msg_module.VadState = _FakeVadState
    fa_interfaces_srv_module = ModuleType("fa_interfaces.srv")
    fa_interfaces_srv_module.TranscribeAudio = _FakeTranscribeAudio

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "builtin_interfaces", builtin_interfaces_module)
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", builtin_interfaces_msg_module)
    monkeypatch.setitem(sys.modules, "rclpy.exceptions", exceptions_module)
    monkeypatch.setitem(sys.modules, "rclpy.callback_groups", callback_groups_module)
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.parameter", parameter_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.msg", fa_interfaces_msg_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.srv", fa_interfaces_srv_module)


def _settings(
    tmp_path: Path,
    *,
    backend_name: str,
    command: Path,
    model: str,
    model_path: str,
    openai_realtime_api_key_env: str = "FA_ASR_OPENAI_REALTIME_API_KEY",
    openai_transcriptions_api_key_env: str = "FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY",
    health_args: tuple[str, ...] = ("--model", "{model}", "--health-check"),
) -> AsrBackendSettings:
    return AsrBackendSettings(
        name=backend_name,
        command=str(command),
        model=model,
        model_path=model_path,
        openai_realtime_api_key_env=openai_realtime_api_key_env,
        openai_transcriptions_api_key_env=openai_transcriptions_api_key_env,
        language="ja",
        args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
        health_args=health_args,
        timeout_sec=10.0,
        working_directory="",
        output_text_path="",
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
        result_format="plain_text",
    )


def test_local_command_requires_existing_model_path(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "asr")

    with pytest.raises(RuntimeError, match="backend.model_path does not exist"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(tmp_path / "missing.bin"),
            language="ja",
            args=("-m", "{model}", "-f", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_default_config_requires_explicit_backend_name() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_asr"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.openai_realtime.api_key_env"] == ""
    assert params["backend.openai_transcriptions.api_key_env"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert params["backend.result_format"] == ""
    assert params["expected_source_id"] == ""
    assert params["expected_stream_id"] == ""
    assert params["audio.qos.depth"] == 20
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] == 50
    assert params["vad.qos.reliable"] is False
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
    assert params["result.qos.depth"] == 10
    assert params["result.qos.reliable"] is True


def test_asr_node_parameter_helpers_reject_wrong_ros_parameter_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")

        assert module.FaAsrNode._string_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "audio/frame")),
            "audio_topic",
        ) == "audio/frame"
        assert module.FaAsrNode._bool_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.BOOL, False)),
            "cleanup_audio_files",
        ) is False
        assert module.FaAsrNode._integer_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 16000)),
            "target_sample_rate",
        ) == 16000
        assert module.FaAsrNode._positive_integer_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 16000)),
            "target_sample_rate",
        ) == 16000
        assert module.FaAsrNode._double_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 0.3)),
            "min_audio_sec",
        ) == 0.3
        assert module.FaAsrNode._positive_double_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 0.3)),
            "min_audio_sec",
        ) == 0.3
        assert module.FaAsrNode._string_array_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING_ARRAY, ("--audio", "{audio}"))),
            "backend.args",
        ) == ("--audio", "{audio}")
        assert module.FaAsrNode._backend_kind_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "asr")),
            "backend.kind",
        ) == "asr"
        qos = module.FaAsrNode._qos_profile(
            _ParameterMapNode(
                {
                    "audio.qos.depth": _TypedParameter(_FakeParameter.Type.INTEGER, 20),
                    "audio.qos.reliable": _TypedParameter(_FakeParameter.Type.BOOL, False),
                }
            ),
            depth_parameter="audio.qos.depth",
            reliable_parameter="audio.qos.reliable",
        )
        assert qos.depth == 20
        assert qos.history == _FakeHistoryPolicy.KEEP_LAST
        assert qos.reliability == _FakeReliabilityPolicy.BEST_EFFORT
        reliable_qos = module.FaAsrNode._qos_profile(
            _ParameterMapNode(
                {
                    "result.qos.depth": _TypedParameter(_FakeParameter.Type.INTEGER, 10),
                    "result.qos.reliable": _TypedParameter(_FakeParameter.Type.BOOL, True),
                }
            ),
            depth_parameter="result.qos.depth",
            reliable_parameter="result.qos.reliable",
        )
        assert reliable_qos.reliability == _FakeReliabilityPolicy.RELIABLE

        with pytest.raises(RuntimeError, match="cleanup_audio_files must be a bool"):
            module.FaAsrNode._bool_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "false")),
                "cleanup_audio_files",
            )
        with pytest.raises(RuntimeError, match="target_sample_rate must be an integer"):
            module.FaAsrNode._integer_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "16000")),
                "target_sample_rate",
            )
        with pytest.raises(RuntimeError, match="target_sample_rate must be greater than zero"):
            module.FaAsrNode._positive_integer_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 0)),
                "target_sample_rate",
            )
        with pytest.raises(RuntimeError, match="min_audio_sec must be a double"):
            module.FaAsrNode._double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 1)),
                "min_audio_sec",
            )
        with pytest.raises(RuntimeError, match="min_audio_sec must be finite and greater than zero"):
            module.FaAsrNode._positive_double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 0.0)),
                "min_audio_sec",
            )
        with pytest.raises(
            RuntimeError,
            match="silence_timeout_sec must be finite and greater than zero",
        ):
            module.FaAsrNode._positive_double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, float("nan"))),
                "silence_timeout_sec",
            )
        with pytest.raises(RuntimeError, match="backend.args must be a string array"):
            module.FaAsrNode._string_array_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "--audio")),
                "backend.args",
            )
        with pytest.raises(RuntimeError, match="backend.kind must be asr"):
            module.FaAsrNode._backend_kind_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "tts")),
                "backend.kind",
            )
        with pytest.raises(RuntimeError, match="backend.kind must be asr"):
            module.FaAsrNode._backend_kind_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, " asr ")),
                "backend.kind",
            )
        with pytest.raises(RuntimeError, match="audio.qos.depth must be greater than zero"):
            module.FaAsrNode._qos_profile(
                _ParameterMapNode(
                    {
                        "audio.qos.depth": _TypedParameter(_FakeParameter.Type.INTEGER, 0),
                        "audio.qos.reliable": _TypedParameter(_FakeParameter.Type.BOOL, False),
                    }
                ),
                depth_parameter="audio.qos.depth",
                reliable_parameter="audio.qos.reliable",
            )
        with pytest.raises(RuntimeError, match="audio.qos.reliable must be a bool"):
            module.FaAsrNode._qos_profile(
                _ParameterMapNode(
                    {
                        "audio.qos.depth": _TypedParameter(_FakeParameter.Type.INTEGER, 20),
                        "audio.qos.reliable": _TypedParameter(
                            _FakeParameter.Type.STRING,
                            "false",
                        ),
                    }
                ),
                depth_parameter="audio.qos.depth",
                reliable_parameter="audio.qos.reliable",
            )
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_rejects_empty_audio_data(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        with pytest.raises(ValueError, match="AudioFrame data is required"):
            module.FaAsrNode._frame_to_float(
                _FakeAudioFrame(b""),
                expected_source_id="mic0",
                expected_stream_id="stream0",
            )
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_rejects_empty_audio_data_from_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        node = module.FaAsrNode.__new__(module.FaAsrNode)
        node._logger = _FakeLogger()
        node._context_active = True
        node._active_session_id = "session-1"
        node._turn_state_lock = threading.RLock()
        node.target_sample_rate = 16000
        node.expected_source_id = "mic0"
        node.expected_stream_id = "stream0"
        node._samples = []
        node._samples_lock = threading.Lock()

        node.on_audio(_FakeAudioFrame(b""))

        assert node._samples == []
        assert node._logger.error_records == [
            "Dropping invalid AudioFrame: AudioFrame data is required"
        ]
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_rejects_non_float32le_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        frame = _FakeAudioFrame(np.zeros(160, dtype=np.float32).tobytes())
        frame.encoding = "PCM32LE"

        with pytest.raises(ValueError, match="AudioFrame encoding must be FLOAT32LE"):
            module.FaAsrNode._frame_to_float(
                frame,
                expected_source_id="mic0",
                expected_stream_id="stream0",
            )
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_rejects_unbound_source_or_stream_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        frame = _FakeAudioFrame(np.zeros(160, dtype=np.float32).tobytes())

        with pytest.raises(ValueError, match="AudioFrame source_id must match expected_source_id"):
            module.FaAsrNode._frame_to_float(
                frame,
                expected_source_id="mic1",
                expected_stream_id="stream0",
            )

        with pytest.raises(ValueError, match="AudioFrame stream_id must match expected_stream_id"):
            module.FaAsrNode._frame_to_float(
                frame,
                expected_source_id="mic0",
                expected_stream_id="audio/other",
            )
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_rejects_unbound_vad_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        msg = _FakeVadState()
        msg.source_id = "mic0"
        msg.stream_id = "stream0"

        module.FaAsrNode._validate_vad_identity(
            msg,
            expected_source_id="mic0",
            expected_stream_id="stream0",
        )

        with pytest.raises(ValueError, match="VadState source_id must match expected_source_id"):
            module.FaAsrNode._validate_vad_identity(
                msg,
                expected_source_id="mic1",
                expected_stream_id="stream0",
            )

        with pytest.raises(ValueError, match="VadState stream_id must match expected_stream_id"):
            module.FaAsrNode._validate_vad_identity(
                msg,
                expected_source_id="mic0",
                expected_stream_id="stream1",
            )
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_finalizes_current_buffer_on_vad_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        node = module.FaAsrNode.__new__(module.FaAsrNode)
        node._context_active = True
        node._turn_state_lock = threading.RLock()
        node.finalize_on_vad_end = True
        node.expected_source_id = "mic0"
        node.expected_stream_id = "stream0"
        submissions: list[tuple[str, bool]] = []

        def submit_current_buffer(
            reason: str,
            *,
            publish_timeout_if_empty: bool,
        ) -> None:
            submissions.append((reason, publish_timeout_if_empty))

        node._submit_current_buffer = submit_current_buffer
        msg = _FakeVadState()
        msg.source_id = "mic0"
        msg.stream_id = "stream0"
        msg.is_speech = False
        msg.end = True

        node.on_vad(msg)

        assert submissions == [("vad_end", False)]
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_maps_unexpected_backend_exception_to_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    _install_asr_node_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        node = module.FaAsrNode.__new__(module.FaAsrNode)
        node._logger = _FakeLogger()
        node.backend = _FailingAsrBackend()
        node._backend_lock = threading.Lock()
        published: list[tuple[str, int, int, str, str]] = []

        def publish_result(
            session_id: str,
            user_turn_id: int,
            status: int,
            reason: str,
            text: str,
        ) -> None:
            published.append((session_id, user_turn_id, status, reason, text))

        node._publish_result = publish_result
        job = module.TranscriptionJob(
            session_id="session-1",
            user_turn_id=9,
            samples=np.zeros(1, dtype=np.float32),
            sample_rate=16000,
            reason="vad_end",
        )

        node._run_transcription(job)

        assert published == [("session-1", 9, _FakeAsrResult.STATUS_ERROR, "backend_error", "")]
        assert shutdown_calls == [True]
        assert node._logger.error_records == [
            "ASR transcription failed: asr backend crashed"
        ]
        assert node._logger.fatal_records == [
            "ASR fail closed: ASR backend failed: asr backend crashed"
        ]
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_maps_backend_timeout_to_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    _install_asr_node_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        node = module.FaAsrNode.__new__(module.FaAsrNode)
        node._logger = _FakeLogger()
        node.backend = _TimeoutAsrBackend()
        node._backend_lock = threading.Lock()
        published: list[tuple[str, int, int, str, str]] = []

        def publish_result(
            session_id: str,
            user_turn_id: int,
            status: int,
            reason: str,
            text: str,
        ) -> None:
            published.append((session_id, user_turn_id, status, reason, text))

        node._publish_result = publish_result
        job = module.TranscriptionJob(
            session_id="session-1",
            user_turn_id=9,
            samples=np.zeros(1, dtype=np.float32),
            sample_rate=16000,
            reason="vad_end",
        )

        node._run_transcription(job)

        assert published == [("session-1", 9, _FakeAsrResult.STATUS_ERROR, "backend_timeout", "")]
        assert shutdown_calls == [True]
        assert node._logger.fatal_records == ["ASR fail closed: ASR backend timed out"]
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_publishes_text_derived_from_backend_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    _install_asr_node_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        node = module.FaAsrNode.__new__(module.FaAsrNode)
        node._logger = _FakeLogger()
        node.backend = _SegmentAsrBackend()
        node._backend_lock = threading.Lock()
        published: list[tuple[str, int, int, str, str]] = []

        def publish_result(
            session_id: str,
            user_turn_id: int,
            status: int,
            reason: str,
            text: str,
        ) -> None:
            published.append((session_id, user_turn_id, status, reason, text))

        node._publish_result = publish_result
        job = module.TranscriptionJob(
            session_id="session-1",
            user_turn_id=9,
            samples=np.zeros(2, dtype=np.float32),
            sample_rate=16000,
            reason="vad_end",
        )

        node._run_transcription(job)

        assert published == [("session-1", 9, _FakeAsrResult.STATUS_FINAL, "vad_end", "hello world")]
        assert shutdown_calls == []
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_node_main_uses_multithreaded_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_calls: list[bool] = []
    _install_asr_node_import_fakes(monkeypatch, shutdown_calls)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        constructed_nodes: list[_MainFakeNode] = []

        class _MainFakeNode(_FakeNode):
            def __init__(self) -> None:
                self.destroyed = False
                constructed_nodes.append(self)

            def destroy_node(self) -> bool:
                self.destroyed = True
                return True

        module.FaAsrNode = _MainFakeNode

        module.main(["--ros-args"])

        assert len(constructed_nodes) == 1
        assert constructed_nodes[0].destroyed is True
        assert len(_FakeMultiThreadedExecutor.instances) == 1
        executor = _FakeMultiThreadedExecutor.instances[0]
        assert executor.num_threads == 2
        assert executor.nodes == [constructed_nodes[0]]
        assert executor.spin_called is True
        assert executor.shutdown_called is True
        assert shutdown_calls == [True]
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_openai_realtime_requires_model_id(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_openai_realtime_config(
            command=str(command),
            model="",
            api_key_env="FA_ASR_OPENAI_REALTIME_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_openai_transcriptions_requires_model_id(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_openai_transcriptions_config(
            command=str(command),
            model="",
            api_key_env="FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_openai_realtime_requires_api_key_env_name(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.openai_realtime.api_key_env is required"):
        load_openai_realtime_config(
            command=str(command),
            model="gpt-4o-realtime-preview",
            api_key_env="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_openai_transcriptions_requires_api_key_env_name(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(
        RuntimeError,
        match="backend.openai_transcriptions.api_key_env is required",
    ):
        load_openai_transcriptions_config(
            command=str(command),
            model="gpt-4o-transcribe",
            api_key_env="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_openai_realtime_requires_api_key_env_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_executable(tmp_path / "worker")
    monkeypatch.delenv("FA_ASR_OPENAI_REALTIME_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match=(
            "environment variable FA_ASR_OPENAI_REALTIME_API_KEY referenced by "
            "backend.openai_realtime.api_key_env is required"
        ),
    ):
        load_openai_realtime_config(
            command=str(command),
            model="gpt-4o-realtime-preview",
            api_key_env="FA_ASR_OPENAI_REALTIME_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_openai_transcriptions_requires_api_key_env_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_executable(tmp_path / "worker")
    monkeypatch.delenv("FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY", raising=False)

    with pytest.raises(
        RuntimeError,
        match=(
            "environment variable FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY referenced by "
            "backend.openai_transcriptions.api_key_env is required"
        ),
    ):
        load_openai_transcriptions_config(
            command=str(command),
            model="gpt-4o-transcribe",
            api_key_env="FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_openai_realtime_maps_configured_api_key_env_to_worker_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_openai_env_probe_worker(tmp_path / "openai_worker")
    monkeypatch.setenv("CUSTOM_OPENAI_REALTIME_KEY", "expected-openai-secret")
    monkeypatch.setenv("EXPECTED_OPENAI_API_KEY_FOR_TEST", "expected-openai-secret")
    backend = OpenAiRealtimeAsrBackend(
        load_openai_realtime_config(
            command=str(command),
            model="gpt-4o-realtime-preview",
            api_key_env="CUSTOM_OPENAI_REALTIME_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )
    )

    assert asr_transcript_text(
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros(160, dtype=np.float32),
                sample_rate=16000,
            )
        )
    ) == "openai-env-ok"


def test_openai_transcriptions_maps_configured_api_key_env_to_worker_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_openai_env_probe_worker(tmp_path / "openai_worker")
    monkeypatch.setenv("CUSTOM_OPENAI_TRANSCRIPTIONS_KEY", "expected-openai-secret")
    monkeypatch.setenv("EXPECTED_OPENAI_API_KEY_FOR_TEST", "expected-openai-secret")
    backend = OpenAiTranscriptionsAsrBackend(
        load_openai_transcriptions_config(
            command=str(command),
            model="gpt-4o-transcribe",
            api_key_env="CUSTOM_OPENAI_TRANSCRIPTIONS_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )
    )

    assert asr_transcript_text(
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros(160, dtype=np.float32),
                sample_rate=16000,
            )
        )
    ) == "openai-env-ok"


def test_parakeet_worker_requires_model_id(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_parakeet_worker_config(
            command=str(command),
            model="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_model_id_worker_backends_require_health_args(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.health_args must not be empty"):
        load_parakeet_worker_config(
            command=str(command),
            model="nvidia/parakeet",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=(),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )

    with pytest.raises(RuntimeError, match="backend.health_args must not be empty"):
        load_openai_realtime_config(
            command=str(command),
            model="gpt-4o-realtime-preview",
            api_key_env="FA_ASR_OPENAI_REALTIME_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=(),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )

    with pytest.raises(RuntimeError, match="backend.health_args must not be empty"):
        load_openai_transcriptions_config(
            command=str(command),
            model="gpt-4o-transcribe",
            api_key_env="FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=(),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_backend_health_args_require_model_placeholder(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(
        RuntimeError,
        match=r"backend.health_args must include placeholders: \{model\}",
    ):
        load_parakeet_worker_config(
            command=str(command),
            model="nvidia/parakeet",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--health-check",),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_backend_health_check_fails_startup_before_transcription(tmp_path: Path) -> None:
    command = _write_health_fail_worker(tmp_path / "worker")

    with pytest.raises(
        RuntimeError,
        match="ASR backend health check failed: code=7 stderr=health failed",
    ):
        ParakeetWorkerAsrBackend(
            load_parakeet_worker_config(
                command=str(command),
                model="nvidia/parakeet",
                language="ja",
                args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
                health_args=("--model", "{model}", "--health-check"),
                timeout_sec=10.0,
                working_directory_value="",
                output_text_path="",
                workspace_dir=tmp_path / "work",
                cleanup_audio_files=True,
                result_format="plain_text",
            )
        )


def test_build_backend_requires_backend_name(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.name is required"):
        build_asr_backend(
            _settings(
                tmp_path,
                backend_name="",
                command=command,
                model="gpt-4o-transcribe",
                model_path="",
            )
        )


def test_build_backend_rejects_unknown_backend(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported ASR backend.name: bogus"):
        build_asr_backend(
            _settings(
                tmp_path,
                backend_name="bogus",
                command=command,
                model="gpt-4o-transcribe",
                model_path="",
            )
        )


def test_backends_use_dedicated_classes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")
    monkeypatch.setenv("FA_ASR_OPENAI_REALTIME_API_KEY", "key")
    monkeypatch.setenv("FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY", "key")

    local_command = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="local_command",
            command=command,
            model="",
            model_path=str(model_path),
        )
    )
    whisper_cpp = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="whisper.cpp",
            command=command,
            model="",
            model_path=str(model_path),
        )
    )
    parakeet_worker = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="parakeet_worker",
            command=command,
            model="nvidia/parakeet",
            model_path="",
        )
    )
    openai_realtime = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="openai_realtime",
            command=command,
            model="gpt-4o-realtime-preview",
            model_path="",
        )
    )
    openai_transcriptions = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="openai_transcriptions",
            command=command,
            model="gpt-4o-transcribe",
            model_path="",
        )
    )

    assert isinstance(local_command, LocalCommandAsrBackend)
    assert isinstance(whisper_cpp, WhisperCppAsrBackend)
    assert isinstance(parakeet_worker, ParakeetWorkerAsrBackend)
    assert isinstance(openai_realtime, OpenAiRealtimeAsrBackend)
    assert isinstance(openai_transcriptions, OpenAiTranscriptionsAsrBackend)


def test_whisper_cpp_legacy_backend_name_is_not_supported(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported ASR backend.name: whisper_cpp"):
        build_asr_backend(
            _settings(
                tmp_path,
                backend_name="whisper_cpp",
                command=command,
                model="",
                model_path=str(model_path),
            )
        )


def test_whisper_cpp_uses_model_path_contract(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        load_whisper_cpp_config(
            command=str(command),
            model_path_value="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_command_and_model_paths_are_resolved_and_executable(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    config = load_local_command_config(
        command=str(command),
        model_path_value=str(model_path),
        language="ja",
        args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
        health_args=("--model", "{model}", "--health-check"),
        timeout_sec=10.0,
        working_directory_value="",
        output_text_path="",
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
        result_format="plain_text",
    )

    assert config.process.executable == str(command.resolve(strict=True))
    assert os.access(config.process.executable, os.X_OK)
    assert config.process.model == str(model_path.resolve(strict=True))
    assert config.process.payload_encoding == "float32le_raw"
    assert config.process.result_format == "plain_text"


def test_command_backend_requires_explicit_result_format(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.result_format is required"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="",
        )


def test_command_backend_rejects_unsupported_result_format(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported backend.result_format: xml"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="xml",
        )


def test_command_backend_rejects_non_executable_command(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")
    command.chmod(command.stat().st_mode & ~0o111)
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.command is not executable"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            health_args=("--model", "{model}", "--health-check"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_backend_args_reject_unknown_or_malformed_placeholders(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported backend.args placeholder: unknown"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=(
                "--model",
                "{model}",
                "--audio",
                "{audio}",
                "--sample-rate",
                "{sample_rate}",
                "--x",
                "{unknown}",
            ),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )

    with pytest.raises(RuntimeError, match="malformed format string"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=("--model", "{model", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_backend_args_require_audio_placeholder(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"backend.args must include placeholders: \{audio\}"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=("--model", "{model}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_backend_args_require_sample_rate_placeholder(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match=r"backend.args must include placeholders: \{sample_rate\}",
    ):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_output_placeholder_requires_output_text_path(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.output_text_path is required"):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=(
                "--model",
                "{model}",
                "--audio",
                "{audio}",
                "--sample-rate",
                "{sample_rate}",
                "--output",
                "{output}",
            ),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_output_text_path_rejects_unknown_placeholder(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("model", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="unsupported backend.output_text_path placeholder: unknown",
    ):
        load_local_command_config(
            command=str(command),
            model_path_value=str(model_path),
            language="ja",
            args=(
                "--model",
                "{model}",
                "--audio",
                "{audio}",
                "--sample-rate",
                "{sample_rate}",
                "--output",
                "{output}",
            ),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="transcript_{unknown}.txt",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
            result_format="plain_text",
        )


def test_asr_request_validation_rejects_implicit_sample_casts(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")
    model_path = tmp_path / "model.bin"
    model_path.write_text("ok", encoding="utf-8")
    backend = build_asr_backend(
        _settings(
            tmp_path,
            backend_name="local_command",
            command=command,
            model="",
            model_path=str(model_path),
        )
    )

    with pytest.raises(ValueError, match="samples must be float32"):
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros(160, dtype=np.float64),
                sample_rate=16000,
            )
        )
    with pytest.raises(ValueError, match="samples must be one-dimensional"):
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros((2, 80), dtype=np.float32),
                sample_rate=16000,
            )
        )
    with pytest.raises(ValueError, match="sample_rate must be positive"):
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros(160, dtype=np.float32),
                sample_rate=0,
            )
        )
