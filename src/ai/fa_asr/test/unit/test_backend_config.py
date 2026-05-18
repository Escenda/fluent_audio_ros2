import threading
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest
import yaml

from fa_asr_py.backends.base import AsrRequest
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
PYTHON_SOURCES = tuple(
    path
    for path in sorted((PACKAGE_ROOT / "fa_asr_py").rglob("*.py"))
    if "__pycache__" not in path.parts
)


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


class _FakeNode:
    def get_logger(self) -> "_FakeLogger":
        return self._logger


class _FakeLogger:
    def __init__(self) -> None:
        self.error_records: list[tuple[str, Exception]] = []

    def error(self, message: str, exc: Exception) -> None:
        self.error_records.append((message, exc))


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


class _FakeTurnContext:
    pass


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


class _BackendCrash(Exception):
    pass


class _FailingAsrBackend:
    name = "failing"

    def transcribe(self, request: AsrRequest) -> str:
        raise _BackendCrash("asr backend crashed")


def _install_asr_node_import_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy_module = ModuleType("rclpy")

    def shutdown() -> None:
        pass

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
    fa_interfaces_msg_module.AsrResult = _FakeAsrResult
    fa_interfaces_msg_module.AudioFrame = _FakeAudioFrame
    fa_interfaces_msg_module.TurnContext = _FakeTurnContext
    fa_interfaces_msg_module.VadState = _FakeVadState

    monkeypatch.setitem(sys.modules, "rclpy", rclpy_module)
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.parameter", parameter_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.msg", fa_interfaces_msg_module)


def _settings(
    tmp_path: Path,
    *,
    backend_name: str,
    command: Path,
    model: str,
    model_path: str,
    openai_realtime_api_key_env: str = "FA_ASR_OPENAI_REALTIME_API_KEY",
    openai_transcriptions_api_key_env: str = "FA_ASR_OPENAI_TRANSCRIPTIONS_API_KEY",
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
        timeout_sec=10.0,
        working_directory="",
        output_text_path="",
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
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
        )


def test_default_config_requires_explicit_backend_name() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source_path = PACKAGE_ROOT / "fa_asr_py" / "asr_node.py"
    source = source_path.read_text(encoding="utf-8")

    params = config["fa_asr"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.openai_realtime.api_key_env"] == ""
    assert params["backend.openai_transcriptions.api_key_env"] == ""
    assert params["backend.args"] == []
    assert params["expected_source_id"] == ""
    assert params["expected_stream_id"] == ""
    assert 'declare_parameter("expected_source_id", "")' in source
    assert 'declare_parameter("expected_stream_id", "")' in source
    assert "ParameterUninitializedException" not in source
    assert "return tuple()" not in source
    assert "tuple(str(part) for part in args_value)" not in source


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
        assert module.FaAsrNode._double_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.DOUBLE, 0.3)),
            "min_audio_sec",
        ) == 0.3
        assert module.FaAsrNode._string_array_parameter(
            _TypedNode(_TypedParameter(_FakeParameter.Type.STRING_ARRAY, ("--audio", "{audio}"))),
            "backend.args",
        ) == ("--audio", "{audio}")

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
        with pytest.raises(RuntimeError, match="min_audio_sec must be a double"):
            module.FaAsrNode._double_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.INTEGER, 1)),
                "min_audio_sec",
            )
        with pytest.raises(RuntimeError, match="backend.args must be a string array"):
            module.FaAsrNode._string_array_parameter(
                _TypedNode(_TypedParameter(_FakeParameter.Type.STRING, "--audio")),
                "backend.args",
            )
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_asr_python_sources_keep_dependency_boundary_explicit() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in PYTHON_SOURCES)

    assert "ImportError" not in combined
    assert "dict[str, Any]" not in combined
    assert "Any" not in combined
    assert "list[object]" not in combined
    assert "dict[str, object]" not in combined
    assert "typing import object" not in combined


def test_openai_backends_are_external_worker_slots() -> None:
    openai_realtime = (
        PACKAGE_ROOT / "fa_asr_py" / "backends" / "openai_realtime.py"
    ).read_text(encoding="utf-8")
    openai_transcriptions = (
        PACKAGE_ROOT / "fa_asr_py" / "backends" / "openai_transcriptions.py"
    ).read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    combined = "\n".join((openai_realtime, openai_transcriptions))

    assert "_CommandProcessRunner" in openai_realtime
    assert "_CommandProcessRunner" in openai_transcriptions
    assert "import openai" not in combined
    assert "from openai" not in combined
    assert "websocket" not in combined.lower()
    assert "aiohttp" not in combined
    assert "requests" not in combined
    assert "openai" not in package_xml.lower()


def test_asr_node_rejects_non_canonical_audio_frames() -> None:
    source_path = PACKAGE_ROOT / "fa_asr_py" / "asr_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "np.zeros(0" not in source
    assert "if samples.size == 0:" not in source
    assert "_resample_linear" not in source
    assert "_to_mono" not in source
    assert "np.frombuffer(bytes(msg.data), dtype=np.int16)" not in source
    assert "AudioFrame data is required" in source
    assert "AudioFrame channels must be 1" in source
    assert "AudioFrame source_id and stream_id are required" in source
    assert "AudioFrame source_id must match expected_source_id" in source
    assert "AudioFrame stream_id must match expected_stream_id" in source
    assert "expected_stream_id=self.expected_stream_id" in source
    assert "AudioFrame layout must be interleaved" in source
    assert "AudioFrame encoding must be FLOAT32LE" in source
    assert "AudioFrame bit_depth must be 32" in source
    assert "AudioFrame sample_rate must match target_sample_rate" in source
    assert "AudioFrame samples must be normalized to [-1.0, 1.0]" in source


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
        node.target_sample_rate = 16000
        node.expected_source_id = "mic0"
        node.expected_stream_id = "stream0"
        node._samples = []
        node._samples_lock = threading.Lock()

        node.on_audio(_FakeAudioFrame(b""))

        assert node._samples == []
        assert len(node._logger.error_records) == 1
        error_message, error_exception = node._logger.error_records[0]
        assert error_message == "Dropping invalid AudioFrame: %s"
        assert str(error_exception) == "AudioFrame data is required"
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


def test_asr_node_maps_unexpected_backend_exception_to_error_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_asr_node_import_fakes(monkeypatch)
    monkeypatch.syspath_prepend(str(PACKAGE_ROOT))
    sys.modules.pop("fa_asr_py.asr_node", None)

    try:
        module = importlib.import_module("fa_asr_py.asr_node")
        node = module.FaAsrNode.__new__(module.FaAsrNode)
        node._logger = _FakeLogger()
        node.backend = _FailingAsrBackend()
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
        assert len(node._logger.error_records) == 1
        error_message, error_exception = node._logger.error_records[0]
        assert error_message == "ASR transcription failed: %s"
        assert isinstance(error_exception, _BackendCrash)
        assert str(error_exception) == "asr backend crashed"
    finally:
        sys.modules.pop("fa_asr_py.asr_node", None)


def test_command_backend_does_not_clip_audio() -> None:
    source_path = PACKAGE_ROOT / "fa_asr_py" / "backends" / "_command_process.py"
    source = source_path.read_text(encoding="utf-8")

    assert "np.clip" not in source
    assert "_float_to_pcm16" not in source
    assert "astype(np.int16)" not in source
    assert "ASR request samples must be normalized to [-1.0, 1.0]" in source


def test_asr_node_source_does_not_hide_parameter_type_conversion() -> None:
    source_path = PACKAGE_ROOT / "fa_asr_py" / "asr_node.py"
    source = source_path.read_text(encoding="utf-8")

    assert "bool(self.get_parameter" not in source
    assert "int(self.get_parameter" not in source
    assert "float(self.get_parameter" not in source
    assert "str(self.get_parameter" not in source
    assert "tuple(str(" not in source


def test_openai_realtime_requires_model_id(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_openai_realtime_config(
            command=str(command),
            model="",
            api_key_env="FA_ASR_OPENAI_REALTIME_API_KEY",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
        )
    )

    assert (
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros(160, dtype=np.float32),
                sample_rate=16000,
            )
        )
        == "openai-env-ok"
    )


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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
        )
    )

    assert (
        backend.transcribe(
            AsrRequest(
                session_id="session",
                user_turn_id=1,
                samples=np.zeros(160, dtype=np.float32),
                sample_rate=16000,
            )
        )
        == "openai-env-ok"
    )


def test_parakeet_worker_requires_model_id(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="backend.model is required"):
        load_parakeet_worker_config(
            command=str(command),
            model="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
            backend_name="whisper_cpp",
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


def test_whisper_cpp_uses_model_path_contract(tmp_path: Path) -> None:
    command = _write_executable(tmp_path / "worker")

    with pytest.raises(RuntimeError, match="backend.model_path is required"):
        load_whisper_cpp_config(
            command=str(command),
            model_path_value="",
            language="ja",
            args=("--model", "{model}", "--audio", "{audio}", "--sample-rate", "{sample_rate}"),
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
        timeout_sec=10.0,
        working_directory_value="",
        output_text_path="",
        workspace_dir=tmp_path / "work",
        cleanup_audio_files=True,
    )

    assert config.process.executable == str(command.resolve(strict=True))
    assert os.access(config.process.executable, os.X_OK)
    assert config.process.model == str(model_path.resolve(strict=True))
    assert config.process.payload_encoding == "float32le_raw"


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
            timeout_sec=10.0,
            working_directory_value="",
            output_text_path="",
            workspace_dir=tmp_path / "work",
            cleanup_audio_files=True,
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
