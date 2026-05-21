import importlib
import json
import queue
import sys
import threading
from types import ModuleType

import numpy as np
import pytest

from fa_asr_py.backends.base import (
    AsrAudioPayload,
    AsrBackendCapability,
    AsrRequest,
    AsrStreamRequest,
    AsrStreamResult,
    AsrTranscript,
    AsrTranscriptSegment,
    plain_text_to_asr_transcript,
)
from fa_asr_py.timeline import (
    ERROR_RANGE_NOT_CONTINUOUS,
    ERROR_RANGE_OUTSIDE_WINDOW,
    ERROR_TIME_RANGE_UNRESOLVED,
    ERROR_WINDOW_NOT_FOUND,
    RollingAsrTimeline,
    TimelineRangeError,
    parse_numeric_time_range,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.error_records: list[str] = []
        self.debug_records: list[str] = []
        self.info_records: list[str] = []
        self.fatal_records: list[str] = []

    def error(self, message: str) -> None:
        self.error_records.append(message)

    def debug(self, message: str) -> None:
        self.debug_records.append(message)

    def info(self, message: str) -> None:
        self.info_records.append(message)

    def fatal(self, message: str) -> None:
        self.fatal_records.append(message)


class _FakeClockNow:
    def __init__(self, *, sec: int = 99, nanosec: int = 123) -> None:
        self.nanoseconds = sec * 1_000_000_000 + nanosec
        self._sec = sec
        self._nanosec = nanosec

    def to_msg(self) -> "_FakeHeaderStamp":
        return _FakeHeaderStamp(sec=self._sec, nanosec=self._nanosec)


class _FakeClock:
    def now(self) -> _FakeClockNow:
        return _FakeClockNow()


class _RecordingPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, msg) -> None:
        self.messages.append(msg)


class _FakeNode:
    def get_logger(self) -> _FakeLogger:
        return self._logger

    def get_clock(self) -> _FakeClock:
        return _FakeClock()


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


class _FakeReliabilityPolicy:
    BEST_EFFORT = "best_effort"
    RELIABLE = "reliable"


class _FakeHistoryPolicy:
    KEEP_LAST = "keep_last"


class _FakeParameterUninitializedException(Exception):
    pass


class _FakeTime:
    pass


class _FakeMutuallyExclusiveCallbackGroup:
    pass


class _FakeMultiThreadedExecutor:
    def __init__(self, *, num_threads: int) -> None:
        self.num_threads = num_threads
        self.nodes: list[_FakeNode] = []
        self.spin_called = False
        self.shutdown_called = False

    def add_node(self, node: _FakeNode) -> None:
        self.nodes.append(node)

    def spin(self) -> None:
        self.spin_called = True

    def shutdown(self) -> None:
        self.shutdown_called = True


class _FakeAsrResult:
    STATUS_FINAL = 1
    STATUS_TIMEOUT = 2
    STATUS_ERROR = 3
    STATUS_PARTIAL = 4


class _FakeAsrState:
    STATE_WAITING = 1
    STATE_COLLECTING = 2
    STATE_QUEUED = 3
    STATE_TRANSCRIBING = 4
    STATE_COMPLETED = 5
    STATE_TIMEOUT = 6
    STATE_FAILED = 7


class _FakeAsrEvent:
    EVENT_STARTUP_IDLE = 1
    EVENT_CONTROL_RECEIVED = 2
    EVENT_CONTROL_REJECTED_IDENTITY = 3
    EVENT_CONTROL_WINDOW_OPENED = 4
    EVENT_CLOSE_IGNORED_NO_WINDOW = 5
    EVENT_CONTROL_WINDOW_CLOSED = 6
    EVENT_SUBMIT_SKIPPED_SUBMIT_ON_CLOSE_FALSE = 7
    EVENT_SUBMIT_SKIPPED_CONTEXT_INACTIVE = 8
    EVENT_INVALID_CLOSE_TIME = 9
    EVENT_TIMELINE_SLICE_UNAVAILABLE = 10
    EVENT_TIMELINE_SLICE_NOT_CONTINUOUS = 11
    EVENT_TIMELINE_OVERLAP_DERIVED_FAILURE = 12
    EVENT_WINDOW_TOO_SHORT = 13
    EVENT_JOB_QUEUED = 14
    EVENT_BACKEND_TRANSCRIPTION_STARTED = 15
    EVENT_BACKEND_COMPLETED = 16
    EVENT_BACKEND_TIMEOUT = 17
    EVENT_BACKEND_ERROR = 18
    EVENT_INVALID_AUDIO_FRAME_DROPPED = 19
    EVENT_AUDIO_FRAME_TIMELINE_APPEND_DROPPED = 20
    EVENT_FAIL_CLOSED = 21
    EVENT_STREAM_OPENED = 22
    EVENT_STREAM_AUDIO_PUSHED = 23
    EVENT_STREAM_PARTIAL_RESULT = 24
    EVENT_STREAM_FINAL_RESULT = 25
    EVENT_STREAM_CLOSED = 26
    EVENT_STREAM_CANCELLED = 27
    EVENT_STREAM_ERROR = 28


class _FakeResolvedTimeRange:
    CLOCK_AGENT = "agent"
    CLOCK_ROBOT = "robot"
    CLOCK_MEDIA = "media"

    def __init__(
        self,
        *,
        start_unix_ns: int = 0,
        end_unix_ns: int = 0,
        clock: str = "",
        uncertainty_ns: int = 0,
        uncertainty_reason: str = "",
    ) -> None:
        self.start_unix_ns = start_unix_ns
        self.end_unix_ns = end_unix_ns
        self.clock = clock
        self.uncertainty_ns = uncertainty_ns
        self.uncertainty_reason = uncertainty_reason


class _FakeAudioWindowRef:
    def __init__(
        self,
        *,
        window_id: str = "",
        window_epoch: int = 0,
        source_id: str = "",
        stream_id: str = "",
        time_range: _FakeResolvedTimeRange | None = None,
    ) -> None:
        self.window_id = window_id
        self.window_epoch = window_epoch
        self.source_id = source_id
        self.stream_id = stream_id
        self.time_range = time_range if time_range is not None else _FakeResolvedTimeRange()


class _FakeAudioModelRef:
    def __init__(
        self,
        *,
        backend_name: str = "",
        backend_kind: str = "",
        model_id: str = "",
        model_path: str = "",
        model_version: str = "",
        model_revision: str = "",
    ) -> None:
        self.backend_name = backend_name
        self.backend_kind = backend_kind
        self.model_id = model_id
        self.model_path = model_path
        self.model_version = model_version
        self.model_revision = model_revision


class _FakeTranscriptSegment:
    def __init__(
        self,
        *,
        start_unix_ns: int = 0,
        end_unix_ns: int = 0,
        text: str = "",
        speaker_label: str = "",
    ) -> None:
        self.start_unix_ns = start_unix_ns
        self.end_unix_ns = end_unix_ns
        self.text = text
        self.speaker_label = speaker_label


class _FakeHeaderStamp:
    def __init__(self, *, sec: int, nanosec: int) -> None:
        self.sec = sec
        self.nanosec = nanosec


class _FakeHeader:
    def __init__(self, *, sec: int, nanosec: int) -> None:
        self.stamp = _FakeHeaderStamp(sec=sec, nanosec=nanosec)


class _FakeAudioFrame:
    def __init__(
        self,
        *,
        samples: np.ndarray,
        sec: int,
        nanosec: int = 0,
        sample_rate: int = 10,
    ) -> None:
        self.header = _FakeHeader(sec=sec, nanosec=nanosec)
        self.data = samples.astype(np.float32).tobytes()
        self.source_id = "mic0"
        self.stream_id = "stream0"
        self.layout = "interleaved"
        self.channels = 1
        self.encoding = "FLOAT32LE"
        self.bit_depth = 32
        self.sample_rate = sample_rate


class _FakeTurnContext:
    pass


class _FakeVadState:
    def __init__(
        self,
        *,
        sec: int,
        nanosec: int = 0,
        source_id: str = "mic0",
        stream_id: str = "stream0",
        is_speech: bool = False,
        start: bool = False,
        end: bool = False,
    ) -> None:
        self.header = _FakeHeader(sec=sec, nanosec=nanosec)
        self.source_id = source_id
        self.stream_id = stream_id
        self.is_speech = is_speech
        self.start = start
        self.end = end


class _FakeTranscribeAudioRequest:
    def __init__(self, *, time_range_spec: str, audio_scope: str = "") -> None:
        self.time_range_spec = time_range_spec
        self.audio_scope = audio_scope


class _FakeTranscribeAudioResponse:
    ERROR_NONE = "none"
    ERROR_TIME_RANGE_UNRESOLVED = "time_range_unresolved"
    ERROR_WINDOW_NOT_FOUND = "window_not_found"
    ERROR_RANGE_OUTSIDE_WINDOW = "range_outside_window"
    ERROR_RANGE_NOT_CONTINUOUS = "range_not_continuous"
    ERROR_UNSUPPORTED_AUDIO_SCOPE = "unsupported_audio_scope"
    ERROR_TRANSCRIBE_FAILED = "transcribe_failed"

    def __init__(self) -> None:
        self.success = False
        self.error_code = ""
        self.message = ""
        self.segments: list[_FakeTranscriptSegment] = []
        self.audio_window_ref = _FakeAudioWindowRef()
        self.model_ref = _FakeAudioModelRef()
        self.time_range = _FakeResolvedTimeRange()


class _FakeTranscribeAudio:
    Request = _FakeTranscribeAudioRequest
    Response = _FakeTranscribeAudioResponse


class _RecordingBackend:
    name = "recording_backend"
    capability = AsrBackendCapability(
        audio_encoding="FLOAT32LE",
        sample_rate_hz=10,
        channels=1,
        streaming=False,
        final_results_only=True,
    )

    def __init__(
        self,
        transcript: str = "transcript",
        *,
        segments: tuple[AsrTranscriptSegment, ...] | None = None,
    ) -> None:
        self.transcript = transcript
        self.segments = segments
        self.requests: list[AsrRequest] = []

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        self.requests.append(request)
        if self.segments is not None:
            return AsrTranscript(segments=self.segments)
        return plain_text_to_asr_transcript(
            self.transcript,
            sample_count=request.payload.sample_count,
        )


def _streaming_result(text: str, *, sample_count: int, is_final: bool) -> AsrStreamResult:
    if is_final and not text.strip():
        transcript = AsrTranscript(
            segments=(
                AsrTranscriptSegment(
                    start_sample=0,
                    end_sample=sample_count,
                    text=text,
                ),
            )
        )
    else:
        transcript = plain_text_to_asr_transcript(text, sample_count=sample_count)
    return AsrStreamResult(
        transcript=transcript,
        is_final=is_final,
        sample_count=sample_count,
    )


class _RecordingStreamingSession:
    def __init__(self, *, partial_text: str = "partial", final_text: str = "final") -> None:
        self.partial_text = partial_text
        self.final_text = final_text
        self.pushed_payloads: list[AsrAudioPayload] = []
        self.pending_results: list[AsrStreamResult] = []
        self.sample_count = 0
        self.partial_emitted = False
        self.finish_called = False
        self.cancel_called = False

    def push_audio(self, payload: AsrAudioPayload) -> tuple[AsrStreamResult, ...]:
        self.pushed_payloads.append(payload)
        self.sample_count += payload.sample_count
        if self.partial_text and not self.partial_emitted:
            self.pending_results.append(
                _streaming_result(
                    self.partial_text,
                    sample_count=self.sample_count,
                    is_final=False,
                )
            )
            self.partial_emitted = True
        return self.drain_results()

    def drain_results(self) -> tuple[AsrStreamResult, ...]:
        results = tuple(self.pending_results)
        self.pending_results.clear()
        return results

    def finish(self) -> tuple[AsrStreamResult, ...]:
        self.finish_called = True
        return (
            _streaming_result(
                self.final_text,
                sample_count=self.sample_count,
                is_final=True,
            ),
        )

    def cancel(self) -> None:
        self.cancel_called = True


class _RecordingStreamingBackend:
    name = "recording_streaming_backend"

    def __init__(
        self,
        *,
        audio_encoding: str = "FLOAT32LE",
        sample_rate_hz: int = 10,
        partial_text: str = "partial",
        final_text: str = "final",
    ) -> None:
        self.capability = AsrBackendCapability(
            audio_encoding=audio_encoding,
            sample_rate_hz=sample_rate_hz,
            channels=1,
            streaming=True,
            final_results_only=False,
        )
        self.partial_text = partial_text
        self.final_text = final_text
        self.streams: list[_RecordingStreamingSession] = []
        self.start_calls: list[tuple[str, int]] = []
        self.requests: list[AsrRequest] = []

    def start_stream(self, request: AsrStreamRequest) -> _RecordingStreamingSession:
        self.start_calls.append((request.session_id, request.user_turn_id))
        session = _RecordingStreamingSession(
            partial_text=self.partial_text,
            final_text=self.final_text,
        )
        self.streams.append(session)
        return session

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        self.requests.append(request)
        return plain_text_to_asr_transcript(
            "timeline transcript",
            sample_count=request.payload.sample_count,
        )


class _TimeoutBackend:
    name = "timeout_backend"
    capability = _RecordingBackend.capability

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        del request
        raise TimeoutError("backend timed out")


class _ErrorBackend:
    name = "error_backend"
    capability = _RecordingBackend.capability

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        del request
        raise RuntimeError("backend failed")


def _install_asr_node_import_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    rclpy_module = ModuleType("rclpy")
    rclpy_module.shutdown = lambda: None
    rclpy_module.init = lambda args=None: None
    rclpy_module.spin = lambda node: None

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
    fa_interfaces_msg_module.AsrEvent = _FakeAsrEvent
    fa_interfaces_msg_module.AsrResult = _FakeAsrResult
    fa_interfaces_msg_module.AsrState = _FakeAsrState
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
    monkeypatch.setitem(sys.modules, "rclpy.node", node_module)
    monkeypatch.setitem(sys.modules, "rclpy.callback_groups", callback_groups_module)
    monkeypatch.setitem(sys.modules, "rclpy.executors", executors_module)
    monkeypatch.setitem(sys.modules, "rclpy.parameter", parameter_module)
    monkeypatch.setitem(sys.modules, "rclpy.exceptions", exceptions_module)
    monkeypatch.setitem(sys.modules, "rclpy.qos", qos_module)
    monkeypatch.setitem(sys.modules, "builtin_interfaces", builtin_interfaces_module)
    monkeypatch.setitem(sys.modules, "builtin_interfaces.msg", builtin_interfaces_msg_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces", fa_interfaces_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.msg", fa_interfaces_msg_module)
    monkeypatch.setitem(sys.modules, "fa_interfaces.srv", fa_interfaces_srv_module)


def _asr_node_module(monkeypatch: pytest.MonkeyPatch):
    _install_asr_node_import_fakes(monkeypatch)
    sys.modules.pop("fa_asr_py.asr_node", None)
    return importlib.import_module("fa_asr_py.asr_node")


def _node(module, *, backend) -> _FakeNode:
    node = module.FaAsrNode.__new__(module.FaAsrNode)
    node._logger = _FakeLogger()
    node.expected_source_id = "mic0"
    node.expected_stream_id = "stream0"
    node.target_sample_rate = 10
    node.min_audio_sec = 0.2
    node.timeline_clock = "media"
    node.timeline_window_id = "fa_asr_window"
    node.timeline_window_epoch = 7
    node.backend_name = "local_command"
    node.backend_kind = "asr"
    node.backend_model = "model-id"
    node.backend_model_path = "/models/asr.bin"
    node.backend_model_version = ""
    node.backend_model_revision = "rev1"
    node.backend = backend
    node.input_capability = backend.capability
    node._backend_lock = threading.Lock()
    node._timeline = _timeline(sample_rate=10, retention_sec=10.0)
    node._timeline_lock = threading.Lock()
    node._turn_state_lock = threading.RLock()
    node._context_active = False
    node._active_session_id = ""
    node._active_user_turn_id = 0
    node._payload_chunks = []
    node._buffer_sample_count = 0
    node._samples_lock = threading.Lock()
    node._jobs = queue.Queue()
    node._asr_state = module.AsrState.STATE_WAITING
    node._event_seq = 0
    node.asr_result_pub = _RecordingPublisher()
    node.asr_state_pub = _RecordingPublisher()
    node.asr_event_pub = _RecordingPublisher()
    node._trace_file = None
    node._trace_lock = threading.Lock()
    node.control_default_enabled = False
    node.control_configs = (
        module.ControlInputConfig(
            control_id="speech_control",
            action="topic",
            topic="voice/vad_state",
            msg_type="fa_interfaces/msg/VadState",
            source_id="mic0",
            stream_id="stream0",
            active_field="is_speech",
            start_field="start",
            end_field="end",
            open_on="start_or_active_rising",
            close_on="end_or_active_falling",
            submit_on_close=True,
            pre_roll_ms=0.0,
            post_roll_ms=0.0,
            qos_depth=50,
            qos_reliable=False,
        ),
    )
    node._control_windows = {"speech_control": module.ControlWindowState()}
    return node


def _response() -> _FakeTranscribeAudioResponse:
    return _FakeTranscribeAudioResponse()


def _timeline(
    *,
    sample_rate: int = 10,
    retention_sec: float = 10.0,
    timestamp_alignment_tolerance_ms: float = 1.0,
) -> RollingAsrTimeline:
    return RollingAsrTimeline(
        sample_rate=sample_rate,
        retention_sec=retention_sec,
        timestamp_alignment_tolerance_ms=timestamp_alignment_tolerance_ms,
    )


def test_numeric_time_range_parser_accepts_exact_numeric_range() -> None:
    time_range = parse_numeric_time_range("1000000000..1200000000")

    assert time_range.start_unix_ns == 1_000_000_000
    assert time_range.end_unix_ns == 1_200_000_000


@pytest.mark.parametrize(
    "spec",
    [
        "now-10s..now",
        "action:start..action:end",
        "..100",
        "100..",
        "-1..100",
        "100..100",
        "200..100",
        " 100..200",
        "100..200 ",
        "\t100..200",
        "abc",
    ],
)
def test_numeric_time_range_parser_rejects_unsupported_ranges(spec: str) -> None:
    with pytest.raises(TimelineRangeError) as exc_info:
        parse_numeric_time_range(spec)

    assert exc_info.value.error_code == ERROR_TIME_RANGE_UNRESOLVED


@pytest.mark.parametrize(
    "timestamp_alignment_tolerance_ms",
    [-0.001, float("nan"), float("inf"), -float("inf")],
)
def test_timeline_constructor_rejects_invalid_timestamp_alignment_tolerance(
    timestamp_alignment_tolerance_ms: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timestamp_alignment_tolerance_ms must be finite and greater than or equal to zero",
    ):
        RollingAsrTimeline(
            sample_rate=16_000,
            retention_sec=10.0,
            timestamp_alignment_tolerance_ms=timestamp_alignment_tolerance_ms,
        )


def test_timeline_retains_configured_horizon_and_rejects_outside_range() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=1.0)
    timeline.append(
        start_unix_ns=1_000_000_000,
        samples=np.linspace(-0.95, 0.95, 20, dtype=np.float32),
    )

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1000000000..1500000000"))

    assert exc_info.value.error_code == ERROR_RANGE_OUTSIDE_WINDOW


def test_empty_timeline_refuses_requested_range() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=1.0)

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1000000000..1100000000"))

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND


def test_timeline_detects_gap_for_range_crossing_missing_audio() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([0.1, 0.2], dtype=np.float32))
    timeline.append(start_unix_ns=1_400_000_000, samples=np.array([0.5, 0.6], dtype=np.float32))

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1000000000..1500000000"))

    assert exc_info.value.error_code == ERROR_RANGE_NOT_CONTINUOUS


def test_timeline_reports_non_continuous_when_requested_start_is_inside_gap() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([0.1, 0.2], dtype=np.float32))
    timeline.append(start_unix_ns=1_500_000_000, samples=np.array([0.5, 0.6], dtype=np.float32))

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1300000000..1600000000"))

    assert exc_info.value.error_code == ERROR_RANGE_NOT_CONTINUOUS


def test_timeline_rejects_overlap_without_corrupting_prior_audio() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([0.1, 0.2], dtype=np.float32))

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.append(
            start_unix_ns=1_198_999_999,
            samples=np.array([0.9, 1.0], dtype=np.float32),
        )

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND
    timeline_slice = timeline.slice(parse_numeric_time_range("1000000000..1200000000"))
    np.testing.assert_array_equal(timeline_slice.samples, np.array([0.1, 0.2], dtype=np.float32))


def test_timeline_aligns_sub_sample_timestamp_overlap() -> None:
    timeline = _timeline(
        sample_rate=16_000,
        retention_sec=10.0,
        timestamp_alignment_tolerance_ms=0.4,
    )
    first_start_unix_ns = 1_000_000_000
    expected_second_start_unix_ns = 1_020_000_000

    timeline.append(
        start_unix_ns=first_start_unix_ns,
        samples=np.full(320, 0.25, dtype=np.float32),
    )
    timeline.append(
        start_unix_ns=expected_second_start_unix_ns - 352_334,
        samples=np.full(320, 0.75, dtype=np.float32),
    )
    timeline_slice = timeline.slice(
        parse_numeric_time_range(
            f"{first_start_unix_ns}..{first_start_unix_ns + 40_000_000}"
        )
    )

    assert timeline_slice.samples.size == 640
    np.testing.assert_array_equal(
        timeline_slice.samples[:320],
        np.full(320, 0.25, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        timeline_slice.samples[320:],
        np.full(320, 0.75, dtype=np.float32),
    )


def test_timeline_aligns_sub_sample_timestamp_gap() -> None:
    timeline = _timeline(
        sample_rate=16_000,
        retention_sec=10.0,
        timestamp_alignment_tolerance_ms=0.4,
    )
    first_start_unix_ns = 1_000_000_000
    expected_second_start_unix_ns = 1_020_000_000

    timeline.append(
        start_unix_ns=first_start_unix_ns,
        samples=np.full(320, 0.25, dtype=np.float32),
    )
    timeline.append(
        start_unix_ns=expected_second_start_unix_ns + 337_014,
        samples=np.full(320, 0.75, dtype=np.float32),
    )
    timeline_slice = timeline.slice(
        parse_numeric_time_range(
            f"{first_start_unix_ns}..{first_start_unix_ns + 40_000_000}"
        )
    )

    assert timeline_slice.samples.size == 640
    np.testing.assert_array_equal(
        timeline_slice.samples[:320],
        np.full(320, 0.25, dtype=np.float32),
    )
    np.testing.assert_array_equal(
        timeline_slice.samples[320:],
        np.full(320, 0.75, dtype=np.float32),
    )


def test_timeline_rejects_overlap_beyond_alignment_tolerance() -> None:
    timeline = _timeline(
        sample_rate=16_000,
        retention_sec=10.0,
        timestamp_alignment_tolerance_ms=0.25,
    )
    first_start_unix_ns = 1_000_000_000
    expected_second_start_unix_ns = 1_020_000_000

    timeline.append(
        start_unix_ns=first_start_unix_ns,
        samples=np.full(320, 0.25, dtype=np.float32),
    )
    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.append(
            start_unix_ns=expected_second_start_unix_ns - 250_001,
            samples=np.full(320, 0.75, dtype=np.float32),
        )

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND


def test_timeline_reports_gap_beyond_configured_alignment_tolerance() -> None:
    timeline = _timeline(
        sample_rate=16_000,
        retention_sec=10.0,
        timestamp_alignment_tolerance_ms=0.25,
    )
    first_start_unix_ns = 1_000_000_000
    expected_second_start_unix_ns = 1_020_000_000

    timeline.append(
        start_unix_ns=first_start_unix_ns,
        samples=np.full(320, 0.25, dtype=np.float32),
    )
    timeline.append(
        start_unix_ns=expected_second_start_unix_ns + 250_001,
        samples=np.full(320, 0.75, dtype=np.float32),
    )

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(
            parse_numeric_time_range(
                f"{first_start_unix_ns}..{first_start_unix_ns + 40_000_000}"
            )
        )

    assert exc_info.value.error_code == ERROR_RANGE_NOT_CONTINUOUS


def test_asr_node_builds_timeline_with_configured_alignment_tolerance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = module.FaAsrNode.__new__(module.FaAsrNode)
    node.target_sample_rate = 16_000
    node.timeline_retention_sec = 10.0
    node.timeline_timestamp_alignment_tolerance_ms = 0.25
    timeline = module.FaAsrNode._build_timeline(node)
    first_start_unix_ns = 1_000_000_000
    expected_second_start_unix_ns = 1_020_000_000

    timeline.append(
        start_unix_ns=first_start_unix_ns,
        samples=np.full(320, 0.25, dtype=np.float32),
    )
    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.append(
            start_unix_ns=expected_second_start_unix_ns - 250_001,
            samples=np.full(320, 0.75, dtype=np.float32),
        )

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND


def test_timeline_slices_exact_values_across_contiguous_frames() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=10.0)
    timeline.append(
        start_unix_ns=1_000_000_000,
        samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )
    timeline.append(
        start_unix_ns=1_300_000_000,
        samples=np.array([0.4, 0.5], dtype=np.float32),
    )

    timeline_slice = timeline.slice(parse_numeric_time_range("1100000000..1500000000"))

    assert timeline_slice.samples.dtype == np.float32
    assert timeline_slice.samples.ndim == 1
    assert timeline_slice.samples.size == 4
    np.testing.assert_array_equal(
        timeline_slice.samples,
        np.array([0.2, 0.3, 0.4, 0.5], dtype=np.float32),
    )


def test_timeline_quantizes_non_sample_boundary_request_to_covering_sample_span() -> None:
    timeline = _timeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([0.1, 0.2], dtype=np.float32))

    timeline_slice = timeline.slice(parse_numeric_time_range("1050000000..1150000000"))

    assert timeline_slice.time_range.start_unix_ns == 1_000_000_000
    assert timeline_slice.time_range.end_unix_ns == 1_200_000_000
    np.testing.assert_array_equal(timeline_slice.samples, np.array([0.1, 0.2], dtype=np.float32))


def test_timeline_slices_non_exact_sample_rate_with_integer_coverage_range() -> None:
    timeline = _timeline(sample_rate=48_000, retention_sec=10.0)
    samples = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

    timeline.append(start_unix_ns=1_000_000_000, samples=samples)
    timeline_slice = timeline.slice(parse_numeric_time_range("1000020834..1000062499"))

    assert timeline.latest_end_unix_ns == 1_000_083_334
    assert timeline_slice.time_range.start_unix_ns == 1_000_020_833
    assert timeline_slice.time_range.end_unix_ns == 1_000_062_500
    assert timeline_slice.time_range.start_unix_ns <= 1_000_020_834
    assert timeline_slice.time_range.end_unix_ns >= 1_000_062_499
    np.testing.assert_array_equal(
        timeline_slice.samples,
        np.array([0.2, 0.3], dtype=np.float32),
    )


def test_timeline_accepts_quantized_floor_contiguous_append_at_48khz() -> None:
    timeline = _timeline(sample_rate=48_000, retention_sec=10.0)
    first_start_unix_ns = 1_000_000_000
    second_start_unix_ns = 1_000_020_833
    requested_end_unix_ns = 1_000_041_666

    timeline.append(start_unix_ns=first_start_unix_ns, samples=np.array([0.1], dtype=np.float32))
    timeline.append(
        start_unix_ns=second_start_unix_ns,
        samples=np.array([0.2, 0.3], dtype=np.float32),
    )
    timeline_slice = timeline.slice(
        parse_numeric_time_range(f"{first_start_unix_ns}..{requested_end_unix_ns}")
    )

    assert timeline_slice.time_range.start_unix_ns == first_start_unix_ns
    assert timeline_slice.time_range.end_unix_ns == 1_000_041_667
    assert timeline_slice.time_range.start_unix_ns <= first_start_unix_ns
    assert timeline_slice.time_range.end_unix_ns >= requested_end_unix_ns
    np.testing.assert_array_equal(
        timeline_slice.samples,
        np.array([0.1, 0.2], dtype=np.float32),
    )


def test_timeline_rejects_before_quantized_floor_boundary_without_corruption() -> None:
    timeline = _timeline(sample_rate=48_000, retention_sec=10.0)
    first_start_unix_ns = 1_000_000_000
    first_floor_end_unix_ns = 1_000_020_833

    timeline.append(start_unix_ns=first_start_unix_ns, samples=np.array([0.1], dtype=np.float32))
    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.append(
            start_unix_ns=first_floor_end_unix_ns - 1_000_001,
            samples=np.array([0.9], dtype=np.float32),
        )

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND
    timeline_slice = timeline.slice(
        parse_numeric_time_range(f"{first_start_unix_ns}..{first_floor_end_unix_ns}")
    )
    np.testing.assert_array_equal(timeline_slice.samples, np.array([0.1], dtype=np.float32))


def test_on_audio_buffers_valid_frame_without_turn_context_and_service_transcribes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)

    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1300000000"),
        _response(),
    )

    assert result.success is True
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_NONE
    assert len(result.segments) == 1
    assert result.segments[0].start_unix_ns == 1_000_000_000
    assert result.segments[0].end_unix_ns == 1_300_000_000
    assert result.segments[0].text == "hello"
    assert result.time_range.clock == "media"
    assert result.time_range.uncertainty_ns == 0
    assert result.time_range.uncertainty_reason == ""
    assert result.audio_window_ref.window_id == "fa_asr_window"
    assert result.audio_window_ref.window_epoch == 7
    assert result.audio_window_ref.source_id == "mic0"
    assert result.audio_window_ref.stream_id == "stream0"
    assert result.audio_window_ref.time_range.start_unix_ns == 1_000_000_000
    assert result.model_ref.backend_name == "local_command"
    assert result.model_ref.backend_kind == "asr"
    assert result.model_ref.model_id == "model-id"
    assert result.model_ref.model_path == "/models/asr.bin"
    assert result.model_ref.model_revision == "rev1"
    assert len(backend.requests) == 1
    assert backend.requests[0].samples.dtype == np.float32
    assert backend.requests[0].samples.ndim == 1
    np.testing.assert_array_equal(
        backend.requests[0].samples,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_transcribe_audio_quantizes_non_sample_boundary_request_to_selected_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1050000000..1250000000"),
        _response(),
    )

    assert result.success is True
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_NONE
    assert result.time_range.start_unix_ns == 1_000_000_000
    assert result.time_range.end_unix_ns == 1_300_000_000
    assert result.audio_window_ref.time_range.start_unix_ns == 1_000_000_000
    assert result.audio_window_ref.time_range.end_unix_ns == 1_300_000_000
    assert result.segments[0].start_unix_ns == 1_000_000_000
    assert result.segments[0].end_unix_ns == 1_300_000_000
    assert len(backend.requests) == 1
    np.testing.assert_array_equal(
        backend.requests[0].samples,
        np.array([0.1, 0.2, 0.3], dtype=np.float32),
    )


def test_transcribe_audio_maps_two_backend_segments_to_absolute_timestamps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend(
        segments=(
            AsrTranscriptSegment(
                start_sample=0,
                end_sample=1,
                text="hello",
                speaker_label="speaker-a",
            ),
            AsrTranscriptSegment(
                start_sample=1,
                end_sample=3,
                text="world",
            ),
        )
    )
    node = _node(module, backend=backend)
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1300000000"),
        _response(),
    )

    assert result.success is True
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_NONE
    assert len(result.segments) == 2
    assert result.segments[0].start_unix_ns == 1_000_000_000
    assert result.segments[0].end_unix_ns == 1_100_000_000
    assert result.segments[0].text == "hello"
    assert result.segments[0].speaker_label == "speaker-a"
    assert result.segments[1].start_unix_ns == 1_100_000_000
    assert result.segments[1].end_unix_ns == 1_300_000_000
    assert result.segments[1].text == "world"
    assert result.segments[1].speaker_label == ""


def test_backend_invalid_segment_range_fails_service_without_bad_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend(
        segments=(
            AsrTranscriptSegment(
                start_sample=0,
                end_sample=4,
                text="too far",
            ),
        )
    )
    node = _node(module, backend=backend)
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1300000000"),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_TRANSCRIBE_FAILED
    assert result.segments == []
    assert "end_sample exceeds request sample count" in result.message


def test_invalid_audio_frame_contract_is_not_admitted_to_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    frame = _FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=1)
    frame.encoding = "PCM16LE"

    node.on_audio(frame)
    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1200000000"),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_WINDOW_NOT_FOUND
    assert backend.requests == []


def test_invalid_timestamp_is_not_admitted_to_timeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))

    node.on_audio(_FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=0))
    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1..200000000"),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_WINDOW_NOT_FOUND


def test_transcribe_audio_gap_returns_range_not_continuous_and_does_not_call_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node.on_audio(_FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=1))
    node.on_audio(_FakeAudioFrame(samples=np.array([0.5, 0.6], dtype=np.float32), sec=1, nanosec=400_000_000))

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1050000000..1450000000"),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_RANGE_NOT_CONTINUOUS
    assert backend.requests == []


def test_transcribe_audio_outside_retained_window_does_not_call_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node._timeline = _timeline(sample_rate=10, retention_sec=0.2)
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1100000000"),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_RANGE_OUTSIDE_WINDOW
    assert backend.requests == []


def test_active_turn_invalid_timestamp_is_not_buffered_or_submitted_on_control_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9

    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=0,
        )
    )
    assert node._payload_chunks == []
    assert node._buffer_sample_count == 0

    event = module.ControlEvent(
        control_id="speech_control",
        source_id="mic0",
        stream_id="stream0",
        active=False,
        start=False,
        end=True,
        stamp_unix_ns=1_300_000_000,
    )
    node.on_control_event(event)

    assert node._payload_chunks == []
    assert node._buffer_sample_count == 0
    assert node._jobs.empty()
    assert backend.requests == []


def test_control_default_disabled_does_not_submit_from_audio_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node.control_default_enabled = False
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9

    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    assert node._payload_chunks == []
    assert node._buffer_sample_count == 0
    assert node._jobs.empty()
    assert backend.requests == []


def test_control_source_stream_mismatch_does_not_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            sec=1,
        )
    )
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic1",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_300_000_000,
        )
    )

    assert node._jobs.empty()
    assert backend.requests == []
    assert node._logger.error_records == [
        "Dropping control event speech_control: source_id/stream_id mismatch"
    ]


def test_control_close_slices_timeline_and_submits_selected_asr_ready_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            sec=1,
        )
    )

    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=True,
            end=False,
            stamp_unix_ns=1_100_000_000,
        )
    )
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_400_000_000,
        )
    )

    assert node._jobs.qsize() == 1
    job = node._jobs.get_nowait()
    assert job.session_id == "session-1"
    assert job.user_turn_id == 9
    assert job.reason == "control:speech_control:close"
    assert job.payload.sample_rate_hz == 10
    assert job.payload.sample_count == 3
    np.testing.assert_array_equal(
        job.payload.float32_samples(),
        np.array([0.2, 0.3, 0.4], dtype=np.float32),
    )
    assert backend.requests == []


def test_streaming_control_open_starts_backend_stream_before_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingStreamingBackend(partial_text="")
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9

    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=True,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )

    assert backend.start_calls == [("session-1", 9)]
    assert len(backend.streams) == 1
    assert backend.streams[0].finish_called is False
    assert node._jobs.empty()
    assert [msg.event for msg in node.asr_event_pub.messages] == [
        module.AsrEvent.EVENT_CONTROL_RECEIVED,
        module.AsrEvent.EVENT_CONTROL_WINDOW_OPENED,
        module.AsrEvent.EVENT_STREAM_OPENED,
    ]


def test_streaming_audio_pushes_incrementally_and_publishes_partial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingStreamingBackend(partial_text="partial text")
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=True,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )

    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2], dtype=np.float32),
            sec=1,
        )
    )

    stream = backend.streams[0]
    assert len(stream.pushed_payloads) == 1
    np.testing.assert_array_equal(
        stream.pushed_payloads[0].float32_samples(),
        np.array([0.1, 0.2], dtype=np.float32),
    )
    assert node._jobs.empty()
    assert len(node.asr_result_pub.messages) == 1
    assert node.asr_result_pub.messages[0].status == module.AsrResult.STATUS_PARTIAL
    assert node.asr_result_pub.messages[0].text == "partial text"
    assert stream.finish_called is False
    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_STREAM_AUDIO_PUSHED in event_codes
    assert module.AsrEvent.EVENT_STREAM_PARTIAL_RESULT in event_codes


def test_streaming_control_close_finishes_stream_and_publishes_final(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingStreamingBackend(
        partial_text="partial text",
        final_text="final text",
    )
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=True,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2], dtype=np.float32),
            sec=1,
        )
    )

    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_200_000_000,
        )
    )

    stream = backend.streams[0]
    assert stream.finish_called is True
    assert node._jobs.empty()
    assert [(msg.status, msg.text) for msg in node.asr_result_pub.messages] == [
        (module.AsrResult.STATUS_PARTIAL, "partial text"),
        (module.AsrResult.STATUS_FINAL, "final text"),
    ]
    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_STREAM_FINAL_RESULT in event_codes
    assert module.AsrEvent.EVENT_STREAM_CLOSED in event_codes


def test_streaming_control_close_allows_empty_final_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingStreamingBackend(
        partial_text="",
        final_text="",
    )
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=True,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2], dtype=np.float32),
            sec=1,
        )
    )

    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_200_000_000,
        )
    )

    stream = backend.streams[0]
    assert stream.finish_called is True
    assert node._jobs.empty()
    assert [(msg.status, msg.text) for msg in node.asr_result_pub.messages] == [
        (module.AsrResult.STATUS_FINAL, ""),
    ]
    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_STREAM_FINAL_RESULT in event_codes
    assert module.AsrEvent.EVENT_STREAM_CLOSED in event_codes
    assert module.AsrEvent.EVENT_STREAM_ERROR not in event_codes


def test_transcribe_audio_stays_timeline_batch_for_streaming_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingStreamingBackend(partial_text="")
    node = _node(module, backend=backend)
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1300000000"),
        _response(),
    )

    assert result.success is True
    assert result.segments[0].text == "timeline transcript"
    assert backend.start_calls == []
    assert len(backend.requests) == 1


def test_streaming_backend_pcm16_requirement_rejects_float32_frame_without_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingStreamingBackend(audio_encoding="PCM16LE", partial_text="")
    node = _node(module, backend=backend)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=True,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )

    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2], dtype=np.float32),
            sec=1,
        )
    )

    assert len(backend.streams) == 1
    assert backend.streams[0].pushed_payloads == []
    assert node.asr_result_pub.messages == []
    assert node.asr_event_pub.messages[-1].event == (
        module.AsrEvent.EVENT_INVALID_AUDIO_FRAME_DROPPED
    )
    assert node._logger.error_records[-1] == (
        "Dropping invalid AudioFrame: AudioFrame encoding must be PCM16LE, got FLOAT32LE"
    )


def test_control_window_observability_reports_open_close_and_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
            sec=1,
        )
    )

    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=True,
            start=True,
            end=False,
            stamp_unix_ns=1_100_000_000,
        )
    )
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_400_000_000,
        )
    )

    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert event_codes == [
        module.AsrEvent.EVENT_CONTROL_RECEIVED,
        module.AsrEvent.EVENT_CONTROL_WINDOW_OPENED,
        module.AsrEvent.EVENT_CONTROL_RECEIVED,
        module.AsrEvent.EVENT_CONTROL_WINDOW_CLOSED,
        module.AsrEvent.EVENT_JOB_QUEUED,
    ]
    assert [msg.event_seq for msg in node.asr_event_pub.messages] == [1, 2, 3, 4, 5]
    assert [
        (msg.state_before, msg.state_after)
        for msg in node.asr_event_pub.messages
    ] == [
        (module.AsrState.STATE_WAITING, module.AsrState.STATE_WAITING),
        (module.AsrState.STATE_WAITING, module.AsrState.STATE_COLLECTING),
        (module.AsrState.STATE_COLLECTING, module.AsrState.STATE_COLLECTING),
        (module.AsrState.STATE_COLLECTING, module.AsrState.STATE_WAITING),
        (module.AsrState.STATE_WAITING, module.AsrState.STATE_QUEUED),
    ]
    assert node.asr_event_pub.messages[-1].state == module.AsrState.STATE_QUEUED
    assert node.asr_event_pub.messages[-1].state_after == module.AsrState.STATE_QUEUED
    assert node.asr_event_pub.messages[-1].sample_count == 3
    assert node.asr_state_pub.messages[-1].reason == "job_queued"


def test_control_skip_observability_reports_no_window_inactive_and_disabled_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_100_000_000,
        )
    )

    node._context_active = False
    node._active_session_id = ""
    node._control_windows["speech_control"].open = True
    node._control_windows["speech_control"].start_unix_ns = 1_100_000_000
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_400_000_000,
        )
    )

    disabled_submit_config = module.ControlInputConfig(
        control_id="manual_control",
        action="topic",
        topic="voice/vad_state",
        msg_type="fa_interfaces/msg/VadState",
        source_id="mic0",
        stream_id="stream0",
        active_field="is_speech",
        start_field="start",
        end_field="end",
        open_on="start_or_active_rising",
        close_on="end_or_active_falling",
        submit_on_close=False,
        pre_roll_ms=0.0,
        post_roll_ms=0.0,
        qos_depth=50,
        qos_reliable=False,
    )
    disabled_submit_window = module.ControlWindowState(open=True, start_unix_ns=1_100_000_000)
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    node._close_control_window(
        disabled_submit_config,
        disabled_submit_window,
        module.ControlEvent(
            control_id="manual_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_400_000_000,
        ),
    )

    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_CLOSE_IGNORED_NO_WINDOW in event_codes
    assert module.AsrEvent.EVENT_SUBMIT_SKIPPED_CONTEXT_INACTIVE in event_codes
    assert module.AsrEvent.EVENT_SUBMIT_SKIPPED_SUBMIT_ON_CLOSE_FALSE in event_codes


def test_invalid_close_time_observability_reports_rejected_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9
    config = node.control_configs[0]
    window = module.ControlWindowState(open=True, start_unix_ns=1_400_000_000)

    node._close_control_window(
        config,
        window,
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_100_000_000,
        ),
    )

    assert node.asr_event_pub.messages[-1].event == module.AsrEvent.EVENT_INVALID_CLOSE_TIME
    assert node.asr_event_pub.messages[-1].error_code == ERROR_TIME_RANGE_UNRESOLVED


def test_control_close_without_active_context_does_not_submit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=True,
            end=False,
            stamp_unix_ns=1_000_000_000,
        )
    )
    node.on_control_event(
        module.ControlEvent(
            control_id="speech_control",
            source_id="mic0",
            stream_id="stream0",
            active=False,
            start=False,
            end=True,
            stamp_unix_ns=1_300_000_000,
        )
    )

    assert node._jobs.empty()
    assert backend.requests == []


def test_unsupported_audio_scope_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(
            time_range_spec="1000000000..1200000000",
            audio_scope="other_stream",
        ),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_UNSUPPORTED_AUDIO_SCOPE


def test_exact_stream_audio_scope_is_supported(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
            sec=1,
        )
    )

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(
            time_range_spec="1000000000..1300000000",
            audio_scope="stream0",
        ),
        _response(),
    )

    assert result.success is True
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_NONE


def test_backend_timeout_error_and_blank_transcript_fail_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    cases = [
        (_TimeoutBackend(), _FakeTranscribeAudioResponse.ERROR_TRANSCRIBE_FAILED),
        (_ErrorBackend(), _FakeTranscribeAudioResponse.ERROR_TRANSCRIBE_FAILED),
        (_RecordingBackend("  "), _FakeTranscribeAudioResponse.ERROR_TRANSCRIBE_FAILED),
    ]

    for backend, expected_error in cases:
        node = _node(module, backend=backend)
        node.on_audio(
            _FakeAudioFrame(
                samples=np.array([0.1, 0.2, 0.3], dtype=np.float32),
                sec=1,
            )
        )

        result = node.handle_transcribe_audio(
            _FakeTranscribeAudioRequest(time_range_spec="1000000000..1300000000"),
            _response(),
        )

        assert result.success is False
        assert result.error_code == expected_error


def test_too_short_requested_audio_fails_before_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    backend = _RecordingBackend("hello")
    node = _node(module, backend=backend)
    node.min_audio_sec = 0.3
    node.on_audio(_FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=1))

    result = node.handle_transcribe_audio(
        _FakeTranscribeAudioRequest(time_range_spec="1000000000..1200000000"),
        _response(),
    )

    assert result.success is False
    assert result.error_code == _FakeTranscribeAudioResponse.ERROR_TRANSCRIBE_FAILED
    assert backend.requests == []


def test_audio_drop_observability_reports_invalid_frame_and_timeline_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))
    node._asr_state = module.AsrState.STATE_COLLECTING
    invalid_frame = _FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=1)
    invalid_frame.encoding = "PCM16LE"

    node.on_audio(invalid_frame)
    node.on_audio(_FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=2))
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.3, 0.4], dtype=np.float32),
            sec=2,
            nanosec=99_000_000,
        )
    )

    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_INVALID_AUDIO_FRAME_DROPPED in event_codes
    assert module.AsrEvent.EVENT_TIMELINE_OVERLAP_DERIVED_FAILURE in event_codes
    assert [msg.event_seq for msg in node.asr_event_pub.messages] == [1, 2]
    assert [
        (msg.state_before, msg.state_after)
        for msg in node.asr_event_pub.messages
    ] == [
        (module.AsrState.STATE_COLLECTING, module.AsrState.STATE_COLLECTING),
        (module.AsrState.STATE_COLLECTING, module.AsrState.STATE_COLLECTING),
    ]
    assert node._asr_state == module.AsrState.STATE_COLLECTING


def test_timeline_slice_observability_reports_unavailable_not_continuous_and_short(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    node = _node(module, backend=_RecordingBackend("hello"))
    node._context_active = True
    node._active_session_id = "session-1"
    node._active_user_turn_id = 9

    node._submit_timeline_window(
        start_unix_ns=1_000_000_000,
        end_unix_ns=1_100_000_000,
        reason="empty_timeline",
    )
    node.on_audio(_FakeAudioFrame(samples=np.array([0.1, 0.2], dtype=np.float32), sec=1))
    node.on_audio(
        _FakeAudioFrame(
            samples=np.array([0.5, 0.6], dtype=np.float32),
            sec=1,
            nanosec=400_000_000,
        )
    )
    node._submit_timeline_window(
        start_unix_ns=1_000_000_000,
        end_unix_ns=1_500_000_000,
        reason="gap_timeline",
    )
    node.min_audio_sec = 0.3
    node._submit_timeline_window(
        start_unix_ns=1_000_000_000,
        end_unix_ns=1_100_000_000,
        reason="short_window",
    )

    event_codes = [msg.event for msg in node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_TIMELINE_SLICE_UNAVAILABLE in event_codes
    assert module.AsrEvent.EVENT_TIMELINE_SLICE_NOT_CONTINUOUS in event_codes
    assert module.AsrEvent.EVENT_WINDOW_TOO_SHORT in event_codes


def test_backend_observability_reports_started_completed_timeout_error_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _asr_node_module(monkeypatch)
    success_node = _node(module, backend=_RecordingBackend("hello"))
    success_node._run_transcription(
        module.TranscriptionJob(
            session_id="session-1",
            user_turn_id=9,
            payload=module.AsrAudioPayload.from_float32_samples(
                np.array([0.1, 0.2], dtype=np.float32),
                sample_rate_hz=10,
            ),
            reason="control:speech_control:close",
        )
    )
    success_codes = [msg.event for msg in success_node.asr_event_pub.messages]
    assert success_codes == [
        module.AsrEvent.EVENT_BACKEND_TRANSCRIPTION_STARTED,
        module.AsrEvent.EVENT_BACKEND_COMPLETED,
    ]

    timeout_node = _node(module, backend=_TimeoutBackend())
    timeout_node._run_transcription(
        module.TranscriptionJob(
            session_id="session-1",
            user_turn_id=9,
            payload=module.AsrAudioPayload.from_float32_samples(
                np.array([0.1, 0.2], dtype=np.float32),
                sample_rate_hz=10,
            ),
            reason="control:speech_control:close",
        )
    )
    timeout_codes = [msg.event for msg in timeout_node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_BACKEND_TIMEOUT in timeout_codes
    assert module.AsrEvent.EVENT_FAIL_CLOSED in timeout_codes

    error_node = _node(module, backend=_ErrorBackend())
    error_node._run_transcription(
        module.TranscriptionJob(
            session_id="session-1",
            user_turn_id=9,
            payload=module.AsrAudioPayload.from_float32_samples(
                np.array([0.1, 0.2], dtype=np.float32),
                sample_rate_hz=10,
            ),
            reason="control:speech_control:close",
        )
    )
    error_codes = [msg.event for msg in error_node.asr_event_pub.messages]
    assert module.AsrEvent.EVENT_BACKEND_ERROR in error_codes
    assert module.AsrEvent.EVENT_FAIL_CLOSED in error_codes


def test_trace_file_is_append_only_jsonl_and_unwritable_path_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _asr_node_module(monkeypatch)
    trace_path = tmp_path / "asr-trace.jsonl"
    trace_file = module.FaAsrNode._open_trace_file(
        enabled=True,
        trace_path=str(trace_path),
    )
    node = _node(module, backend=_RecordingBackend("hello"))
    node._trace_file = trace_file
    node._emit_event(module.AsrEvent.EVENT_STARTUP_IDLE, "startup_idle")
    trace_file.close()

    lines = trace_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["event_seq"] == 1
    assert record["event"] == module.AsrEvent.EVENT_STARTUP_IDLE
    assert record["state"] == module.AsrState.STATE_WAITING
    assert record["state_before"] == module.AsrState.STATE_WAITING
    assert record["state_after"] == module.AsrState.STATE_WAITING
    assert record["reason"] == "startup_idle"

    with pytest.raises(RuntimeError, match="trace.path is not writable"):
        module.FaAsrNode._open_trace_file(enabled=True, trace_path=str(tmp_path))
