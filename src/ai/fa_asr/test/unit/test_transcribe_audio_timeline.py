import importlib
import queue
import sys
import threading
from types import ModuleType

import numpy as np
import pytest

from fa_asr_py.backends.base import (
    AsrRequest,
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

    def error(self, message: str) -> None:
        self.error_records.append(message)

    def debug(self, message: str) -> None:
        self.debug_records.append(message)


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
    pass


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
            sample_count=int(request.samples.size),
        )


class _TimeoutBackend:
    name = "timeout_backend"

    def transcribe(self, request: AsrRequest) -> AsrTranscript:
        del request
        raise TimeoutError("backend timed out")


class _ErrorBackend:
    name = "error_backend"

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
    node._backend_lock = threading.Lock()
    node._timeline = RollingAsrTimeline(sample_rate=10, retention_sec=10.0)
    node._timeline_lock = threading.Lock()
    node._turn_state_lock = threading.RLock()
    node._context_active = False
    node._active_session_id = ""
    node._active_user_turn_id = 0
    node._samples = []
    node._samples_lock = threading.Lock()
    node._jobs = queue.Queue()
    node.finalize_on_vad_end = True
    return node


def _response() -> _FakeTranscribeAudioResponse:
    return _FakeTranscribeAudioResponse()


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


def test_timeline_retains_configured_horizon_and_rejects_outside_range() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=1.0)
    timeline.append(
        start_unix_ns=1_000_000_000,
        samples=np.arange(20, dtype=np.float32),
    )

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1000000000..1500000000"))

    assert exc_info.value.error_code == ERROR_RANGE_OUTSIDE_WINDOW


def test_empty_timeline_refuses_requested_range() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=1.0)

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1000000000..1100000000"))

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND


def test_timeline_detects_gap_for_range_crossing_missing_audio() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([1.0, 2.0], dtype=np.float32))
    timeline.append(start_unix_ns=1_400_000_000, samples=np.array([5.0, 6.0], dtype=np.float32))

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1000000000..1500000000"))

    assert exc_info.value.error_code == ERROR_RANGE_NOT_CONTINUOUS


def test_timeline_reports_non_continuous_when_requested_start_is_inside_gap() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([1.0, 2.0], dtype=np.float32))
    timeline.append(start_unix_ns=1_500_000_000, samples=np.array([5.0, 6.0], dtype=np.float32))

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.slice(parse_numeric_time_range("1300000000..1600000000"))

    assert exc_info.value.error_code == ERROR_RANGE_NOT_CONTINUOUS


def test_timeline_rejects_overlap_without_corrupting_prior_audio() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([1.0, 2.0], dtype=np.float32))

    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.append(
            start_unix_ns=1_100_000_000,
            samples=np.array([9.0, 10.0], dtype=np.float32),
        )

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND
    timeline_slice = timeline.slice(parse_numeric_time_range("1000000000..1200000000"))
    np.testing.assert_array_equal(timeline_slice.samples, np.array([1.0, 2.0], dtype=np.float32))


def test_timeline_slices_exact_values_across_contiguous_frames() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=10.0)
    timeline.append(
        start_unix_ns=1_000_000_000,
        samples=np.array([1.0, 2.0, 3.0], dtype=np.float32),
    )
    timeline.append(
        start_unix_ns=1_300_000_000,
        samples=np.array([4.0, 5.0], dtype=np.float32),
    )

    timeline_slice = timeline.slice(parse_numeric_time_range("1100000000..1500000000"))

    assert timeline_slice.samples.dtype == np.float32
    assert timeline_slice.samples.ndim == 1
    assert timeline_slice.samples.size == 4
    np.testing.assert_array_equal(
        timeline_slice.samples,
        np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32),
    )


def test_timeline_quantizes_non_sample_boundary_request_to_covering_sample_span() -> None:
    timeline = RollingAsrTimeline(sample_rate=10, retention_sec=10.0)
    timeline.append(start_unix_ns=1_000_000_000, samples=np.array([1.0, 2.0], dtype=np.float32))

    timeline_slice = timeline.slice(parse_numeric_time_range("1050000000..1150000000"))

    assert timeline_slice.time_range.start_unix_ns == 1_000_000_000
    assert timeline_slice.time_range.end_unix_ns == 1_200_000_000
    np.testing.assert_array_equal(timeline_slice.samples, np.array([1.0, 2.0], dtype=np.float32))


def test_timeline_slices_non_exact_sample_rate_with_integer_coverage_range() -> None:
    timeline = RollingAsrTimeline(sample_rate=48_000, retention_sec=10.0)
    samples = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

    timeline.append(start_unix_ns=1_000_000_000, samples=samples)
    timeline_slice = timeline.slice(parse_numeric_time_range("1000020834..1000062499"))

    assert timeline.latest_end_unix_ns == 1_000_083_334
    assert timeline_slice.time_range.start_unix_ns == 1_000_020_833
    assert timeline_slice.time_range.end_unix_ns == 1_000_062_500
    assert timeline_slice.time_range.start_unix_ns <= 1_000_020_834
    assert timeline_slice.time_range.end_unix_ns >= 1_000_062_499
    np.testing.assert_array_equal(
        timeline_slice.samples,
        np.array([20.0, 30.0], dtype=np.float32),
    )


def test_timeline_accepts_quantized_floor_contiguous_append_at_48khz() -> None:
    timeline = RollingAsrTimeline(sample_rate=48_000, retention_sec=10.0)
    first_start_unix_ns = 1_000_000_000
    second_start_unix_ns = 1_000_020_833
    requested_end_unix_ns = 1_000_041_666

    timeline.append(start_unix_ns=first_start_unix_ns, samples=np.array([10.0], dtype=np.float32))
    timeline.append(
        start_unix_ns=second_start_unix_ns,
        samples=np.array([20.0, 30.0], dtype=np.float32),
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
        np.array([10.0, 20.0], dtype=np.float32),
    )


def test_timeline_rejects_before_quantized_floor_boundary_without_corruption() -> None:
    timeline = RollingAsrTimeline(sample_rate=48_000, retention_sec=10.0)
    first_start_unix_ns = 1_000_000_000
    first_floor_end_unix_ns = 1_000_020_833

    timeline.append(start_unix_ns=first_start_unix_ns, samples=np.array([10.0], dtype=np.float32))
    with pytest.raises(TimelineRangeError) as exc_info:
        timeline.append(
            start_unix_ns=first_floor_end_unix_ns - 1,
            samples=np.array([90.0], dtype=np.float32),
        )

    assert exc_info.value.error_code == ERROR_WINDOW_NOT_FOUND
    timeline_slice = timeline.slice(
        parse_numeric_time_range(f"{first_start_unix_ns}..{first_floor_end_unix_ns}")
    )
    np.testing.assert_array_equal(timeline_slice.samples, np.array([10.0], dtype=np.float32))


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
    node._timeline = RollingAsrTimeline(sample_rate=10, retention_sec=0.2)
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


def test_active_turn_invalid_timestamp_is_not_buffered_or_submitted_on_vad_end(
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
    assert node._samples == []

    vad = _FakeVadState()
    vad.source_id = "mic0"
    vad.stream_id = "stream0"
    vad.is_speech = False
    vad.end = True
    node.on_vad(vad)

    assert node._samples == []
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
