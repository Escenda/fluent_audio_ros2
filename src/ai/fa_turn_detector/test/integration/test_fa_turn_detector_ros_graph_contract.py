from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import sys
import time

import numpy as np
import pytest

rclpy = pytest.importorskip("rclpy")
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AudioFrame, TurnContext, TurnEnd, VadState
from fa_turn_detector_py.turn_detector_node import FaTurnDetectorNode


PACKAGE_ROOT = Path(__file__).parents[2]
AUDIO_QOS_DEPTH = 10
AUDIO_QOS_RELIABLE = False
CONTROL_QOS_DEPTH = 10
CONTROL_QOS_RELIABLE = False
TURN_CONTEXT_QOS_DEPTH = 10
TURN_CONTEXT_QOS_RELIABLE = True
OUTPUT_QOS_DEPTH = 10
OUTPUT_QOS_RELIABLE = True


def _write_fake_model(path: Path, probability: str) -> None:
    path.write_text(probability + "\n" + ("x" * 2048), encoding="utf-8")


def _qos_profile(*, depth: int, reliable: bool) -> QoSProfile:
    qos = QoSProfile(depth=depth)
    qos.reliability = (
        ReliabilityPolicy.RELIABLE if reliable else ReliabilityPolicy.BEST_EFFORT
    )
    qos.history = HistoryPolicy.KEEP_LAST
    return qos


def _spin_until(
    executor: rclpy.executors.SingleThreadedExecutor,
    predicate: Callable[[], bool],
    *,
    timeout_sec: float = 3.0,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if predicate():
            return True
        executor.spin_once(timeout_sec=0.01)
    return predicate()


def _audio_frame(stream_id: str, samples: np.ndarray) -> AudioFrame:
    frame = AudioFrame()
    frame.header.frame_id = "mic-a-frame"
    frame.source_id = "mic-a"
    frame.stream_id = stream_id
    frame.encoding = "FLOAT32LE"
    frame.sample_rate = 16000
    frame.channels = 1
    frame.bit_depth = 32
    frame.layout = "interleaved"
    frame.data = samples.astype(np.float32, copy=False).tobytes()
    frame.epoch = 1
    return frame


def _active_context(session_id: str, user_turn_id: int) -> TurnContext:
    msg = TurnContext()
    msg.session_id = session_id
    msg.user_turn_id = user_turn_id
    msg.active = True
    return msg


def _vad_end(source_id: str, stream_id: str) -> VadState:
    msg = VadState()
    msg.source_id = source_id
    msg.stream_id = stream_id
    msg.probability = 0.0
    msg.is_speech = False
    msg.start = False
    msg.end = True
    return msg


def _parameter_overrides(
    *,
    tmp_path: Path,
    model_path: Path,
    audio_topic: str,
    expected_stream_id: str,
    control_topic: str,
    turn_context_topic: str,
    output_topic: str,
) -> list[Parameter]:
    worker_path = PACKAGE_ROOT / "test" / "fixtures" / "fake_turn_worker.py"
    return [
        Parameter("audio_topic", Parameter.Type.STRING, audio_topic),
        Parameter("expected_stream_id", Parameter.Type.STRING, expected_stream_id),
        Parameter("turn_context_topic", Parameter.Type.STRING, turn_context_topic),
        Parameter("output_topic", Parameter.Type.STRING, output_topic),
        Parameter("expected_source_id", Parameter.Type.STRING, "mic-a"),
        Parameter("control.default_enabled", Parameter.Type.BOOL, False),
        Parameter("control.inputs", Parameter.Type.STRING_ARRAY, ["speech_control"]),
        Parameter("control.speech_control.action", Parameter.Type.STRING, "topic"),
        Parameter("control.speech_control.topic", Parameter.Type.STRING, control_topic),
        Parameter(
            "control.speech_control.msg_type",
            Parameter.Type.STRING,
            "fa_interfaces/msg/VadState",
        ),
        Parameter("control.speech_control.source_id", Parameter.Type.STRING, "mic-a"),
        Parameter(
            "control.speech_control.stream_id",
            Parameter.Type.STRING,
            expected_stream_id,
        ),
        Parameter("control.speech_control.active_field", Parameter.Type.STRING, "is_speech"),
        Parameter("control.speech_control.start_field", Parameter.Type.STRING, "start"),
        Parameter("control.speech_control.end_field", Parameter.Type.STRING, "end"),
        Parameter(
            "control.speech_control.close_on",
            Parameter.Type.STRING,
            "end_or_active_falling",
        ),
        Parameter("control.speech_control.qos.depth", Parameter.Type.INTEGER, CONTROL_QOS_DEPTH),
        Parameter(
            "control.speech_control.qos.reliable",
            Parameter.Type.BOOL,
            CONTROL_QOS_RELIABLE,
        ),
        Parameter("backend.name", Parameter.Type.STRING, "smart_turn_onnx"),
        Parameter("backend.model_path", Parameter.Type.STRING, str(model_path)),
        Parameter("backend.threshold", Parameter.Type.DOUBLE, 0.5),
        Parameter("backend.execution_provider", Parameter.Type.STRING, "CPUExecutionProvider"),
        Parameter("backend.command", Parameter.Type.STRING, sys.executable),
        Parameter(
            "backend.args",
            Parameter.Type.STRING_ARRAY,
            [
                str(worker_path),
                "detect",
                "--audio",
                "{audio}",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
            ],
        ),
        Parameter(
            "backend.health_args",
            Parameter.Type.STRING_ARRAY,
            [
                str(worker_path),
                "health",
                "--model",
                "{model}",
                "--provider",
                "{provider}",
            ],
        ),
        Parameter("backend.timeout_sec", Parameter.Type.DOUBLE, 1.0),
        Parameter("backend.workspace_dir", Parameter.Type.STRING, str(tmp_path / "workspace")),
        Parameter("backend.cleanup_audio_files", Parameter.Type.BOOL, True),
        Parameter("audio.qos.depth", Parameter.Type.INTEGER, AUDIO_QOS_DEPTH),
        Parameter("audio.qos.reliable", Parameter.Type.BOOL, AUDIO_QOS_RELIABLE),
        Parameter("turn_context.qos.depth", Parameter.Type.INTEGER, TURN_CONTEXT_QOS_DEPTH),
        Parameter("turn_context.qos.reliable", Parameter.Type.BOOL, TURN_CONTEXT_QOS_RELIABLE),
        Parameter("output.qos.depth", Parameter.Type.INTEGER, OUTPUT_QOS_DEPTH),
        Parameter("output.qos.reliable", Parameter.Type.BOOL, OUTPUT_QOS_RELIABLE),
    ]


def test_turn_detector_publishes_turn_end_from_ros_graph_fixture(tmp_path: Path) -> None:
    if not rclpy.ok():
        rclpy.init()

    suffix = str(time.time_ns())
    audio_topic = f"audio/test/turn_detector_audio_{suffix}"
    expected_stream_id = f"audio/raw/turn_detector_{suffix}"
    control_topic = f"voice/test/turn_detector_control_{suffix}"
    turn_context_topic = f"conversation/test/turn_context_{suffix}"
    output_topic = f"voice/test/turn_end_{suffix}"
    model_path = tmp_path / "smart_turn.onnx"
    _write_fake_model(model_path, "0.75")

    node = FaTurnDetectorNode(
        parameter_overrides=_parameter_overrides(
            tmp_path=tmp_path,
            model_path=model_path,
            audio_topic=audio_topic,
            expected_stream_id=expected_stream_id,
            control_topic=control_topic,
            turn_context_topic=turn_context_topic,
            output_topic=output_topic,
        )
    )
    io_node = rclpy.create_node(f"fa_turn_detector_graph_contract_io_{suffix}")
    received: list[TurnEnd] = []
    audio_pub = io_node.create_publisher(
        AudioFrame,
        audio_topic,
        _qos_profile(depth=AUDIO_QOS_DEPTH, reliable=AUDIO_QOS_RELIABLE),
    )
    vad_pub = io_node.create_publisher(
        VadState,
        control_topic,
        _qos_profile(depth=CONTROL_QOS_DEPTH, reliable=CONTROL_QOS_RELIABLE),
    )
    context_pub = io_node.create_publisher(
        TurnContext,
        turn_context_topic,
        _qos_profile(depth=TURN_CONTEXT_QOS_DEPTH, reliable=TURN_CONTEXT_QOS_RELIABLE),
    )
    subscription = io_node.create_subscription(
        TurnEnd,
        output_topic,
        lambda msg: received.append(msg),
        _qos_profile(depth=OUTPUT_QOS_DEPTH, reliable=OUTPUT_QOS_RELIABLE),
    )
    assert subscription is not None

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(node)
    executor.add_node(io_node)
    try:
        assert _spin_until(
            executor,
            lambda: (
                io_node.count_subscribers(audio_topic) > 0
                and io_node.count_subscribers(control_topic) > 0
                and io_node.count_subscribers(turn_context_topic) > 0
                and io_node.count_publishers(output_topic) > 0
            ),
        )

        context_pub.publish(_active_context("session-a", 42))
        assert _spin_until(executor, lambda: node._context_active)

        audio_pub.publish(
            _audio_frame(expected_stream_id, np.full(16000, 0.1, dtype=np.float32))
        )
        assert _spin_until(executor, lambda: len(node.audio_buffer) >= 16000)

        vad_pub.publish(_vad_end("mic-b", expected_stream_id))
        assert not _spin_until(executor, lambda: len(received) == 1, timeout_sec=0.3)

        vad_pub.publish(_vad_end("mic-a", expected_stream_id))
        assert _spin_until(executor, lambda: len(received) == 1)

        assert received[0].session_id == "session-a"
        assert received[0].user_turn_id == 42
        assert received[0].probability == pytest.approx(0.75)
        assert received[0].is_end is True
    finally:
        executor.remove_node(io_node)
        executor.remove_node(node)
        io_node.destroy_node()
        node.destroy_node()
        rclpy.shutdown()
