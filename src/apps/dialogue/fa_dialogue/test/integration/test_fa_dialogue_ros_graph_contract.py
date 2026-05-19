from dataclasses import dataclass
import importlib.util
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from typing import Callable, TypeAlias
import uuid

import pytest
import yaml


YamlScalar: TypeAlias = str | int | float | bool | None
YamlMapping: TypeAlias = dict[str, "YamlValue"]
YamlSequence: TypeAlias = list["YamlValue"]
YamlValue: TypeAlias = YamlScalar | YamlMapping | YamlSequence


ROS2 = shutil.which("ros2")
if ROS2 is None:
    pytest.skip(
        "ros2 executable is required for fa_dialogue graph integration",
        allow_module_level=True,
    )
if importlib.util.find_spec("rclpy") is None:
    pytest.skip(
        "rclpy is required for fa_dialogue graph integration",
        allow_module_level=True,
    )
if (
    importlib.util.find_spec("fa_interfaces") is None
    or importlib.util.find_spec("fa_interfaces.msg") is None
):
    pytest.skip(
        "fa_interfaces generated messages are required for fa_dialogue graph integration",
        allow_module_level=True,
    )

import rclpy
from fa_interfaces.msg import AsrResult, TurnContext, TurnEnd, WakeWordResult
from rclpy.node import Node

TerminalMessageFactory: TypeAlias = Callable[[TurnContext], AsrResult | TurnEnd]


@dataclass(frozen=True)
class DialogueSmokeTopics:
    wake_word: str
    asr_result: str
    turn_end: str
    turn_context: str


def _write_yaml(path: Path, value: YamlValue) -> Path:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return path


def _make_topics(unique: str) -> DialogueSmokeTopics:
    return DialogueSmokeTopics(
        wake_word=f"/fa_dialogue_smoke/{unique}/wake_word",
        asr_result=f"/fa_dialogue_smoke/{unique}/asr_result",
        turn_end=f"/fa_dialogue_smoke/{unique}/turn_end",
        turn_context=f"/fa_dialogue_smoke/{unique}/turn_context",
    )


def _write_dialogue_config(
    tmp_path: Path,
    unique: str,
    topics: DialogueSmokeTopics,
) -> Path:
    return _write_yaml(
        tmp_path / "fa_dialogue.params.yaml",
        {
            "fa_dialogue": {
                "ros__parameters": {
                    "wake_word_topic": topics.wake_word,
                    "asr_result_topic": topics.asr_result,
                    "turn_end_topic": topics.turn_end,
                    "turn_context_topic": topics.turn_context,
                    "session_prefix": f"dialogue-smoke-{unique}-",
                    "wake.max_age_ms": 5000,
                    "wake.allow_zero_stamp": False,
                    "wake.qos.depth": 10,
                    "wake.qos.reliable": True,
                    "asr.qos.depth": 10,
                    "asr.qos.reliable": True,
                    "turn_end.qos.depth": 10,
                    "turn_end.qos.reliable": True,
                    "turn_context.qos.depth": 10,
                    "turn_context.qos.reliable": True,
                }
            }
        },
    )


def _stop_process(process: subprocess.Popen[str]) -> str:
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        stdout, _stderr = process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, _stderr = process.communicate(timeout=5)
    return stdout


def _assert_process_running(process: subprocess.Popen[str]) -> None:
    assert process.poll() is None, _stop_process(process)


def _start_process(command: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        command,
        env=os.environ.copy(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )


def _start_run_node(config_path: Path) -> subprocess.Popen[str]:
    assert ROS2 is not None
    return _start_process(
        [
            ROS2,
            "run",
            "fa_dialogue",
            "fa_dialogue_node",
            "--ros-args",
            "--params-file",
            str(config_path),
        ]
    )


def _start_launch_node(config_path: Path) -> subprocess.Popen[str]:
    assert ROS2 is not None
    return _start_process(
        [
            ROS2,
            "launch",
            "fa_dialogue",
            "fa_dialogue.launch.py",
            f"config_file:={config_path}",
        ]
    )


def _wait_for_subscription(
    node: Node,
    publisher,
    process: subprocess.Popen[str],
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _assert_process_running(process)
        if publisher.get_subscription_count() > 0:
            return True
        rclpy.spin_once(node, timeout_sec=0.02)
    return False


def _wait_for_publisher(
    node: Node,
    topic: str,
    process: subprocess.Popen[str],
    timeout_sec: float,
) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _assert_process_running(process)
        if node.count_publishers(topic) > 0:
            return True
        rclpy.spin_once(node, timeout_sec=0.02)
    return False


def _wait_for_context(
    node: Node,
    process: subprocess.Popen[str],
    received: list[TurnContext],
    matcher: Callable[[TurnContext], bool],
    timeout_sec: float,
) -> TurnContext | None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        _assert_process_running(process)
        for context in received:
            if matcher(context):
                return context
        rclpy.spin_once(node, timeout_sec=0.05)
    return None


def _publish_wake_until_active_context(
    node: Node,
    publisher,
    process: subprocess.Popen[str],
    received: list[TurnContext],
) -> TurnContext | None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        _assert_process_running(process)
        msg = WakeWordResult()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.detected = True
        msg.keyword = "fluent"
        msg.score = 1.0
        publisher.publish(msg)
        context = _wait_for_context(
            node,
            process,
            received,
            lambda turn_context: turn_context.active
            and turn_context.session_id != ""
            and turn_context.user_turn_id == 1,
            timeout_sec=0.1,
        )
        if context is not None:
            return context
    return None


def _asr_final_message(context: TurnContext) -> AsrResult:
    msg = AsrResult()
    msg.timestamp = context.timestamp
    msg.session_id = context.session_id
    msg.user_turn_id = context.user_turn_id
    msg.status = AsrResult.STATUS_FINAL
    msg.text = "hello"
    return msg


def _turn_end_message(context: TurnContext) -> TurnEnd:
    msg = TurnEnd()
    msg.timestamp = context.timestamp
    msg.session_id = context.session_id
    msg.user_turn_id = context.user_turn_id
    msg.probability = 1.0
    msg.is_end = True
    return msg


def _assert_active_then_inactive_turn_context(
    process: subprocess.Popen[str],
    unique: str,
    topics: DialogueSmokeTopics,
    terminal_publisher: str,
    terminal_message: TerminalMessageFactory,
) -> None:
    rclpy_initialized = False
    node: Node | None = None
    try:
        rclpy.init()
        rclpy_initialized = True
        node = rclpy.create_node(f"fa_dialogue_graph_test_{unique}")
        wake_pub = node.create_publisher(WakeWordResult, topics.wake_word, 10)
        asr_pub = node.create_publisher(AsrResult, topics.asr_result, 10)
        turn_end_pub = node.create_publisher(TurnEnd, topics.turn_end, 10)
        received: list[TurnContext] = []
        subscription = node.create_subscription(
            TurnContext,
            topics.turn_context,
            lambda msg: received.append(msg),
            10,
        )

        assert _wait_for_subscription(node, wake_pub, process, 8.0), _stop_process(
            process
        )
        if terminal_publisher == "asr":
            assert _wait_for_subscription(node, asr_pub, process, 8.0), _stop_process(
                process
            )
        else:
            assert _wait_for_subscription(
                node,
                turn_end_pub,
                process,
                8.0,
            ), _stop_process(process)
        assert _wait_for_publisher(
            node,
            topics.turn_context,
            process,
            8.0,
        ), _stop_process(process)

        active_context = _publish_wake_until_active_context(
            node,
            wake_pub,
            process,
            received,
        )
        assert active_context is not None, _stop_process(process)
        assert active_context.session_id != ""
        assert active_context.user_turn_id == 1
        assert active_context.active is True

        if terminal_publisher == "asr":
            asr_pub.publish(terminal_message(active_context))
        else:
            turn_end_pub.publish(terminal_message(active_context))

        inactive_context = _wait_for_context(
            node,
            process,
            received,
            lambda turn_context: turn_context.session_id == active_context.session_id
            and turn_context.user_turn_id == active_context.user_turn_id
            and not turn_context.active,
            timeout_sec=5.0,
        )
        assert inactive_context is not None, _stop_process(process)
        assert subscription.topic_name == topics.turn_context
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy_initialized:
            rclpy.shutdown()
        if process.poll() is None:
            _stop_process(process)


@pytest.mark.parametrize(
    ("terminal_publisher", "terminal_message"),
    (
        ("asr", _asr_final_message),
        ("turn_end", _turn_end_message),
    ),
)
def test_fa_dialogue_ros_graph_publishes_active_then_inactive_turn_context(
    tmp_path: Path,
    terminal_publisher: str,
    terminal_message: TerminalMessageFactory,
) -> None:
    unique = "s_" + uuid.uuid4().hex
    topics = _make_topics(unique)
    config_path = _write_dialogue_config(tmp_path, unique, topics)
    process = _start_run_node(config_path)

    _assert_active_then_inactive_turn_context(
        process,
        unique,
        topics,
        terminal_publisher,
        terminal_message,
    )


def test_fa_dialogue_launch_uses_config_file_and_publishes_turn_context(
    tmp_path: Path,
) -> None:
    unique = "s_" + uuid.uuid4().hex
    topics = _make_topics(unique)
    config_path = _write_dialogue_config(tmp_path, unique, topics)
    process = _start_launch_node(config_path)

    _assert_active_then_inactive_turn_context(
        process,
        unique,
        topics,
        "asr",
        _asr_final_message,
    )
