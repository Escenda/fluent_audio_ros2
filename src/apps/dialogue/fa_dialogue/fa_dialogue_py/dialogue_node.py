#!/usr/bin/env python3
from __future__ import annotations

from builtin_interfaces.msg import Time
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AsrResult, TurnContext, TurnEnd, WakeWordResult
from fa_dialogue_py.session_state import (
    CompletionEvent,
    DialogueSessionState,
    MessageStamp,
    SessionStateConfig,
    TurnContextSnapshot,
    WakeEvent,
)


class FaDialogueNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_dialogue")
        self._declare_required_parameters()

        self.wake_word_topic = self._string_parameter("wake_word_topic")
        self.asr_result_topic = self._string_parameter("asr_result_topic")
        self.turn_end_topic = self._string_parameter("turn_end_topic")
        self.turn_context_topic = self._string_parameter("turn_context_topic")
        session_prefix = self._string_parameter("session_prefix")
        wake_max_age_ms = self._positive_integer_parameter("wake.max_age_ms")
        wake_allow_zero_stamp = self._bool_parameter("wake.allow_zero_stamp")

        self._state = DialogueSessionState(
            SessionStateConfig(
                session_prefix=session_prefix,
                wake_max_age_ms=wake_max_age_ms,
                wake_allow_zero_stamp=wake_allow_zero_stamp,
            )
        )

        wake_qos = self._qos_profile("wake.qos.depth", "wake.qos.reliable")
        asr_qos = self._qos_profile("asr.qos.depth", "asr.qos.reliable")
        turn_end_qos = self._qos_profile(
            "turn_end.qos.depth",
            "turn_end.qos.reliable",
        )
        context_qos = self._qos_profile(
            "turn_context.qos.depth",
            "turn_context.qos.reliable",
        )

        self.turn_context_pub = self.create_publisher(
            TurnContext,
            self.turn_context_topic,
            context_qos,
        )
        self.wake_word_sub = self.create_subscription(
            WakeWordResult,
            self.wake_word_topic,
            self.on_wake_word,
            wake_qos,
        )
        self.asr_result_sub = self.create_subscription(
            AsrResult,
            self.asr_result_topic,
            self.on_asr_result,
            asr_qos,
        )
        self.turn_end_sub = self.create_subscription(
            TurnEnd,
            self.turn_end_topic,
            self.on_turn_end,
            turn_end_qos,
        )

        self.get_logger().info(
            "fa_dialogue started: wake=%s asr_result=%s turn_end=%s turn_context=%s",
            self.wake_word_topic,
            self.asr_result_topic,
            self.turn_end_topic,
            self.turn_context_topic,
        )

    def on_wake_word(self, msg: WakeWordResult) -> None:
        decision = self._state.handle_wake(
            WakeEvent(
                detected=msg.detected,
                keyword=msg.keyword,
                stamp=MessageStamp(
                    sec=msg.header.stamp.sec,
                    nanosec=msg.header.stamp.nanosec,
                ),
            ),
            now_ms=self._now_ms(),
        )
        if decision.context is None:
            self.get_logger().debug(
                "wake ignored: kind=%s reason=%s",
                decision.kind,
                decision.reason,
            )
            return
        self._publish_context(decision.context)

    def on_asr_result(self, msg: AsrResult) -> None:
        decision = self._state.handle_asr_result(
            CompletionEvent(
                session_id=msg.session_id,
                user_turn_id=msg.user_turn_id,
                terminal=True,
            ),
            status=msg.status,
        )
        if decision.context is None:
            self.get_logger().debug("asr result ignored: reason=%s", decision.reason)
            return
        self._publish_context(decision.context)

    def on_turn_end(self, msg: TurnEnd) -> None:
        decision = self._state.handle_turn_end(
            CompletionEvent(
                session_id=msg.session_id,
                user_turn_id=msg.user_turn_id,
                terminal=msg.is_end,
            )
        )
        if decision.context is None:
            self.get_logger().debug("turn end ignored: reason=%s", decision.reason)
            return
        self._publish_context(decision.context)

    def _publish_context(self, context: TurnContextSnapshot) -> None:
        msg = TurnContext()
        msg.timestamp = self._now_time_msg()
        msg.session_id = context.session_id
        msg.user_turn_id = context.user_turn_id
        msg.active = context.active
        self.turn_context_pub.publish(msg)

    def _declare_required_parameters(self) -> None:
        self.declare_parameter("wake_word_topic", Parameter.Type.STRING)
        self.declare_parameter("asr_result_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_end_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_context_topic", Parameter.Type.STRING)
        self.declare_parameter("session_prefix", Parameter.Type.STRING)
        self.declare_parameter("wake.max_age_ms", Parameter.Type.INTEGER)
        self.declare_parameter("wake.allow_zero_stamp", Parameter.Type.BOOL)
        self.declare_parameter("wake.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("wake.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("asr.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("asr.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_end.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_end.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_context.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_context.qos.reliable", Parameter.Type.BOOL)

    def _string_parameter(self, name: str) -> str:
        value = self.get_parameter(name).get_parameter_value().string_value.strip()
        if not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _bool_parameter(self, name: str) -> bool:
        return self.get_parameter(name).get_parameter_value().bool_value

    def _positive_integer_parameter(self, name: str) -> int:
        value = self.get_parameter(name).get_parameter_value().integer_value
        if value <= 0:
            raise RuntimeError(f"{name} must be positive")
        return value

    def _qos_profile(self, depth_parameter: str, reliable_parameter: str) -> QoSProfile:
        depth = self._positive_integer_parameter(depth_parameter)
        reliable = self._bool_parameter(reliable_parameter)
        profile = QoSProfile(depth=depth)
        profile.history = HistoryPolicy.KEEP_LAST
        if reliable:
            profile.reliability = ReliabilityPolicy.RELIABLE
        else:
            profile.reliability = ReliabilityPolicy.BEST_EFFORT
        return profile

    def _now_time_msg(self) -> Time:
        return self.get_clock().now().to_msg()

    def _now_ms(self) -> int:
        now = self._now_time_msg()
        return (now.sec * 1000) + (now.nanosec // 1_000_000)


def main() -> None:
    rclpy.init()
    node: FaDialogueNode | None = None
    executor: SingleThreadedExecutor | None = None
    try:
        node = FaDialogueNode()
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
