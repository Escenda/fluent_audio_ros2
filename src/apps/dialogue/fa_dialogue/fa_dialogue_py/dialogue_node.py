#!/usr/bin/env python3
from __future__ import annotations

from builtin_interfaces.msg import Time
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy

from fa_interfaces.msg import AsrControl, TurnContext, TurnEnd, VoiceActivity, WakeWordResult
from fa_dialogue_py.session_state import (
    AsrControlCommand,
    DialogueDecision,
    DialogueTurnConfig,
    DialogueTurnController,
    MessageStamp,
    TurnContextSnapshot,
    TurnEndCandidate,
    VoiceActivityEvent,
    WakeEvent,
)


class FaDialogueNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_dialogue")
        self._declare_required_parameters()

        self.wake_word_topic = self._string_parameter("wake_word_topic")
        self.voice_activity_topic = self._string_parameter("voice_activity_topic")
        self.turn_end_topic = self._string_parameter("turn_end_topic")
        self.turn_context_topic = self._string_parameter("turn_context_topic")
        self.asr_control_topic = self._string_parameter("asr_control_topic")
        self.expected_source_id = self._string_parameter("expected_source_id")
        self.expected_stream_id = self._string_parameter("expected_stream_id")
        self._state = DialogueTurnController(
            DialogueTurnConfig(
                session_prefix=self._string_parameter("session_prefix"),
                wake_max_age_ms=self._positive_integer_parameter("wake.max_age_ms"),
                wake_allow_zero_stamp=self._bool_parameter("wake.allow_zero_stamp"),
                min_turn_ms=self._non_negative_integer_parameter("turn.min_duration_ms"),
                min_listen_ms=self._non_negative_integer_parameter("turn.min_listen_ms"),
                no_speech_timeout_ms=self._positive_integer_parameter(
                    "turn.no_speech_timeout_ms"
                ),
                td_min_silence_ms=self._non_negative_integer_parameter(
                    "turn.td_min_silence_ms"
                ),
                vad_fallback_silence_ms=self._positive_integer_parameter(
                    "turn.vad_fallback_silence_ms"
                ),
            )
        )

        wake_qos = self._qos_profile("wake.qos.depth", "wake.qos.reliable")
        vad_qos = self._qos_profile("voice_activity.qos.depth", "voice_activity.qos.reliable")
        turn_end_qos = self._qos_profile("turn_end.qos.depth", "turn_end.qos.reliable")
        context_qos = self._qos_profile("turn_context.qos.depth", "turn_context.qos.reliable")
        asr_control_qos = self._qos_profile("asr_control.qos.depth", "asr_control.qos.reliable")

        self.turn_context_pub = self.create_publisher(
            TurnContext,
            self.turn_context_topic,
            context_qos,
        )
        self.asr_control_pub = self.create_publisher(
            AsrControl,
            self.asr_control_topic,
            asr_control_qos,
        )
        self.wake_word_sub = self.create_subscription(
            WakeWordResult,
            self.wake_word_topic,
            self.on_wake_word,
            wake_qos,
        )
        self.voice_activity_sub = self.create_subscription(
            VoiceActivity,
            self.voice_activity_topic,
            self.on_voice_activity,
            vad_qos,
        )
        self.turn_end_sub = self.create_subscription(
            TurnEnd,
            self.turn_end_topic,
            self.on_turn_end,
            turn_end_qos,
        )
        self.tick_timer = self.create_timer(
            self._positive_integer_parameter("turn.tick_period_ms") / 1000.0,
            self.on_tick,
        )

        self.get_logger().info(
            "fa_dialogue started: "
            f"wake={self.wake_word_topic} "
            f"vad={self.voice_activity_topic} "
            f"turn_end={self.turn_end_topic} "
            f"turn_context={self.turn_context_topic} "
            f"asr_control={self.asr_control_topic}"
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
        self._publish_decision(decision)

    def on_voice_activity(self, msg: VoiceActivity) -> None:
        if msg.source_id != self.expected_source_id or msg.stream_id != self.expected_stream_id:
            self.get_logger().warning("Dropping VoiceActivity with unexpected source/stream")
            return
        decision = self._state.handle_voice_activity(
            VoiceActivityEvent(
                is_speech=bool(msg.is_speech),
                speech_started=bool(msg.speech_started),
                speech_ended=bool(msg.speech_ended),
            ),
            now_ms=self._now_ms(),
        )
        self._publish_decision(decision)

    def on_turn_end(self, msg: TurnEnd) -> None:
        decision = self._state.handle_turn_end_candidate(
            TurnEndCandidate(
                session_id=msg.session_id,
                user_turn_id=int(msg.user_turn_id),
                terminal=bool(msg.is_end),
                probability=float(msg.probability),
            ),
            now_ms=self._now_ms(),
        )
        self._publish_decision(decision)

    def on_tick(self) -> None:
        decision = self._state.tick(now_ms=self._now_ms())
        self._publish_decision(decision)

    def _publish_decision(self, decision: DialogueDecision) -> None:
        if not decision.contexts and not decision.asr_controls:
            if decision.kind in ("rejected",):
                self.get_logger().debug(
                    f"dialogue decision ignored: kind={decision.kind} reason={decision.reason}"
                )
            return
        for control in decision.asr_controls:
            if control.action in ("stop", "cancel"):
                self._publish_asr_control(control)
        for context in decision.contexts:
            self._publish_context(context)
        for control in decision.asr_controls:
            if control.action == "start":
                self._publish_asr_control(control)
        self.get_logger().info(
            f"dialogue decision: kind={decision.kind} reason={decision.reason}"
        )

    def _publish_context(self, context: TurnContextSnapshot) -> None:
        msg = TurnContext()
        msg.timestamp = self._now_time_msg()
        msg.session_id = context.session_id
        msg.user_turn_id = context.user_turn_id
        msg.active = context.active
        self.turn_context_pub.publish(msg)

    def _publish_asr_control(self, command: AsrControlCommand) -> None:
        msg = AsrControl()
        msg.timestamp = self._now_time_msg()
        msg.action = self._asr_action_value(command.action)
        msg.session_id = command.session_id
        msg.user_turn_id = int(command.user_turn_id)
        msg.reason = command.reason
        self.asr_control_pub.publish(msg)

    @staticmethod
    def _asr_action_value(action: str) -> int:
        if action == "start":
            return AsrControl.ACTION_START
        if action == "stop":
            return AsrControl.ACTION_STOP
        if action == "cancel":
            return AsrControl.ACTION_CANCEL
        raise ValueError(f"unsupported ASR action: {action}")

    def _declare_required_parameters(self) -> None:
        self.declare_parameter("wake_word_topic", Parameter.Type.STRING)
        self.declare_parameter("voice_activity_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_end_topic", Parameter.Type.STRING)
        self.declare_parameter("turn_context_topic", Parameter.Type.STRING)
        self.declare_parameter("asr_control_topic", Parameter.Type.STRING)
        self.declare_parameter("expected_source_id", Parameter.Type.STRING)
        self.declare_parameter("expected_stream_id", Parameter.Type.STRING)
        self.declare_parameter("session_prefix", Parameter.Type.STRING)
        self.declare_parameter("wake.max_age_ms", Parameter.Type.INTEGER)
        self.declare_parameter("wake.allow_zero_stamp", Parameter.Type.BOOL)
        self.declare_parameter("turn.min_duration_ms", Parameter.Type.INTEGER)
        self.declare_parameter("turn.min_listen_ms", Parameter.Type.INTEGER)
        self.declare_parameter("turn.no_speech_timeout_ms", Parameter.Type.INTEGER)
        self.declare_parameter("turn.td_min_silence_ms", Parameter.Type.INTEGER)
        self.declare_parameter("turn.vad_fallback_silence_ms", Parameter.Type.INTEGER)
        self.declare_parameter("turn.tick_period_ms", Parameter.Type.INTEGER)
        self.declare_parameter("wake.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("wake.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("voice_activity.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("voice_activity.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_end.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_end.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("turn_context.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("turn_context.qos.reliable", Parameter.Type.BOOL)
        self.declare_parameter("asr_control.qos.depth", Parameter.Type.INTEGER)
        self.declare_parameter("asr_control.qos.reliable", Parameter.Type.BOOL)

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

    def _non_negative_integer_parameter(self, name: str) -> int:
        value = self.get_parameter(name).get_parameter_value().integer_value
        if value < 0:
            raise RuntimeError(f"{name} must be >= 0")
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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if executor is not None:
            executor.shutdown()
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
