#!/usr/bin/env python3
import json
from typing import List, Optional

import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from fa_interfaces.srv import Speak


class FaVoiceCommandRouterNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_voice_command_router")

        self.declare_parameter("command_topic", "voice/command")
        self.declare_parameter("state_topic", "voice/router/state")
        self.declare_parameter("active", False)
        self.declare_parameter("mode", "standby")
        self.declare_parameter("allowed_modes", ["standby", "command", "dictation", "mute"])

        self.declare_parameter("announce_tts", False)
        self.declare_parameter("tts_service", "speak")
        self.declare_parameter("tts_voice_id", "")
        self.declare_parameter("tts_play", True)
        self.declare_parameter("tts_volume_db", 0.0)

        self.declare_parameter("stop_output_on_stop", True)
        self.declare_parameter("output_stop_topic", "audio/output/stop")

        self._active = bool(self.get_parameter("active").value)
        self._mode = str(self.get_parameter("mode").value)
        self._allowed_modes = self._load_allowed_modes()

        command_topic = str(self.get_parameter("command_topic").value)
        state_topic = str(self.get_parameter("state_topic").value)

        self._state_pub = self.create_publisher(String, state_topic, 10)
        self._stop_pub: Optional[rclpy.publisher.Publisher] = None
        if bool(self.get_parameter("stop_output_on_stop").value):
            self._stop_pub = self.create_publisher(
                Empty, str(self.get_parameter("output_stop_topic").value), 10
            )

        self._command_sub = self.create_subscription(
            String, command_topic, self._on_command, 10
        )

        self._start_srv = self.create_service(Trigger, "start", self._handle_start)
        self._stop_srv = self.create_service(Trigger, "stop", self._handle_stop)
        self._status_srv = self.create_service(Trigger, "status", self._handle_status)

        self._tts_client = self.create_client(
            Speak, str(self.get_parameter("tts_service").value)
        )

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._publish_state()
        self.get_logger().info(
            "FA VoiceCommandRouter: command_topic=%s state_topic=%s active=%s mode=%s",
            command_topic,
            state_topic,
            self._active,
            self._mode,
        )

    def _load_allowed_modes(self) -> List[str]:
        value = self.get_parameter("allowed_modes").value
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return ["standby", "command", "dictation", "mute"]

    def _normalize_mode(self, mode: str) -> str:
        return mode.strip().lower().replace("-", "_")

    def _publish_state(self) -> None:
        msg = String()
        msg.data = json.dumps(
            {"active": self._active, "mode": self._mode}, ensure_ascii=False
        )
        self._state_pub.publish(msg)

    def _maybe_announce(self, text: str) -> None:
        if not bool(self.get_parameter("announce_tts").value):
            return

        if not self._tts_client.service_is_ready():
            return

        request = Speak.Request()
        request.text = text
        request.voice_id = str(self.get_parameter("tts_voice_id").value)
        request.play = bool(self.get_parameter("tts_play").value)
        request.volume_db = float(self.get_parameter("tts_volume_db").value)
        request.cache_key = ""
        self._tts_client.call_async(request)

    def _set_active(self, active: bool) -> None:
        if self._active == active:
            return
        self._active = active
        self._publish_state()
        self._maybe_announce("起動しました" if self._active else "停止しました")
        if not self._active and self._stop_pub is not None:
            self._stop_pub.publish(Empty())

    def _set_mode(self, mode: str) -> bool:
        normalized = self._normalize_mode(mode)
        if normalized not in self._allowed_modes:
            self.get_logger().warning(
                "Rejected mode '%s' (allowed=%s)", normalized, self._allowed_modes
            )
            return False
        if self._mode == normalized:
            return True
        self._mode = normalized
        self._publish_state()
        self._maybe_announce(f"モードを {self._mode} に変更しました")
        return True

    def _on_command(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return

        parts = raw.split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd in ("start", "on", "enable"):
            self._set_active(True)
            return

        if cmd in ("stop", "off", "disable"):
            self._set_active(False)
            return

        if cmd in ("mode", "set_mode") and args:
            self._set_mode(args[0])
            return

        if cmd.startswith("mode:") and len(cmd) > 5:
            self._set_mode(cmd.split(":", 1)[1])
            return

        if cmd in ("status", "state"):
            self._publish_state()
            return

        # Future: route other commands only when active.
        if not self._active:
            self.get_logger().debug("Ignored command while inactive: %s", raw)
            return

        self.get_logger().info("Unhandled command (pass-through): %s", raw)

    def _handle_start(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self._set_active(True)
        response.success = True
        response.message = "ok"
        return response

    def _handle_stop(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        self._set_active(False)
        response.success = True
        response.message = "ok"
        return response

    def _handle_status(
        self, _request: Trigger.Request, response: Trigger.Response
    ) -> Trigger.Response:
        response.success = True
        response.message = json.dumps(
            {"active": self._active, "mode": self._mode}, ensure_ascii=False
        )
        return response

    def _on_set_parameters(self, params: List[Parameter]) -> SetParametersResult:
        for param in params:
            if param.name == "active" and param.type_ == Parameter.Type.BOOL:
                self._set_active(bool(param.value))
            if param.name == "mode" and param.type_ == Parameter.Type.STRING:
                if not self._set_mode(str(param.value)):
                    return SetParametersResult(successful=False)
            if (
                param.name == "allowed_modes"
                and param.type_ == Parameter.Type.STRING_ARRAY
            ):
                if not isinstance(param.value, list) or not all(
                    isinstance(item, str) for item in param.value
                ):
                    return SetParametersResult(successful=False)
                self._allowed_modes = list(param.value)
        return SetParametersResult(successful=True)


def main(argv=None) -> None:
    rclpy.init(args=argv)
    node = FaVoiceCommandRouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

