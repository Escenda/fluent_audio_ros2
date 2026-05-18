#!/usr/bin/env python3
import json

import rclpy
from rclpy.exceptions import ParameterUninitializedException
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.publisher import Publisher
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from fa_interfaces.srv import Speak


class FaVoiceCommandRouterNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_voice_command_router")

        self._declare_required_parameters()

        command_topic = self._read_required_string("command_topic")
        state_topic = self._read_required_string("state_topic")
        self._active = self._read_required_bool("active")
        self._mode = self._normalize_parameter_mode(
            self._read_required_string("mode")
        )
        self._allowed_modes = self._load_allowed_modes()
        self._announce_tts = self._read_required_bool("announce_tts")
        self._tts_service = self._read_required_string("tts_service")
        self._tts_voice_id = self._read_required_string(
            "tts_voice_id", allow_empty=True
        )
        self._stop_output_on_stop = self._read_required_bool("stop_output_on_stop")

        if self._mode not in self._allowed_modes:
            raise RuntimeError(
                f"mode '{self._mode}' is not in allowed_modes={self._allowed_modes}"
            )

        self._state_pub = self.create_publisher(String, state_topic, 10)
        self._stop_pub: Publisher | None = None
        if self._stop_output_on_stop:
            output_stop_topic = self._read_required_string("output_stop_topic")
            self._stop_pub = self.create_publisher(
                Empty, output_stop_topic, 10
            )

        self._command_sub = self.create_subscription(
            String, command_topic, self._on_command, 10
        )

        self._start_srv = self.create_service(Trigger, "start", self._handle_start)
        self._stop_srv = self.create_service(Trigger, "stop", self._handle_stop)
        self._status_srv = self.create_service(Trigger, "status", self._handle_status)

        self._tts_client = self.create_client(Speak, self._tts_service)

        self.add_on_set_parameters_callback(self._on_set_parameters)

        self._publish_state()
        self.get_logger().info(
            "FA VoiceCommandRouter: "
            f"command_topic={command_topic} "
            f"state_topic={state_topic} "
            f"active={self._active} "
            f"mode={self._mode}"
        )

    def _declare_required_parameters(self) -> None:
        for parameter_name, parameter_type in (
            ("command_topic", Parameter.Type.STRING),
            ("state_topic", Parameter.Type.STRING),
            ("active", Parameter.Type.BOOL),
            ("mode", Parameter.Type.STRING),
            ("allowed_modes", Parameter.Type.STRING_ARRAY),
            ("announce_tts", Parameter.Type.BOOL),
            ("tts_service", Parameter.Type.STRING),
            ("tts_voice_id", Parameter.Type.STRING),
            ("stop_output_on_stop", Parameter.Type.BOOL),
            ("output_stop_topic", Parameter.Type.STRING),
        ):
            self.declare_parameter(parameter_name, parameter_type)

    def _get_required_parameter(self, name: str) -> Parameter:
        try:
            return self.get_parameter(name)
        except ParameterUninitializedException as exc:
            raise RuntimeError(f"{name} is required") from exc

    def _read_required_string(self, name: str, *, allow_empty: bool = False) -> str:
        parameter = self._get_required_parameter(name)
        if parameter.type_ != Parameter.Type.STRING:
            raise RuntimeError(f"{name} must be a string parameter")
        value = parameter.value
        if not isinstance(value, str):
            raise RuntimeError(f"{name} must be a string parameter")
        if not allow_empty and not value:
            raise RuntimeError(f"{name} is required")
        return value

    def _read_required_bool(self, name: str) -> bool:
        parameter = self._get_required_parameter(name)
        if parameter.type_ != Parameter.Type.BOOL:
            raise RuntimeError(f"{name} must be a bool parameter")
        value = parameter.value
        if not isinstance(value, bool):
            raise RuntimeError(f"{name} must be a bool parameter")
        return value

    def _read_required_string_array(self, name: str) -> list[str]:
        parameter = self._get_required_parameter(name)
        if parameter.type_ != Parameter.Type.STRING_ARRAY:
            raise RuntimeError(f"{name} must be a string array parameter")
        value = parameter.value
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            raise RuntimeError(f"{name} must be a string array parameter")
        return value

    def _load_allowed_modes(self) -> list[str]:
        return self._normalize_allowed_modes(
            self._read_required_string_array("allowed_modes")
        )

    def _normalize_mode(self, mode: str) -> str:
        return mode.strip().lower().replace("-", "_")

    def _normalize_parameter_mode(self, mode: str) -> str:
        normalized = self._normalize_mode(mode)
        if not normalized:
            raise RuntimeError("mode must not be empty")
        return normalized

    def _normalize_allowed_modes(self, modes: list[str]) -> list[str]:
        normalized_modes = [self._normalize_mode(item) for item in modes]
        normalized_modes = [item for item in normalized_modes if item]
        if not normalized_modes:
            raise RuntimeError("allowed_modes must not be empty")
        if len(set(normalized_modes)) != len(normalized_modes):
            raise RuntimeError("allowed_modes must not contain duplicate modes")
        return normalized_modes

    def _parameter_failure(self, reason: str) -> SetParametersResult:
        return SetParametersResult(successful=False, reason=reason)

    def _parameter_failure_from_runtime_error(
        self, error: RuntimeError
    ) -> SetParametersResult:
        if error.args and isinstance(error.args[0], str):
            return self._parameter_failure(error.args[0])
        return self._parameter_failure("invalid parameter")

    def _read_parameter_update_bool(
        self, parameter: Parameter
    ) -> bool | SetParametersResult:
        if parameter.type_ != Parameter.Type.BOOL:
            return self._parameter_failure(f"{parameter.name} must be a bool parameter")
        value = parameter.value
        if not isinstance(value, bool):
            return self._parameter_failure(f"{parameter.name} must be a bool parameter")
        return value

    def _read_parameter_update_string(
        self, parameter: Parameter
    ) -> str | SetParametersResult:
        if parameter.type_ != Parameter.Type.STRING:
            return self._parameter_failure(
                f"{parameter.name} must be a string parameter"
            )
        value = parameter.value
        if not isinstance(value, str):
            return self._parameter_failure(
                f"{parameter.name} must be a string parameter"
            )
        return value

    def _read_parameter_update_string_array(
        self, parameter: Parameter
    ) -> list[str] | SetParametersResult:
        if parameter.type_ != Parameter.Type.STRING_ARRAY:
            return self._parameter_failure(
                f"{parameter.name} must be a string array parameter"
            )
        value = parameter.value
        if not isinstance(value, list) or not all(
            isinstance(item, str) for item in value
        ):
            return self._parameter_failure(
                f"{parameter.name} must be a string array parameter"
            )
        return value

    def _is_parameter_failure(
        self, result: bool | str | list[str] | SetParametersResult
    ) -> bool:
        return isinstance(result, SetParametersResult)

    def _publish_state(self) -> None:
        msg = String()
        msg.data = json.dumps(
            {"active": self._active, "mode": self._mode}, ensure_ascii=False
        )
        self._state_pub.publish(msg)

    def _maybe_announce(self, text: str) -> None:
        if not self._announce_tts:
            return

        if not self._tts_client.service_is_ready():
            self.get_logger().error(
                "announce_tts is enabled but TTS service is not ready: "
                f"{self._tts_service}"
            )
            return

        request = Speak.Request()
        request.text = text
        request.voice_id = self._tts_voice_id
        request.play = False
        request.volume_db = 0.0
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
                f"Rejected mode '{normalized}' (allowed={self._allowed_modes})"
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

        if cmd == "start":
            self._set_active(True)
            return

        if cmd == "stop":
            self._set_active(False)
            return

        if cmd == "mode" and len(args) == 1:
            self._set_mode(args[0])
            return

        if cmd == "status":
            self._publish_state()
            return

        # Future: route other commands only when active.
        if not self._active:
            self.get_logger().debug(f"Ignored command while inactive: {raw}")
            return

        self.get_logger().info(f"Unhandled command (pass-through): {raw}")

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

    def _on_set_parameters(self, params: list[Parameter]) -> SetParametersResult:
        next_active = self._active
        next_mode = self._mode
        next_allowed_modes = list(self._allowed_modes)

        for param in params:
            if param.name == "active":
                active = self._read_parameter_update_bool(param)
                if self._is_parameter_failure(active):
                    return active
                next_active = active
                continue

            if param.name == "mode":
                mode = self._read_parameter_update_string(param)
                if self._is_parameter_failure(mode):
                    return mode
                try:
                    next_mode = self._normalize_parameter_mode(mode)
                except RuntimeError as exc:
                    return self._parameter_failure_from_runtime_error(exc)
                continue

            if param.name == "allowed_modes":
                allowed_modes = self._read_parameter_update_string_array(param)
                if self._is_parameter_failure(allowed_modes):
                    return allowed_modes
                try:
                    next_allowed_modes = self._normalize_allowed_modes(allowed_modes)
                except RuntimeError as exc:
                    return self._parameter_failure_from_runtime_error(exc)
                continue

            return self._parameter_failure(f"{param.name} is a startup-only parameter")

        if next_mode not in next_allowed_modes:
            return self._parameter_failure(
                f"mode '{next_mode}' is not in allowed_modes={next_allowed_modes}"
            )

        self._allowed_modes = next_allowed_modes
        if not self._set_mode(next_mode):
            return self._parameter_failure(
                f"mode '{next_mode}' is not in allowed_modes={next_allowed_modes}"
            )
        self._set_active(next_active)
        return SetParametersResult(successful=True)


def main(argv: list[str] | None = None) -> None:
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
