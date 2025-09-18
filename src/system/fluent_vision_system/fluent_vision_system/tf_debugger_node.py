"""TF debug bridge node for Foxglove inspections.

This node publishes static transforms that stitch together otherwise disconnected
frames so that TF trees look consistent during desktop debugging. The node is
fully optional and can be disabled from configuration.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from geometry_msgs.msg import TransformStamped
import yaml
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from tf2_ros.transform_broadcaster import TransformBroadcaster


@dataclass
class TransformConfig:
    parent_frame: str
    child_frame: str
    translation: List[float]
    rotation_quat: List[float]
    is_static: bool = True

    @classmethod
    def from_mapping(cls, data: dict, logger: Node) -> Optional["TransformConfig"]:
        parent = str(data.get("parent") or data.get("parent_frame") or data.get("from") or "").strip()
        child = str(data.get("child") or data.get("child_frame") or data.get("to") or "").strip()
        if not parent or not child:
            logger.get_logger().warning(
                "Invalid TF definition (missing parent/child): parent='%s' child='%s'", parent, child
            )
            return None

        translation = _parse_vector(
            data.get("translation") or data.get("xyz") or [0.0, 0.0, 0.0], 3, logger,
            label=f"translation for {parent}->{child}"
        )
        rotation_quat: Optional[List[float]]

        if "rotation" in data:
            rotation_quat = _parse_vector(data["rotation"], 4, logger, label=f"quaternion for {parent}->{child}")
        elif "quaternion" in data:
            rotation_quat = _parse_vector(data["quaternion"], 4, logger, label=f"quaternion for {parent}->{child}")
        elif "rotation_rpy" in data or "rpy" in data:
            rpy = _parse_vector(
                data.get("rotation_rpy") or data.get("rpy"), 3, logger, label=f"RPY for {parent}->{child}"
            )
            rotation_quat = list(_quaternion_from_rpy(*rpy))
        elif {"roll", "pitch", "yaw"}.issubset(data.keys()):
            rpy = [float(data["roll"]), float(data["pitch"]), float(data["yaw"])]
            rotation_quat = list(_quaternion_from_rpy(*rpy))
        else:
            rotation_quat = [0.0, 0.0, 0.0, 1.0]

        is_static = bool(data.get("static", True))
        return cls(parent, child, translation, rotation_quat, is_static)

    @classmethod
    def from_inline(cls, text: str, logger: Node) -> Optional["TransformConfig"]:
        parts = [p for p in text.split() if p]
        if len(parts) not in (8, 9):
            logger.get_logger().warning(
                "transforms_inline entries must follow 'parent child x y z roll pitch yaw [static]' format: %s",
                text,
            )
            return None

        parent, child = parts[0], parts[1]
        try:
            numbers = [float(v) for v in parts[2:8]]
        except ValueError:
            logger.get_logger().warning("Failed to parse numeric values: %s", text)
            return None

        is_static = True
        if len(parts) == 9:
            flag = parts[8].lower()
            is_static = flag in ("1", "true", "yes", "on", "static")

        translation = numbers[0:3]
        rotation_quat = list(_quaternion_from_rpy(*numbers[3:6]))
        return cls(parent, child, translation, rotation_quat, is_static)


def _parse_vector(raw, expected_size: int, node: Node, label: str) -> List[float]:
    if isinstance(raw, dict):
        if expected_size == 3:
            values = [raw.get(k, 0.0) for k in ("x", "y", "z")]
        else:
            values = [raw.get(k, 0.0) for k in ("x", "y", "z", "w")]
    elif isinstance(raw, (list, tuple)):
        values = list(raw)[:expected_size]
    else:
        node.get_logger().warning("%s could not be parsed; defaulting to zeros. raw=%s", label, raw)
        values = [0.0] * expected_size

    if len(values) < expected_size:
        values.extend([0.0] * (expected_size - len(values)))

    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        node.get_logger().warning("%s conversion failed; defaulting to zeros. raw=%s", label, values)
        return [0.0] * expected_size


def _quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> Iterable[float]:
    half_roll = roll * 0.5
    half_pitch = pitch * 0.5
    half_yaw = yaw * 0.5

    cr = math.cos(half_roll)
    sr = math.sin(half_roll)
    cp = math.cos(half_pitch)
    sp = math.sin(half_pitch)
    cy = math.cos(half_yaw)
    sy = math.sin(half_yaw)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class TfDebugBridgeNode(Node):
    """Broadcasts static or slow-moving transforms for debugging purposes."""

    def __init__(self) -> None:
        super().__init__("fv_tf_debugger")
        self.declare_parameter("enabled", True)
        self.declare_parameter("transform_file", "")
        self.declare_parameter("transforms_inline", [])
        self.declare_parameter("dynamic_publish_rate", 0.0)

        self._enabled = self.get_parameter("enabled").get_parameter_value().bool_value
        if not self._enabled:
            self.get_logger().info("TF debug bridge is disabled via parameter (enabled=false).")
            return

        transform_file = self.get_parameter("transform_file").get_parameter_value().string_value.strip()
        inline_entries = self._get_string_array_parameter("transforms_inline")
        publish_rate = self.get_parameter("dynamic_publish_rate").get_parameter_value().double_value

        configs: List[TransformConfig] = []
        if transform_file:
            configs.extend(self._load_from_file(transform_file))
        if inline_entries:
            configs.extend(self._load_inline(inline_entries))

        if not configs:
            self.get_logger().warn(
                "No TF definitions provided. Configure transform_file or transforms_inline."
            )
            self._enabled = False
            return

        static_cfgs = [c for c in configs if c.is_static]
        dynamic_cfgs = [c for c in configs if not c.is_static]

        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._dynamic_broadcaster: Optional[TransformBroadcaster] = None
        self._dynamic_cfgs = dynamic_cfgs

        if static_cfgs:
            transforms = [self._to_transform_stamped(cfg) for cfg in static_cfgs]
            self._static_broadcaster.sendTransform(transforms)
            self.get_logger().info(f'Published {len(static_cfgs)} static TF transforms.')

        self._timer = None
        if dynamic_cfgs:
            if publish_rate <= 0.0:
                publish_rate = 5.0
            self._dynamic_broadcaster = TransformBroadcaster(self)
            period = 1.0 / publish_rate
            self._timer = self.create_timer(period, self._publish_dynamic)
            self.get_logger().info(
                "Publishing %d dynamic TF transforms at %.2f Hz.", len(dynamic_cfgs), publish_rate
            )

        if not static_cfgs and dynamic_cfgs and publish_rate > 0.0:
            self.get_logger().info('Dynamic-only TF publication is active.')

    def _get_string_array_parameter(self, name: str) -> List[str]:
        param = self.get_parameter(name)
        if param.type_ == Parameter.Type.NOT_SET:
            return []
        if param.type_ in (Parameter.Type.STRING_ARRAY, Parameter.Type.BYTE_ARRAY):
            return list(param.get_parameter_value().string_array_value)
        if param.type_ == Parameter.Type.STRING:
            value = param.get_parameter_value().string_value
            return [v.strip() for v in value.split(";") if v.strip()]
        return []

    def _load_from_file(self, path: str) -> List[TransformConfig]:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if not os.path.isabs(expanded):
            expanded = os.path.join("/config", expanded)

        if not os.path.exists(expanded):
            self.get_logger().warning("transform_file '%s' not found.", expanded)
            return []

        try:
            with open(expanded, "r", encoding="utf-8") as f_handle:
                data = yaml.safe_load(f_handle) or {}
        except Exception as exc:  # pylint: disable=broad-except
            self.get_logger().error("Failed to read transform_file: %s", exc)
            return []

        records = data.get("transforms") if isinstance(data, dict) else None
        if not isinstance(records, list):
            self.get_logger().warning("transform_file does not contain a 'transforms' list.")
            return []

        configs: List[TransformConfig] = []
        for item in records:
            if not isinstance(item, dict):
                self.get_logger().warning("Skipping non-dict transform entry: %s", item)
                continue
            cfg = TransformConfig.from_mapping(item, self)
            if cfg:
                configs.append(cfg)
        return configs

    def _load_inline(self, entries: Iterable[str]) -> List[TransformConfig]:
        configs: List[TransformConfig] = []
        for entry in entries:
            cfg = TransformConfig.from_inline(entry, self)
            if cfg:
                configs.append(cfg)
        return configs

    def _to_transform_stamped(self, cfg: TransformConfig) -> TransformStamped:
        msg = TransformStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = cfg.parent_frame
        msg.child_frame_id = cfg.child_frame
        msg.transform.translation.x = cfg.translation[0]
        msg.transform.translation.y = cfg.translation[1]
        msg.transform.translation.z = cfg.translation[2]
        msg.transform.rotation.x = cfg.rotation_quat[0]
        msg.transform.rotation.y = cfg.rotation_quat[1]
        msg.transform.rotation.z = cfg.rotation_quat[2]
        msg.transform.rotation.w = cfg.rotation_quat[3]
        return msg

    def _publish_dynamic(self) -> None:
        if not self._dynamic_broadcaster:
            return
        transforms = [self._to_transform_stamped(cfg) for cfg in self._dynamic_cfgs]
        self._dynamic_broadcaster.sendTransform(transforms)


def main(args: Optional[List[str]] = None) -> None:
    rclpy.init(args=args)
    node = TfDebugBridgeNode()
    try:
        if node._enabled:  # pylint: disable=protected-access
            rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:  # defensive cleanup
            pass
        # rclpy.shutdown() が既に呼ばれている場合にRCLErrorを避ける
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            # 既にshutdown済みなどのケースは黙認
            pass


if __name__ == "__main__":
    main()
