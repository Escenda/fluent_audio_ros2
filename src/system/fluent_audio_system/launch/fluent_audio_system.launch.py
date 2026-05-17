import os

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path):
    if not path:
        raise RuntimeError("config launch argument is required")
    if not os.path.isfile(path):
        raise RuntimeError(f"fluent_audio_system config not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise RuntimeError("fluent_audio_system config root must be a mapping")
    return data


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _require_sequence(value, label):
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a sequence")
    return value


def _required_text(mapping, key, label):
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label}.{key} is required")
    return value.strip()


def _optional_text(mapping, key, default_value=""):
    value = mapping.get(key, default_value)
    if value is None:
        return default_value
    if not isinstance(value, str):
        raise RuntimeError(f"{key} must be a string")
    return value


def _optional_bool(mapping, key, default_value=True):
    value = mapping.get(key, default_value)
    if not isinstance(value, bool):
        raise RuntimeError(f"{key} must be a bool")
    return value


def _optional_float(mapping, key, default_value):
    value = mapping.get(key, default_value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{key} must be a number")
    return float(value)


def _node_parameters(node_config):
    params_file = _optional_text(node_config, "params_file", "").strip()
    inline_params = node_config.get("parameters", {})
    if inline_params is None:
        inline_params = {}
    if not isinstance(inline_params, dict):
        raise RuntimeError(f"node {node_config.get('id', '?')} parameters must be a mapping")

    parameters = []
    if params_file:
        if not os.path.isfile(params_file):
            raise RuntimeError(f"params_file not found: {params_file}")
        parameters.append(params_file)
    if inline_params:
        parameters.append(inline_params)
    if not parameters:
        raise RuntimeError(f"node {node_config.get('id', '?')} requires params_file or parameters")
    return parameters


def _launch_setup(context):
    config_path = LaunchConfiguration("config").perform(context)
    config = _load_yaml(config_path)
    system = _require_mapping(config.get("system", {}), "system")
    groups = _require_sequence(config.get("groups", []), "groups")

    default_start_delay = _optional_float(system, "default_start_delay", 0.0)
    inter_group_delay = _optional_float(system, "inter_group_delay", 0.0)

    actions = [LogInfo(msg=f"[fluent_audio_system] config={config_path}")]
    delay = 0.0

    for group_index, raw_group in enumerate(groups):
        group = _require_mapping(raw_group, f"groups[{group_index}]")
        group_id = _required_text(group, "id", f"groups[{group_index}]")
        if not _optional_bool(group, "enable", True):
            actions.append(LogInfo(msg=f"[fluent_audio_system] skip group {group_id}"))
            continue

        nodes = _require_sequence(group.get("nodes", []), f"group {group_id}.nodes")
        for node_index, raw_node in enumerate(nodes):
            node_config = _require_mapping(raw_node, f"group {group_id}.nodes[{node_index}]")
            node_id = _required_text(node_config, "id", f"group {group_id}.nodes[{node_index}]")
            if not _optional_bool(node_config, "enable", True):
                actions.append(LogInfo(msg=f"[fluent_audio_system] skip node {node_id}"))
                continue

            package = _required_text(node_config, "package", f"node {node_id}")
            executable = _required_text(node_config, "exec", f"node {node_id}")
            node_name = _optional_text(node_config, "node_name", node_id)
            namespace = _optional_text(node_config, "namespace", "")
            if namespace == "/":
                namespace = ""

            node_action = Node(
                package=package,
                executable=executable,
                name=node_name,
                namespace=namespace,
                output=_optional_text(node_config, "output", "screen"),
                parameters=_node_parameters(node_config),
            )
            actions.append(
                LogInfo(
                    msg=(
                        "[fluent_audio_system] launch "
                        f"{package}:{executable} name={node_name} ns={namespace}"
                    )
                )
            )
            actions.append(TimerAction(period=delay, actions=[node_action]))
            delay += default_start_delay

        if group_index < len(groups) - 1:
            delay += inter_group_delay

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value="/config/fluent_audio_system.yaml",
                description="Absolute path to fluent_audio_system yaml.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
