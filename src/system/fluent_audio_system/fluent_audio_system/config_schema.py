from __future__ import annotations

import os
from dataclasses import dataclass

import yaml


ScalarParam = str | int | float | bool
ParamValue = ScalarParam | list[str] | list[int] | list[float] | list[bool]


@dataclass(frozen=True)
class AudioNodeSpec:
    id: str
    package: str
    executable: str
    node_name: str
    namespace: str
    output: str
    params_file: str
    parameters: dict[str, ParamValue]

    def launch_parameters(self):
        sources = []
        if self.params_file:
            sources.append(self.params_file)
        if self.parameters:
            sources.append(self.parameters)
        return sources


@dataclass(frozen=True)
class AudioGroupSpec:
    id: str
    nodes: list[AudioNodeSpec]


@dataclass(frozen=True)
class AudioSystemSpec:
    default_start_delay: float
    inter_group_delay: float
    groups: list[AudioGroupSpec]


def load_system_config(path: str) -> AudioSystemSpec:
    if not path:
        raise RuntimeError("config launch argument is required")
    if not os.path.isfile(path):
        raise RuntimeError(f"fluent_audio_system config not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return parse_system_config(raw)


def parse_system_config(raw) -> AudioSystemSpec:
    root = _require_mapping(raw, "fluent_audio_system config root")
    system = _require_mapping(root.get("system", {}), "system")
    groups_raw = _require_sequence(root.get("groups", []), "groups")
    default_start_delay = _optional_float(system, "default_start_delay", 0.0)
    inter_group_delay = _optional_float(system, "inter_group_delay", 0.0)

    groups: list[AudioGroupSpec] = []
    for group_index, raw_group in enumerate(groups_raw):
        group = _require_mapping(raw_group, f"groups[{group_index}]")
        group_id = _required_text(group, "id", f"groups[{group_index}]")
        if not _optional_bool(group, "enable", True):
            continue
        nodes_raw = _require_sequence(group.get("nodes", []), f"group {group_id}.nodes")
        nodes: list[AudioNodeSpec] = []
        for node_index, raw_node in enumerate(nodes_raw):
            node = _require_mapping(raw_node, f"group {group_id}.nodes[{node_index}]")
            if not _optional_bool(node, "enable", True):
                continue
            nodes.append(_parse_node(node, group_id, node_index))
        groups.append(AudioGroupSpec(id=group_id, nodes=nodes))

    return AudioSystemSpec(
        default_start_delay=default_start_delay,
        inter_group_delay=inter_group_delay,
        groups=groups,
    )


def _parse_node(node, group_id: str, node_index: int) -> AudioNodeSpec:
    node_id = _required_text(node, "id", f"group {group_id}.nodes[{node_index}]")
    package = _required_text(node, "package", f"node {node_id}")
    executable = _required_text(node, "exec", f"node {node_id}")
    node_name = _optional_text(node, "node_name", node_id)
    namespace = _optional_text(node, "namespace", "")
    if namespace == "/":
        namespace = ""
    output = _optional_text(node, "output", "screen")
    params_file = _optional_text(node, "params_file", "").strip()
    if params_file and not os.path.isfile(params_file):
        raise RuntimeError(f"params_file not found: {params_file}")
    parameters = _optional_parameters(node.get("parameters", {}), node_id)
    if not params_file and not parameters:
        raise RuntimeError(f"node {node_id} requires params_file or parameters")
    return AudioNodeSpec(
        id=node_id,
        package=package,
        executable=executable,
        node_name=node_name,
        namespace=namespace,
        output=output,
        params_file=params_file,
        parameters=parameters,
    )


def _require_mapping(value, label: str):
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a mapping")
    return value


def _require_sequence(value, label: str):
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a sequence")
    return value


def _required_text(mapping, key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label}.{key} is required")
    return value.strip()


def _optional_text(mapping, key: str, default_value: str = "") -> str:
    value = mapping.get(key, default_value)
    if value is None:
        return default_value
    if not isinstance(value, str):
        raise RuntimeError(f"{key} must be a string")
    return value.strip()


def _optional_bool(mapping, key: str, default_value: bool = True) -> bool:
    value = mapping.get(key, default_value)
    if not isinstance(value, bool):
        raise RuntimeError(f"{key} must be a bool")
    return value


def _optional_float(mapping, key: str, default_value: float) -> float:
    value = mapping.get(key, default_value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{key} must be a number")
    return float(value)


def _optional_parameters(value, node_id: str) -> dict[str, ParamValue]:
    if value is None:
        return {}
    mapping = _require_mapping(value, f"node {node_id} parameters")
    params: dict[str, ParamValue] = {}
    for key, param_value in mapping.items():
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError(f"node {node_id} parameter keys must be non-empty strings")
        if not _is_param_value(param_value):
            raise RuntimeError(f"node {node_id} parameter '{key}' has unsupported value type")
        params[key.strip()] = param_value
    return params


def _is_param_value(value) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (str, int, float)):
        return True
    if not isinstance(value, list):
        return False
    if not value:
        return True
    if all(isinstance(item, bool) for item in value):
        return True
    if all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        return True
    if all(isinstance(item, float) for item in value):
        return True
    if all(isinstance(item, str) for item in value):
        return True
    return False
