from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypeGuard

import yaml


ScalarParam = str | int | float | bool
ParamValue = ScalarParam | list[str] | list[int] | list[float] | list[bool]
ConfigScalar: TypeAlias = str | int | float | bool | None
ConfigMapping: TypeAlias = dict[str, "ConfigValue"]
ConfigSequence: TypeAlias = list["ConfigValue"]
ConfigValue: TypeAlias = ConfigScalar | ConfigMapping | ConfigSequence
_INLINE_SHARE_RE = re.compile(r"\$\{share:([A-Za-z0-9_]+)\}")


@dataclass(frozen=True)
class RemappingSpec:
    source: str
    target: str


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
    remappings: list[RemappingSpec]

    def launch_parameters(self):
        sources = []
        if self.params_file:
            sources.append(self.params_file)
        if self.parameters:
            sources.append(self.parameters)
        return sources

    def launch_remappings(self):
        return [(item.source, item.target) for item in self.remappings]


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
    path = _resolve_share_refs(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"fluent_audio_system config not found: {path}")
    with open(path, "r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return parse_system_config(raw)


def parse_system_config(raw: ConfigValue) -> AudioSystemSpec:
    root = _require_mapping(raw, "fluent_audio_system config root")
    system = _require_mapping(
        _required_key(root, "system", "fluent_audio_system config root"),
        "system",
    )
    groups_raw = _require_sequence(
        _required_key(root, "groups", "fluent_audio_system config root"),
        "groups",
    )
    default_start_delay = _optional_float(system, "default_start_delay", 0.0)
    inter_group_delay = _optional_float(system, "inter_group_delay", 0.0)
    if default_start_delay < 0.0:
        raise RuntimeError("system.default_start_delay must be >= 0")
    if inter_group_delay < 0.0:
        raise RuntimeError("system.inter_group_delay must be >= 0")

    groups: list[AudioGroupSpec] = []
    for group_index, raw_group in enumerate(groups_raw):
        group = _require_mapping(raw_group, f"groups[{group_index}]")
        group_id = _required_text(group, "id", f"groups[{group_index}]")
        if not _optional_bool(group, "enable", True):
            continue
        nodes_raw = _require_sequence(
            _required_key(group, "nodes", f"group {group_id}"),
            f"group {group_id}.nodes",
        )
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


def _parse_node(node: ConfigMapping, group_id: str, node_index: int) -> AudioNodeSpec:
    node_id = _required_text(node, "id", f"group {group_id}.nodes[{node_index}]")
    package = _required_text(node, "package", f"node {node_id}")
    executable = _required_text(node, "exec", f"node {node_id}")
    node_name = _optional_text(node, "node_name", node_id)
    namespace = _optional_text(node, "namespace", "")
    if namespace == "/":
        namespace = ""
    output = _optional_text(node, "output", "screen")
    params_file = _resolve_share_refs(_optional_text(node, "params_file", "").strip())
    if params_file and not os.path.isfile(params_file):
        raise RuntimeError(f"params_file not found: {params_file}")
    parameters = _optional_parameters(node.get("parameters", {}), node_id)
    remappings = _optional_remappings(node.get("remappings", []), node_id)
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
        remappings=remappings,
    )


def _require_mapping(value: ConfigValue, label: str) -> ConfigMapping:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a mapping")
    for key in value:
        if not isinstance(key, str):
            raise RuntimeError(f"{label} keys must be strings")
    return value


def _require_sequence(value: ConfigValue, label: str) -> ConfigSequence:
    if not isinstance(value, list):
        raise RuntimeError(f"{label} must be a sequence")
    return value


def _required_key(mapping: ConfigMapping, key: str, label: str) -> ConfigValue:
    if key not in mapping:
        raise RuntimeError(f"{label}.{key} is required")
    return mapping[key]


def _required_text(mapping: ConfigMapping, key: str, label: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{label}.{key} is required")
    return value.strip()


def _optional_text(mapping: ConfigMapping, key: str, default_value: str = "") -> str:
    value = mapping.get(key, default_value)
    if value is None:
        return default_value
    if not isinstance(value, str):
        raise RuntimeError(f"{key} must be a string")
    return value.strip()


def _optional_bool(mapping: ConfigMapping, key: str, default_value: bool = True) -> bool:
    value = mapping.get(key, default_value)
    if not isinstance(value, bool):
        raise RuntimeError(f"{key} must be a bool")
    return value


def _optional_float(mapping: ConfigMapping, key: str, default_value: float) -> float:
    value = mapping.get(key, default_value)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{key} must be a number")
    return float(value)


def _optional_parameters(value: ConfigValue, node_id: str) -> dict[str, ParamValue]:
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


def _optional_remappings(value: ConfigValue, node_id: str) -> list[RemappingSpec]:
    if value is None:
        return []
    if isinstance(value, dict):
        remappings: list[RemappingSpec] = []
        for source, target in value.items():
            if not isinstance(source, str) or not source.strip():
                raise RuntimeError(f"node {node_id} remapping source must be a non-empty string")
            if not isinstance(target, str) or not target.strip():
                raise RuntimeError(f"node {node_id} remapping target must be a non-empty string")
            remappings.append(RemappingSpec(source=source.strip(), target=target.strip()))
        return remappings
    sequence = _require_sequence(value, f"node {node_id} remappings")
    remappings: list[RemappingSpec] = []
    for index, item in enumerate(sequence):
        mapping = _require_mapping(item, f"node {node_id} remappings[{index}]")
        source = _required_text(mapping, "from", f"node {node_id} remappings[{index}]")
        target = _required_text(mapping, "to", f"node {node_id} remappings[{index}]")
        remappings.append(RemappingSpec(source=source, target=target))
    return remappings


def _resolve_share_refs(value: str) -> str:
    def _replace(match_obj: re.Match[str]) -> str:
        return _get_package_share_directory(match_obj.group(1))

    return _INLINE_SHARE_RE.sub(_replace, value)


def _get_package_share_directory(package_name: str) -> str:
    for prefix in os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep):
        if not prefix:
            continue
        candidate = Path(prefix) / "share" / package_name
        if candidate.is_dir():
            return str(candidate)
    raise RuntimeError(f"package share directory not found: {package_name}")


def _is_param_value(value: ConfigValue) -> TypeGuard[ParamValue]:
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
