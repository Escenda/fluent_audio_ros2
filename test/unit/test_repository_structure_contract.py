from pathlib import Path
from typing import TypeAlias

import yaml


REPO_ROOT = Path(__file__).parents[2]
SRC_ROOT = REPO_ROOT / "src"


YamlScalar: TypeAlias = str | int | float | bool | None
YamlMapping: TypeAlias = dict[str, "YamlValue"]
YamlSequence: TypeAlias = list["YamlValue"]
YamlValue: TypeAlias = YamlScalar | YamlMapping | YamlSequence
YamlPath: TypeAlias = tuple[str, ...]
YamlStringValue: TypeAlias = tuple[YamlPath, str]


def _collect_yaml_keys(value: YamlValue) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(_collect_yaml_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_collect_yaml_keys(child))
    return keys


def _collect_yaml_string_values(
    value: YamlValue,
    path: YamlPath = (),
) -> list[YamlStringValue]:
    values: list[YamlStringValue] = []
    if isinstance(value, dict):
        for key, child in value.items():
            values.extend(_collect_yaml_string_values(child, path + (key,)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            values.extend(_collect_yaml_string_values(child, path + (str(index),)))
    elif isinstance(value, str):
        values.append((path, value))
    return values


def _path_contains_topic_key(path: YamlPath) -> bool:
    return any(
        part in {"topic", "topics"}
        or part.endswith("_topic")
        or part.endswith("_topics")
        for part in path
    )


def _path_contains_stream_identity_key(path: YamlPath) -> bool:
    return any(
        part in {"stream_id", "stream_ids"}
        or part.endswith("_stream_id")
        or part.endswith("_stream_ids")
        or part.endswith(".stream_id")
        or part.endswith(".stream_ids")
        for part in path
    )


def test_config_files_do_not_use_legacy_backend_mapping_key() -> None:
    violations: list[str] = []

    for config_path in sorted(SRC_ROOT.rglob("config/*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        keys = _collect_yaml_keys(config)
        if "backend" in keys:
            violations.append(str(config_path.relative_to(REPO_ROOT)))

    assert violations == []


def test_config_files_keep_topic_and_stream_identities_distinct() -> None:
    violations: list[str] = []

    for config_path in sorted(SRC_ROOT.rglob("config/*.yaml")):
        config: YamlValue = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        string_values = _collect_yaml_string_values(config)
        topic_values = {
            value.lstrip("/")
            for path, value in string_values
            if _path_contains_topic_key(path)
        }
        for path, value in string_values:
            if not _path_contains_stream_identity_key(path):
                continue
            if value.lstrip("/") in topic_values:
                relative_path = config_path.relative_to(REPO_ROOT)
                yaml_path = ".".join(path)
                violations.append(f"{relative_path}:{yaml_path}={value}")

    assert violations == []
