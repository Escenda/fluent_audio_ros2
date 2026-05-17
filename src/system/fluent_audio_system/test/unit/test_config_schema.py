from pathlib import Path

import pytest

from fluent_audio_system import config_schema
from fluent_audio_system.config_schema import load_system_config, parse_system_config


def test_parse_valid_inline_config() -> None:
    spec = parse_system_config(
        {
            "system": {"default_start_delay": 0.2, "inter_group_delay": 0.5},
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_in",
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "remappings": {"audio/frame": "robot/audio/frame"},
                            "parameters": {
                                "audio.sample_rate": 48000,
                                "audio.encoding": "PCM16LE",
                                "audio.qos.reliable": True,
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert spec.default_start_delay == 0.2
    assert spec.inter_group_delay == 0.5
    assert len(spec.groups) == 1
    assert spec.groups[0].nodes[0].package == "fa_in"
    assert spec.groups[0].nodes[0].launch_remappings() == [
        ("audio/frame", "robot/audio/frame")
    ]
    assert spec.groups[0].nodes[0].launch_parameters() == [
        {
            "audio.sample_rate": 48000,
            "audio.encoding": "PCM16LE",
            "audio.qos.reliable": True,
        }
    ]


def test_disabled_nodes_are_not_expanded() -> None:
    spec = parse_system_config(
        {
            "system": {},
            "groups": [
                {
                    "id": "io",
                    "nodes": [
                        {
                            "id": "fa_out",
                            "enable": False,
                            "package": "fa_out",
                            "exec": "fa_out_node",
                            "parameters": {"audio.device_id": "default"},
                        }
                    ],
                }
            ]
        }
    )

    assert len(spec.groups) == 1
    assert spec.groups[0].nodes == []


def test_missing_params_file_fails(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(RuntimeError, match="params_file not found"):
        parse_system_config(
            {
                "system": {},
                "groups": [
                    {
                        "id": "io",
                        "nodes": [
                            {
                                "id": "fa_in",
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "params_file": str(missing),
                            }
                        ],
                    }
                ]
            }
        )


def test_node_requires_params_file_or_inline_parameters() -> None:
    with pytest.raises(RuntimeError, match="requires params_file or parameters"):
        parse_system_config(
            {
                "system": {},
                "groups": [
                    {
                        "id": "io",
                        "nodes": [{"id": "fa_in", "package": "fa_in", "exec": "fa_in_node"}],
                    }
                ]
            }
        )


def test_nested_inline_parameters_fail() -> None:
    with pytest.raises(RuntimeError, match="unsupported value type"):
        parse_system_config(
            {
                "system": {},
                "groups": [
                    {
                        "id": "io",
                        "nodes": [
                            {
                                "id": "fa_in",
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "parameters": {"audio": {"sample_rate": 48000}},
                            }
                        ],
                    }
                ]
            }
        )


def test_load_missing_config_fails(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="config not found"):
        load_system_config(str(tmp_path / "missing.yaml"))


def test_sequence_remappings_are_supported() -> None:
    spec = parse_system_config(
        {
            "system": {},
            "groups": [
                {
                    "id": "io",
                    "nodes": [
                        {
                            "id": "fa_in",
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "remappings": [
                                {"from": "audio/frame", "to": "robot/audio/frame"}
                            ],
                            "parameters": {"audio.sample_rate": 48000},
                        }
                    ],
                }
            ]
        }
    )

    assert spec.groups[0].nodes[0].launch_remappings() == [
        ("audio/frame", "robot/audio/frame")
    ]


def test_invalid_remappings_fail() -> None:
    with pytest.raises(RuntimeError, match="remapping target"):
        parse_system_config(
            {
                "system": {},
                "groups": [
                    {
                        "id": "io",
                        "nodes": [
                            {
                                "id": "fa_in",
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "remappings": {"audio/frame": ""},
                                "parameters": {"audio.sample_rate": 48000},
                            }
                        ],
                    }
                ]
            }
        )


def test_share_path_expansion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")
    system_file = tmp_path / "system.yaml"
    system_file.write_text(
        """
system: {}
groups:
  - id: io
    nodes:
      - id: fa_in
        package: fa_in
        exec: fa_in_node
        params_file: "${share:demo_pkg}/fa_in.yaml"
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        config_schema,
        "_get_package_share_directory",
        lambda package_name: str(tmp_path),
    )

    spec = load_system_config("${share:demo_pkg}/system.yaml")

    assert spec.groups[0].nodes[0].params_file == str(params_file)


def test_inline_parameter_share_path_expansion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        config_schema,
        "_get_package_share_directory",
        lambda package_name: str(tmp_path / package_name),
    )

    spec = parse_system_config(
        {
            "system": {},
            "groups": [
                {
                    "id": "correction",
                    "nodes": [
                        {
                            "id": "fa_denoise",
                            "package": "fa_denoise",
                            "exec": "fa_denoise_node",
                            "parameters": {
                                "dtln.model_1_path": "${share:fa_denoise}/models/model_1.onnx",
                                "model_paths": [
                                    "${share:fa_denoise}/models/model_1.onnx",
                                    "${share:fa_denoise}/models/model_2.onnx",
                                ],
                            },
                        }
                    ],
                }
            ],
        }
    )

    assert spec.groups[0].nodes[0].parameters == {
        "dtln.model_1_path": str(tmp_path / "fa_denoise" / "models" / "model_1.onnx"),
        "model_paths": [
            str(tmp_path / "fa_denoise" / "models" / "model_1.onnx"),
            str(tmp_path / "fa_denoise" / "models" / "model_2.onnx"),
        ],
    }


def test_missing_system_fails() -> None:
    with pytest.raises(RuntimeError, match="system is required"):
        parse_system_config({"groups": []})


def test_missing_groups_fails() -> None:
    with pytest.raises(RuntimeError, match="groups is required"):
        parse_system_config({"system": {}})


def test_enabled_group_requires_nodes() -> None:
    with pytest.raises(RuntimeError, match="group io.nodes is required"):
        parse_system_config({"system": {}, "groups": [{"id": "io"}]})


def test_negative_delays_fail() -> None:
    with pytest.raises(RuntimeError, match="default_start_delay must be >= 0"):
        parse_system_config({"system": {"default_start_delay": -0.1}, "groups": []})
    with pytest.raises(RuntimeError, match="inter_group_delay must be >= 0"):
        parse_system_config({"system": {"inter_group_delay": -0.1}, "groups": []})
