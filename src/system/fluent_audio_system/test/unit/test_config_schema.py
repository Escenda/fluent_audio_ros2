from pathlib import Path

import pytest

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
