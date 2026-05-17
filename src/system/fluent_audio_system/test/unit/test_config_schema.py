from pathlib import Path

import pytest

from fluent_audio_system import config_schema
from fluent_audio_system.config_schema import load_system_config, parse_system_config


def _valid_system() -> dict[str, float]:
    return {"default_start_delay": 0.0, "inter_group_delay": 0.0}


def test_parse_valid_config_with_params_file(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

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
                            "enable": True,
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "params_file": str(params_file),
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
        str(params_file),
        {
            "audio.sample_rate": 48000,
            "audio.encoding": "PCM16LE",
            "audio.qos.reliable": True,
        }
    ]


def test_disabled_nodes_are_not_expanded() -> None:
    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_out",
                            "enable": False,
                            "package": "fa_out",
                            "exec": "fa_out_node",
                            "params_file": "/disabled/node/is/not/parsed.yaml",
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
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "enable": True,
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "params_file": str(missing),
                            }
                        ],
                    }
                ]
            }
        )


def test_node_requires_params_file() -> None:
    with pytest.raises(RuntimeError, match="node fa_in.params_file is required"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "enable": True,
                                "package": "fa_in",
                                "exec": "fa_in_node",
                            }
                        ],
                    }
                ]
            }
        )


def test_inline_parameters_without_params_file_fail() -> None:
    with pytest.raises(RuntimeError, match="node fa_in.params_file is required"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "enable": True,
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "parameters": {"audio.sample_rate": 48000},
                            }
                        ],
                    }
                ]
            }
        )


def test_empty_params_file_fails() -> None:
    with pytest.raises(RuntimeError, match="node fa_in.params_file is required"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "enable": True,
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "params_file": "",
                            }
                        ],
                    }
                ]
            }
        )


def test_nested_inline_parameters_fail(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="unsupported value type"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "enable": True,
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "params_file": str(params_file),
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


def test_sequence_remappings_are_supported(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_in",
                            "enable": True,
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "params_file": str(params_file),
                            "remappings": [
                                {"from": "audio/frame", "to": "robot/audio/frame"}
                            ],
                        }
                    ],
                }
            ]
        }
    )

    assert spec.groups[0].nodes[0].launch_remappings() == [
        ("audio/frame", "robot/audio/frame")
    ]


def test_invalid_remappings_fail(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="remapping target"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "enable": True,
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "params_file": str(params_file),
                                "remappings": {"audio/frame": ""},
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
system:
  default_start_delay: 0.0
  inter_group_delay: 0.0
groups:
  - id: io
    enable: true
    nodes:
      - id: fa_in
        enable: true
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
    params_file = tmp_path / "fa_denoise.yaml"
    params_file.write_text("fa_denoise:\n  ros__parameters: {}\n", encoding="utf-8")
    monkeypatch.setattr(
        config_schema,
        "_get_package_share_directory",
        lambda package_name: str(tmp_path / package_name)
        if package_name != "params_pkg"
        else str(tmp_path),
    )

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "correction",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_denoise",
                            "enable": True,
                            "package": "fa_denoise",
                            "exec": "fa_denoise_node",
                            "params_file": "${share:params_pkg}/fa_denoise.yaml",
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
        parse_system_config({"system": _valid_system()})


def test_enabled_group_requires_nodes() -> None:
    with pytest.raises(RuntimeError, match="group io.nodes is required"):
        parse_system_config({"system": _valid_system(), "groups": [{"id": "io", "enable": True}]})


def test_disabled_group_does_not_require_nodes() -> None:
    spec = parse_system_config({"system": _valid_system(), "groups": [{"id": "io", "enable": False}]})

    assert spec.groups == []


def test_group_enable_is_required() -> None:
    with pytest.raises(RuntimeError, match="group io.enable is required"):
        parse_system_config({"system": _valid_system(), "groups": [{"id": "io", "nodes": []}]})


def test_node_enable_is_required(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="node fa_in.enable is required"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_in",
                                "package": "fa_in",
                                "exec": "fa_in_node",
                                "params_file": str(params_file),
                            }
                        ],
                    }
                ],
            }
        )


def test_negative_delays_fail() -> None:
    with pytest.raises(RuntimeError, match="default_start_delay must be >= 0"):
        parse_system_config(
            {
                "system": {"default_start_delay": -0.1, "inter_group_delay": 0.0},
                "groups": [],
            }
        )
    with pytest.raises(RuntimeError, match="inter_group_delay must be >= 0"):
        parse_system_config(
            {
                "system": {"default_start_delay": 0.0, "inter_group_delay": -0.1},
                "groups": [],
            }
        )


def test_config_schema_uses_pydantic_boundary() -> None:
    source = Path(config_schema.__file__).read_text(encoding="utf-8")

    assert "BaseModel" in source
    assert "_AudioSystemConfig.model_validate(raw)" in source
    assert "dict[str, Any]" not in source
    assert ": object" not in source


def test_system_delays_are_required() -> None:
    with pytest.raises(RuntimeError, match="system.default_start_delay is required"):
        parse_system_config(
            {
                "system": {"inter_group_delay": 0.0},
                "groups": [],
            }
        )
    with pytest.raises(RuntimeError, match="system.inter_group_delay is required"):
        parse_system_config(
            {
                "system": {"default_start_delay": 0.0},
                "groups": [],
            }
        )
