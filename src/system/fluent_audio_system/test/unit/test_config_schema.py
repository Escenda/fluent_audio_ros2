from pathlib import Path

import pytest

from fluent_audio_system import config_schema
from fluent_audio_system.config_schema import (
    load_required_packages,
    load_system_config,
    parse_system_config,
    required_packages_for_system,
)
from fluent_audio_system.list_required_packages import main as list_required_packages_main


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


def test_required_packages_include_base_and_enabled_nodes_in_launch_order(
    tmp_path: Path,
) -> None:
    fa_in_params = tmp_path / "fa_in.yaml"
    resample_params = tmp_path / "fa_resample.yaml"
    gain_params = tmp_path / "fa_gain.yaml"
    fa_in_params.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")
    resample_params.write_text("fa_resample:\n  ros__parameters: {}\n", encoding="utf-8")
    gain_params.write_text("fa_gain:\n  ros__parameters: {}\n", encoding="utf-8")

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "io_format",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_in",
                            "enable": True,
                            "package": "fa_in",
                            "exec": "fa_in_node",
                            "params_file": str(fa_in_params),
                        },
                        {
                            "id": "fa_out",
                            "enable": False,
                            "package": "fa_out",
                            "exec": "fa_out_node",
                            "params_file": "/disabled/node/is/not/parsed.yaml",
                        },
                        {
                            "id": "fa_resample",
                            "enable": True,
                            "package": "fa_resample",
                            "exec": "fa_resample_node",
                            "params_file": str(resample_params),
                        },
                        {
                            "id": "fa_resample_duplicate",
                            "enable": True,
                            "package": "fa_resample",
                            "exec": "fa_resample_node",
                            "params_file": str(resample_params),
                        },
                    ],
                },
                {
                    "id": "dynamics",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_gain",
                            "enable": True,
                            "package": "fa_gain",
                            "exec": "fa_gain_node",
                            "params_file": str(gain_params),
                        }
                    ],
                },
            ],
        }
    )

    assert required_packages_for_system(spec) == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_resample",
        "fa_gain",
    ]


def test_load_required_packages_fails_closed_on_missing_params_file(
    tmp_path: Path,
) -> None:
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
        params_file: /missing/fa_in.yaml
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="params_file not found"):
        load_required_packages(str(system_file))


def test_list_required_packages_cli_prints_one_package_per_line(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")
    system_file = tmp_path / "system.yaml"
    system_file.write_text(
        f"""
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
        params_file: {params_file}
""",
        encoding="utf-8",
    )

    result = list_required_packages_main(["--config", str(system_file)])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.out.splitlines() == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
    ]
    assert captured.err == ""


def test_list_required_packages_cli_returns_error_for_invalid_config(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = list_required_packages_main(["--config", str(tmp_path / "missing.yaml")])
    captured = capsys.readouterr()

    assert result == 2
    assert captured.out == ""
    assert "config not found" in captured.err


@pytest.mark.parametrize(
    "package_name",
    [
        "fa_asr",
        "fa_audio_embedding",
        "fa_kws",
        "fa_sed",
        "fa_speaker",
        "fa_turn_detector",
        "fa_vad",
    ],
)
def test_analysis_group_rejects_ai_package_even_when_node_disabled(
    package_name: str,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=f"group analysis must not contain AI package {package_name}",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "analysis",
                        "enable": True,
                        "nodes": [
                            {
                                "id": package_name,
                                "enable": False,
                                "package": package_name,
                            }
                        ],
                    }
                ],
            }
        )


def test_analysis_group_rejects_streaming_package_even_when_node_disabled() -> None:
    with pytest.raises(
        RuntimeError,
        match="group analysis must not contain streaming package fa_frame_buffer",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "analysis",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_frame_buffer",
                                "enable": False,
                                "package": "fa_frame_buffer",
                            }
                        ],
                    }
                ],
            }
        )


def test_non_streaming_group_rejects_streaming_package_even_when_node_disabled() -> None:
    with pytest.raises(
        RuntimeError,
        match="group format must not contain streaming package fa_jitter_buffer",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "format",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_jitter_buffer",
                                "enable": False,
                                "package": "fa_jitter_buffer",
                            }
                        ],
                    }
                ],
            }
        )


def test_streaming_group_accepts_streaming_package(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_frame_buffer.yaml"
    params_file.write_text("fa_frame_buffer:\n  ros__parameters: {}\n", encoding="utf-8")

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "streaming",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_frame_buffer",
                            "enable": True,
                            "package": "fa_frame_buffer",
                            "exec": "fa_frame_buffer_node",
                            "params_file": str(params_file),
                        }
                    ],
                }
            ],
        }
    )

    assert spec.groups[0].nodes[0].package == "fa_frame_buffer"


def test_format_group_rejects_ai_package_even_when_node_disabled() -> None:
    with pytest.raises(
        RuntimeError,
        match="group format must not contain fa_vad; package category is ai",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "format",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_vad",
                                "enable": False,
                                "package": "fa_vad",
                            }
                        ],
                    }
                ],
            }
        )


def test_ai_group_rejects_format_package_even_when_node_disabled() -> None:
    with pytest.raises(
        RuntimeError,
        match="group ai must not contain fa_resample; package category is format",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "ai",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_resample",
                                "enable": False,
                                "package": "fa_resample",
                            }
                        ],
                    }
                ],
            }
        )


def test_audio_correction_group_rejects_dynamics_package_even_when_node_disabled() -> None:
    with pytest.raises(
        RuntimeError,
        match="group audio_correction must not contain fa_gain; package category is dynamics",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "audio_correction",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_gain",
                                "enable": False,
                                "package": "fa_gain",
                            }
                        ],
                    }
                ],
            }
        )


def test_generation_routing_group_accepts_generation_and_routing_packages(
    tmp_path: Path,
) -> None:
    tts_params = tmp_path / "fa_tts.yaml"
    mix_params = tmp_path / "fa_mix.yaml"
    tts_params.write_text("fa_tts:\n  ros__parameters: {}\n", encoding="utf-8")
    mix_params.write_text("fa_mix:\n  ros__parameters: {}\n", encoding="utf-8")

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "generation_routing",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_tts",
                            "enable": True,
                            "package": "fa_tts",
                            "exec": "fa_tts_node",
                            "params_file": str(tts_params),
                        },
                        {
                            "id": "fa_mix",
                            "enable": True,
                            "package": "fa_mix",
                            "exec": "fa_mix_node",
                            "params_file": str(mix_params),
                        },
                    ],
                }
            ],
        }
    )

    assert [node.package for node in spec.groups[0].nodes] == ["fa_tts", "fa_mix"]


def test_analysis_group_accepts_non_ai_feature_package(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_log_mel.yaml"
    params_file.write_text("fa_log_mel:\n  ros__parameters: {}\n", encoding="utf-8")

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "analysis",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_log_mel",
                            "enable": True,
                            "package": "fa_log_mel",
                            "exec": "fa_log_mel_node",
                            "params_file": str(params_file),
                        }
                    ],
                }
            ],
        }
    )

    assert spec.groups[0].nodes[0].package == "fa_log_mel"


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


def test_from_to_sequence_remappings_fail(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="remappings"):
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
                                "remappings": [
                                    {"from": "audio/frame", "to": "robot/audio/frame"}
                                ],
                            }
                        ],
                    }
                ]
            }
        )


def test_pair_sequence_remappings_fail(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="remappings"):
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
                                "remappings": [["audio/frame", "robot/audio/frame"]],
                            }
                        ],
                    }
                ],
            }
        )


def test_invalid_pair_sequence_remappings_fail(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="remappings"):
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
                                "remappings": [["audio/frame"]],
                            }
                        ],
                    }
                ],
            }
        )


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


def test_environment_path_expansion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "config"
    params_dir = tmp_path / "params"
    dictionary_dir = tmp_path / "open_jtalk_dic"
    config_dir.mkdir()
    params_dir.mkdir()
    dictionary_dir.mkdir()

    params_file = params_dir / "fa_tts.yaml"
    params_file.write_text("fa_tts:\n  ros__parameters: {}\n", encoding="utf-8")
    system_file = config_dir / "system.yaml"
    system_file.write_text(
        """
system:
  default_start_delay: 0.0
  inter_group_delay: 0.0
groups:
  - id: generation
    enable: true
    nodes:
      - id: fa_tts
        enable: true
        package: fa_tts
        exec: fa_tts_node
        params_file: "${env:FA_TEST_PARAMS_DIR}/fa_tts.yaml"
        parameters:
          backend.openjtalk_dict_dir: "${env:FA_TEST_OPENJTALK_DICT_DIR}"
          model_paths:
            - "${env:FA_TEST_OPENJTALK_DICT_DIR}/left"
            - "${env:FA_TEST_OPENJTALK_DICT_DIR}/right"
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("FA_TEST_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("FA_TEST_PARAMS_DIR", str(params_dir))
    monkeypatch.setenv("FA_TEST_OPENJTALK_DICT_DIR", str(dictionary_dir))

    spec = load_system_config("${env:FA_TEST_CONFIG_DIR}/system.yaml")

    node = spec.groups[0].nodes[0]
    assert node.params_file == str(params_file)
    assert node.parameters == {
        "backend.openjtalk_dict_dir": str(dictionary_dir),
        "model_paths": [
            str(dictionary_dir / "left"),
            str(dictionary_dir / "right"),
        ],
    }


def test_missing_environment_path_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    params_file = tmp_path / "fa_tts.yaml"
    params_file.write_text("fa_tts:\n  ros__parameters: {}\n", encoding="utf-8")
    monkeypatch.delenv("FA_TEST_OPENJTALK_DICT_DIR", raising=False)

    with pytest.raises(
        RuntimeError,
        match="environment variable FA_TEST_OPENJTALK_DICT_DIR is required",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "generation",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_tts",
                                "enable": True,
                                "package": "fa_tts",
                                "exec": "fa_tts_node",
                                "params_file": str(params_file),
                                "parameters": {
                                    "backend.openjtalk_dict_dir": (
                                        "${env:FA_TEST_OPENJTALK_DICT_DIR}"
                                    ),
                                },
                            }
                        ],
                    }
                ],
            }
        )


def test_empty_environment_path_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    params_file = tmp_path / "fa_tts.yaml"
    params_file.write_text("fa_tts:\n  ros__parameters: {}\n", encoding="utf-8")
    monkeypatch.setenv("FA_TEST_OPENJTALK_DICT_DIR", "  ")

    with pytest.raises(
        RuntimeError,
        match="environment variable FA_TEST_OPENJTALK_DICT_DIR is required",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "generation",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_tts",
                                "enable": True,
                                "package": "fa_tts",
                                "exec": "fa_tts_node",
                                "params_file": str(params_file),
                                "parameters": {
                                    "backend.openjtalk_dict_dir": (
                                        "${env:FA_TEST_OPENJTALK_DICT_DIR}"
                                    ),
                                },
                            }
                        ],
                    }
                ],
            }
        )


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


def test_node_executable_field_name_is_rejected(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in_node:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="executable"):
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
                                "executable": "fa_in_node",
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
