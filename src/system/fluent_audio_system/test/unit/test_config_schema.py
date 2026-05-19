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


PACKAGE_ROOT = Path(__file__).parents[2]


def _valid_system() -> dict[str, float]:
    return {"default_start_delay": 0.0, "inter_group_delay": 0.0}


def test_parse_valid_config_with_params_file(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text(
        "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n",
        encoding="utf-8",
    )

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
                            "node_name": "fa_in",
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
    assert spec.groups[0].nodes[0].backend_name == "alsa_capture"


def test_node_backend_name_uses_effective_inline_override(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text(
        """
fa_in:
  ros__parameters:
    backend.name: alsa_capture
""",
        encoding="utf-8",
    )

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
                            "node_name": "fa_in",
                            "params_file": str(params_file),
                            "parameters": {"backend.name": "pcm_file_reader"},
                        }
                    ],
                }
            ],
        }
    )

    assert spec.groups[0].nodes[0].backend_name == "pcm_file_reader"


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
    fa_in_params.write_text(
        "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n",
        encoding="utf-8",
    )
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
                            "node_name": "fa_in",
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
                            "node_name": "fa_resample",
                            "params_file": str(resample_params),
                        },
                        {
                            "id": "fa_resample_duplicate",
                            "enable": True,
                            "package": "fa_resample",
                            "exec": "fa_resample_node",
                            "node_name": "fa_resample_duplicate",
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
                            "node_name": "fa_gain",
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


def test_load_required_packages_does_not_require_params_file_to_exist(
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
        node_name: fa_in
        params_file: /missing/fa_in.yaml
""",
        encoding="utf-8",
    )

    assert load_required_packages(str(system_file)) == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
    ]


def test_load_required_packages_requires_params_file_contract(tmp_path: Path) -> None:
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
        node_name: fa_in
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="node fa_in.params_file is required"):
        load_required_packages(str(system_file))


def test_load_required_packages_works_before_node_packages_are_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def package_share(package_name: str) -> str:
        raise RuntimeError(f"unexpected package share lookup: {package_name}")

    monkeypatch.setattr(config_schema, "_get_package_share_directory", package_share)

    packages = load_required_packages(
        str(PACKAGE_ROOT / "config" / "profiles" / "so101_mic_frontend.yaml")
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_sample_format",
        "fa_resample",
    ]


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
        node_name: fa_in
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
                            "node_name": "fa_frame_buffer",
                            "params_file": str(params_file),
                        }
                    ],
                }
            ],
        }
    )

    assert spec.groups[0].nodes[0].package == "fa_frame_buffer"


def test_ai_group_accepts_audio_embedding_package(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_audio_embedding.yaml"
    params_file.write_text(
        "fa_audio_embedding:\n  ros__parameters: {}\n",
        encoding="utf-8",
    )

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "ai",
                    "enable": True,
                    "nodes": [
                        {
                            "id": "fa_audio_embedding",
                            "enable": True,
                            "package": "fa_audio_embedding",
                            "exec": "fa_audio_embedding_node",
                            "node_name": "fa_audio_embedding",
                            "params_file": str(params_file),
                            "parameters": {
                                "input_topic": "audio/resample16k/mic",
                                "expected_stream_id": "audio/embedding/input_stream",
                            },
                        }
                    ],
                }
            ],
        }
    )

    node = spec.groups[0].nodes[0]
    assert node.package == "fa_audio_embedding"
    assert node.executable == "fa_audio_embedding_node"
    assert node.parameters == {
        "input_topic": "audio/resample16k/mic",
        "expected_stream_id": "audio/embedding/input_stream",
    }
    assert required_packages_for_system(spec) == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_audio_embedding",
    ]


@pytest.mark.parametrize(
    ("group_id", "package_name", "executable_name", "parameters"),
    [
        (
            "format",
            "fa_sample_format",
            "fa_sample_format_node",
            {
                "input_topic": "audio/sample_format/input",
                "input_stream_id": "audio/sample_format/input",
            },
        ),
        (
            "format",
            "fa_resample",
            "fa_resample_node",
            {
                "mic.input_topic": "/audio/resample/input",
                "mic.input_stream_id": "audio/resample/input",
            },
        ),
        (
            "ai",
            "fa_audio_embedding",
            "fa_audio_embedding_node",
            {
                "input_topic": "audio/embedding/input",
                "expected_stream_id": "audio/embedding/input",
            },
        ),
        (
            "routing",
            "fa_mix",
            "fa_mix_node",
            {
                "input_topics": ["audio/mix/input"],
                "input_stream_ids": ["/audio/mix/input"],
            },
        ),
        (
            "io",
            "fa_out",
            "fa_out_node",
            {
                "input_topic": "audio/output/frame",
                "input_stream_id": "audio/output/frame",
            },
        ),
    ],
)
def test_inline_parameters_reject_topic_values_used_as_stream_identities(
    tmp_path: Path,
    group_id: str,
    package_name: str,
    executable_name: str,
    parameters: dict[str, config_schema.ParamValue],
) -> None:
    params_file = tmp_path / f"{package_name}.yaml"
    params_file.write_text(f"{package_name}:\n  ros__parameters: {{}}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must not reuse ROS topic value"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": group_id,
                        "enable": True,
                        "nodes": [
                            {
                                "id": package_name,
                                "enable": True,
                                "package": package_name,
                                "exec": executable_name,
                                "node_name": package_name,
                                "params_file": str(params_file),
                                "parameters": parameters,
                            }
                        ],
                    }
                ],
            }
        )


def test_stream_identity_contract_rejects_non_string_role_values(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_mix.yaml"
    params_file.write_text("fa_mix:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="node fa_mix parameter 'input_stream_ids' must be a string or a list of strings",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "routing",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_mix",
                                "enable": True,
                                "package": "fa_mix",
                                "exec": "fa_mix_node",
                                "node_name": "fa_mix",
                                "params_file": str(params_file),
                                "parameters": {
                                    "input_topics": ["audio/mix/input"],
                                    "input_stream_ids": [1],
                                },
                            }
                        ],
                    }
                ],
            }
        )


def test_params_file_rejects_topic_values_used_as_stream_identities(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_sample_format.yaml"
    params_file.write_text(
        """
fa_sample_format:
  ros__parameters:
    input_topic: audio/sample_format/input
    input_stream_id: /audio/sample_format/input
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must not reuse ROS topic value"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "format",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_sample_format",
                                "enable": True,
                                "package": "fa_sample_format",
                                "exec": "fa_sample_format_node",
                                "node_name": "fa_sample_format",
                                "params_file": str(params_file),
                            }
                        ],
                    }
                ],
            }
        )


def test_params_file_flattens_nested_ros_parameters_for_identity_contract(
    tmp_path: Path,
) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text(
        """
fa_in:
  ros__parameters:
    output_topic: /audio/raw/mic
    audio:
      stream_id: audio/raw/mic
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must not reuse ROS topic value"):
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
                                "node_name": "fa_in",
                                "params_file": str(params_file),
                            }
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    "package_name, executable_name",
    [
        ("fa_in", "fa_in_node"),
        ("fa_out", "fa_out_node"),
        ("fa_record", "fa_record_node"),
        ("fa_stream", "fa_stream_node"),
    ],
)
def test_io_group_accepts_source_sink_and_io_utility_packages(
    tmp_path: Path,
    package_name: str,
    executable_name: str,
) -> None:
    params_file = tmp_path / f"{package_name}.yaml"
    if package_name == "fa_in":
        params_file.write_text(
            "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n",
            encoding="utf-8",
        )
    elif package_name == "fa_out":
        params_file.write_text(
            "fa_out:\n  ros__parameters:\n    backend.name: alsa_playback\n",
            encoding="utf-8",
        )
    else:
        params_file.write_text(
            f"{package_name}:\n  ros__parameters: {{}}\n",
            encoding="utf-8",
        )

    spec = parse_system_config(
        {
            "system": _valid_system(),
            "groups": [
                {
                    "id": "io",
                    "enable": True,
                    "nodes": [
                        {
                            "id": package_name,
                            "enable": True,
                            "package": package_name,
                            "exec": executable_name,
                            "node_name": package_name,
                            "params_file": str(params_file),
                        }
                    ],
                }
            ],
        }
    )

    assert spec.groups[0].nodes[0].package == package_name


@pytest.mark.parametrize("package_name", ["fa_file_in", "fa_file_out", "fa_network_in", "fa_network_out"])
def test_system_config_rejects_removed_or_unknown_packages_even_when_disabled(
    package_name: str,
) -> None:
    with pytest.raises(
        RuntimeError,
        match=f"group io contains unsupported FluentAudio package {package_name}",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
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
                            "node_name": "fa_tts",
                            "params_file": str(tts_params),
                        },
                        {
                            "id": "fa_mix",
                            "enable": True,
                            "package": "fa_mix",
                            "exec": "fa_mix_node",
                            "node_name": "fa_mix",
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
                            "node_name": "fa_log_mel",
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
                                "node_name": "fa_in",
                                "params_file": str(missing),
                            }
                        ],
                    }
                ]
            }
        )


def test_empty_params_file_content_fails_closed(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_gain.yaml"
    params_file.write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="must define ros__parameters for node fa_gain"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "dynamics",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_gain",
                                "enable": True,
                                "package": "fa_gain",
                                "exec": "fa_gain_node",
                                "node_name": "fa_gain",
                                "params_file": str(params_file),
                            }
                        ],
                    }
                ],
            }
        )


def test_params_file_without_matching_ros_parameters_fails_closed(
    tmp_path: Path,
) -> None:
    params_file = tmp_path / "fa_gain.yaml"
    params_file.write_text(
        "other_node:\n  ros__parameters:\n    gain.linear: 1.0\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="must define ros__parameters for node fa_gain"):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "dynamics",
                        "enable": True,
                        "nodes": [
                            {
                                "id": "fa_gain",
                                "enable": True,
                                "package": "fa_gain",
                                "exec": "fa_gain_node",
                                "node_name": "fa_gain",
                                "params_file": str(params_file),
                            }
                        ],
                    }
                ],
            }
        )


@pytest.mark.parametrize(
    ("package_name", "executable_name"),
    [
        ("fa_in", "fa_in_node"),
        ("fa_out", "fa_out_node"),
    ],
)
def test_source_and_sink_require_backend_name(
    tmp_path: Path,
    package_name: str,
    executable_name: str,
) -> None:
    params_file = tmp_path / f"{package_name}.yaml"
    params_file.write_text(
        f"{package_name}:\n  ros__parameters: {{}}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        RuntimeError,
        match=f"node {package_name}.backend.name is required for {package_name}",
    ):
        parse_system_config(
            {
                "system": _valid_system(),
                "groups": [
                    {
                        "id": "io",
                        "enable": True,
                        "nodes": [
                            {
                                "id": package_name,
                                "enable": True,
                                "package": package_name,
                                "exec": executable_name,
                                "node_name": package_name,
                                "params_file": str(params_file),
                            }
                        ],
                    }
                ],
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
                                "node_name": "fa_in",
                            }
                        ],
                    }
                ]
            }
        )


def test_node_requires_explicit_node_name(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="node fa_in.node_name is required"):
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
                            }
                        ],
                    }
                ],
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
                                "node_name": "fa_in",
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
                                "node_name": "fa_in",
                                "params_file": "",
                            }
                        ],
                    }
                ]
            }
        )


def test_nested_inline_parameters_fail(tmp_path: Path) -> None:
    params_file = tmp_path / "fa_in.yaml"
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")

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
                                "node_name": "fa_in",
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
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")

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
                                "node_name": "fa_in",
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
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")

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
                                "node_name": "fa_in",
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
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")

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
                                "node_name": "fa_in",
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
    params_file.write_text("fa_in:\n  ros__parameters: {}\n", encoding="utf-8")

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
                                "node_name": "fa_in",
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
    params_file.write_text(
        "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n",
        encoding="utf-8",
    )
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
        node_name: fa_in
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
                            "node_name": "fa_denoise",
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
        node_name: fa_tts
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
                                "node_name": "fa_tts",
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
                                "node_name": "fa_tts",
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
    params_file.write_text(
        "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n",
        encoding="utf-8",
    )

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
    params_file.write_text(
        "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n",
        encoding="utf-8",
    )

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
                                "node_name": "fa_in",
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
