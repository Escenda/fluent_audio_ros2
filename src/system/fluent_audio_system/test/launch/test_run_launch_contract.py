from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_run_launch_includes_system_launch_with_site_binding_args() -> None:
    run_text = (PACKAGE_ROOT / "launch" / "run.py").read_text(encoding="utf-8")

    assert "IncludeLaunchDescription" in run_text
    assert "PythonLaunchDescriptionSource" in run_text
    assert '"fluent_audio_system.launch.py"' in run_text
    assert '"config": config_path' in run_text
    for arg_name in ("fa_in_enabled", "fa_out_enabled", "fa_in_source_id", "fa_out_sink_id"):
        assert f'"{arg_name}": {arg_name}' in run_text


def test_system_entrypoints_require_explicit_launch_arguments() -> None:
    launch_text = (
        PACKAGE_ROOT / "launch" / "fluent_audio_system.launch.py"
    ).read_text(encoding="utf-8")
    run_text = (PACKAGE_ROOT / "launch" / "run.py").read_text(encoding="utf-8")
    combined = launch_text + "\n" + run_text

    assert "default_value" not in combined
    assert "/config/fluent_audio_system.yaml" not in combined
    for arg_name in (
        "config",
        "fa_in_enabled",
        "fa_out_enabled",
        "fa_in_source_id",
        "fa_out_sink_id",
    ):
        assert f'"{arg_name}"' in launch_text
        assert f'"{arg_name}"' in run_text


def test_system_launch_uses_node_actions_without_temp_yaml_rewrite() -> None:
    launch_text = (
        PACKAGE_ROOT / "launch" / "fluent_audio_system.launch.py"
    ).read_text(encoding="utf-8")

    assert "Node(" in launch_text
    assert "parameters=_node_launch_parameters(node, overrides)" in launch_text
    assert "remappings=node.launch_remappings()" in launch_text
    assert "tempfile" not in launch_text
    assert "NamedTemporaryFile" not in launch_text
    assert "safe_dump" not in launch_text


def test_vlabor_entrypoints_do_not_accept_backend_or_model_site_args() -> None:
    launch_text = (
        PACKAGE_ROOT / "launch" / "fluent_audio_system.launch.py"
    ).read_text(encoding="utf-8")
    run_text = (PACKAGE_ROOT / "launch" / "run.py").read_text(encoding="utf-8")
    combined = launch_text + "\n" + run_text

    for forbidden_arg in (
        "backend.name",
        "backend.command",
        "backend.model",
        "backend.model_id",
        "backend.model_path",
        "embedding.dimension",
        "vad_threshold",
        "asr_prompt",
        "openai_api_key",
    ):
        assert forbidden_arg not in combined


def test_fixture_files_document_launch_contract() -> None:
    fixture_dir = PACKAGE_ROOT / "test" / "fixtures"

    for fixture_name in (
        "valid_io_system.yaml",
        "missing_params_system.yaml",
        "invalid_schema_system.yaml",
        "remapping_system.yaml",
        "fa_in.params.yaml",
        "fa_out.params.yaml",
        "fa_vad.params.yaml",
        "vlabor_include_action.yaml",
    ):
        assert (fixture_dir / fixture_name).is_file()


def test_profile_configs_are_package_owned_and_installed() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")

    assert (PACKAGE_ROOT / "config" / "profiles" / "so101.yaml").is_file()
    assert (PACKAGE_ROOT / "config" / "profiles" / "so101_mic_frontend.yaml").is_file()
    assert (PACKAGE_ROOT / "config" / "profiles" / "so101_tts_output.yaml").is_file()
    assert '"/config/profiles"' in setup_text
    assert 'files_in_tree("config/profiles")' in setup_text


def test_required_package_cli_is_installed_for_vlabor_build_resolution() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    setup_cfg_text = (PACKAGE_ROOT / "setup.cfg").read_text(encoding="utf-8")

    assert (
        "list_required_packages = fluent_audio_system.list_required_packages:main"
        in setup_text
    )
    assert "script_dir=$base/lib/fluent_audio_system" in setup_cfg_text
    assert "install_scripts=$base/lib/fluent_audio_system" in setup_cfg_text


def test_vlabor_include_action_matches_fluent_vision_profile_shape() -> None:
    fixture_path = PACKAGE_ROOT / "test" / "fixtures" / "vlabor_include_action.yaml"
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    action = fixture["action"]

    assert action["type"] == "include"
    assert action["package"] == "fluent_audio_system"
    assert action["launch"] == "run.py"
    assert action["enabled"] == "${fluent_audio_enabled}"
    assert set(action["args"]) == {
        "config",
        "fa_in_enabled",
        "fa_out_enabled",
        "fa_in_source_id",
        "fa_out_sink_id",
    }
    assert action["args"] == {
        "config": "${fluent_audio_config}",
        "fa_in_enabled": "${fluent_audio_fa_in_enabled}",
        "fa_out_enabled": "${fluent_audio_fa_out_enabled}",
        "fa_in_source_id": "${fluent_audio_fa_in_source_id}",
        "fa_out_sink_id": "${fluent_audio_fa_out_sink_id}",
    }


def test_vlabor_include_action_does_not_expose_backend_or_model_args() -> None:
    fixture_path = PACKAGE_ROOT / "test" / "fixtures" / "vlabor_include_action.yaml"
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    action = fixture["action"]
    serialized_args = "\n".join(
        [*action["args"].keys(), *[str(value) for value in action["args"].values()]]
    )

    for forbidden_token in (
        "backend",
        "model",
        "threshold",
        "prompt",
        "timeout",
        "endpoint",
        "api_key",
        "secret",
        "openai",
    ):
        assert forbidden_token not in serialized_args
