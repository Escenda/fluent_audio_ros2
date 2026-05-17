from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_run_launch_includes_system_launch_with_site_binding_args() -> None:
    run_text = (PACKAGE_ROOT / "launch" / "run.py").read_text(encoding="utf-8")

    assert "IncludeLaunchDescription" in run_text
    assert "PythonLaunchDescriptionSource" in run_text
    assert '"fluent_audio_system.launch.py"' in run_text
    for arg_name in (
        "config",
        "fa_in_enabled",
        "fa_out_enabled",
        "fa_in_source_id",
        "fa_out_sink_id",
    ):
        assert f'"{arg_name}": {arg_name}' in run_text


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
        "backend.model",
        "backend.model_path",
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
    ):
        assert (fixture_dir / fixture_name).is_file()


def test_profile_configs_are_package_owned_and_installed() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")

    assert (PACKAGE_ROOT / "config" / "profiles" / "so101.yaml").is_file()
    assert (PACKAGE_ROOT / "config" / "profiles" / "so101_mic_frontend.yaml").is_file()
    assert (PACKAGE_ROOT / "config" / "profiles" / "so101_tts_output.yaml").is_file()
    assert '"/config/profiles"' in setup_text
    assert 'files_in_tree("config/profiles")' in setup_text
