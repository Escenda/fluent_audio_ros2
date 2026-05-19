from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_explicit_config_file_contract() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_vad.launch.py").read_text(
        encoding="utf-8"
    )

    assert "default_value" not in launch_text
    forbidden_find_package_share = "Find" + "PackageShare"
    forbidden_path_join_substitution = "PathJoin" + "Substitution"
    assert forbidden_find_package_share not in launch_text
    assert forbidden_path_join_substitution not in launch_text
    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "parameters=[config_file]" in launch_text


def test_default_config_requires_explicit_backend_and_external_worker_command() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    assert "fa_vad" in config
    assert "fa_vad_node" not in config
    params = config["fa_vad"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["input_topic"] == "audio/frame"
    assert params["input_stream_id"] == "audio/raw/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["backend.command"] == ""
    assert params["backend.frame_ms"] == 20
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["expected_source_id"] == ""
    assert params["backend.timeout_sec"] > 0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["qos.depth"] == 10
    assert params["qos.reliable"] is False

    rendered_args = " ".join(params["backend.args"])
    for placeholder in ("{audio}", "{model}", "{provider}", "{sample_rate}"):
        assert placeholder in rendered_args


def test_reference_worker_script_is_thin_backend_entrypoint() -> None:
    script_text = (PACKAGE_ROOT / "scripts" / "silero_vad_worker").read_text(
        encoding="utf-8"
    )
    backend_worker = PACKAGE_ROOT / "fa_vad_py" / "backends" / "silero_worker.py"

    assert backend_worker.is_file()
    assert "from fa_vad_py.backends.silero_worker import main" in script_text
    assert "torch.hub.load" not in script_text
