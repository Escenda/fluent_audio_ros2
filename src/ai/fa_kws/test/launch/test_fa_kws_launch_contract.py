from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_explicit_config_file_contract() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_kws.launch.py").read_text(
        encoding="utf-8"
    )

    assert "default_value" not in launch_text
    assert "get_package_share_directory" not in launch_text
    assert 'DeclareLaunchArgument(\n                "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n                "config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert 'LaunchConfiguration("config"' not in launch_text


def test_default_config_requires_explicit_backend_and_kws_inputs() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_kws"]["ros__parameters"]

    assert params["backend.name"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert params["backend.timeout_sec"] > 0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["model.encoder"] == ""
    assert params["model.decoder"] == ""
    assert params["model.joiner"] == ""
    assert params["model.tokens"] == ""
    assert params["kws.keywords_file"] == ""
    assert params["audio_topic"] == "audio/frame"
    assert params["expected_stream_id"] == "audio/raw/mic"
    assert params["audio_topic"] != params["expected_stream_id"]
    assert params["expected_source_id"] == ""
    assert params["vad.probability_gate"] == 0.35
    assert params["vad.max_age_ms"] > 0
    assert params["audio.qos.depth"] > 0
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] > 0
    assert params["vad.qos.reliable"] is False
    assert params["output.qos.depth"] > 0
    assert params["output.qos.reliable"] is False
    assert "dump_audio.enable" not in params
    assert "dump_audio.path" not in params


def test_launch_does_not_embed_backend_or_model_fallback() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_kws.launch.py").read_text(
        encoding="utf-8"
    )

    assert "sherpa_onnx_kws" not in launch_text
    assert "model.encoder" not in launch_text
    assert "backend.execution_provider" not in launch_text
    assert "backend.command" not in launch_text
    assert "cpu" not in launch_text


def test_launch_and_install_use_only_declared_kws_node_executable() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_kws.launch.py").read_text(
        encoding="utf-8"
    )
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert 'executable="fa_kws_node"' in launch_text
    assert "LaunchConfiguration(\"executable\")" not in launch_text
    assert "fa_kws_node_fallback" not in launch_text
    assert "fa_kws_stub" not in launch_text
    assert "add_executable(fa_kws_node" in cmake_text
    assert "install(TARGETS fa_kws_node fa_kws_wav_tool" in cmake_text
