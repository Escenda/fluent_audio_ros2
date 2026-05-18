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


def test_default_config_requires_explicit_kws_backend_inputs() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_kws"]["ros__parameters"]

    assert params["backend.name"] == "sherpa_onnx_kws"
    assert params["backend.execution_provider"] == ""
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
    assert "dump_audio.enable" not in params
    assert "dump_audio.path" not in params


def test_launch_does_not_embed_backend_or_model_fallback() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_kws.launch.py").read_text(
        encoding="utf-8"
    )

    assert "sherpa_onnx_kws" not in launch_text
    assert "model.encoder" not in launch_text
    assert "backend.execution_provider" not in launch_text
    assert "cpu" not in launch_text
