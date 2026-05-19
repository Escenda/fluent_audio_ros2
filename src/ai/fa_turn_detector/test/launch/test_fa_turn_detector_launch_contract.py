from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_explicit_config_file_without_model_override_contract() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_turn_detector.launch.py").read_text(
        encoding="utf-8"
    )

    assert "default_value" not in launch_text
    forbidden_find_package_share = "Find" + "PackageShare"
    forbidden_path_join_substitution = "PathJoin" + "Substitution"
    assert forbidden_find_package_share not in launch_text
    assert forbidden_path_join_substitution not in launch_text
    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'DeclareLaunchArgument(\n            "model_path"' not in launch_text
    assert 'LaunchConfiguration("model_path")' not in launch_text
    assert "backend.model_path" not in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "onnxruntime" not in launch_text


def test_default_config_requires_explicit_backend_and_external_worker_boundary() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_turn_detector"]["ros__parameters"]

    assert params["audio_topic"] == "audio/frame"
    assert params["expected_stream_id"] == "audio/raw/mic"
    assert params["audio_topic"] != params["expected_stream_id"]
    assert params["expected_source_id"] == ""
    assert params["backend.name"] == ""
    assert params["backend.model_path"] == ""
    assert params["backend.execution_provider"] == ""
    assert params["backend.command"] == ""
    assert params["backend.args"] == []
    assert params["backend.health_args"] == []
    assert params["backend.timeout_sec"] > 0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["backend.threshold"] == 0.5
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is False
    assert params["vad.qos.depth"] == 10
    assert params["vad.qos.reliable"] is False
    assert params["turn_context.qos.depth"] == 10
    assert params["turn_context.qos.reliable"] is True
    assert params["output.qos.depth"] == 10
    assert params["output.qos.reliable"] is True


def test_reference_worker_script_is_thin_backend_entrypoint() -> None:
    script_text = (PACKAGE_ROOT / "scripts" / "smart_turn_onnx_worker").read_text(
        encoding="utf-8"
    )
    worker_path = (
        PACKAGE_ROOT
        / "fa_turn_detector_py"
        / "backends"
        / "smart_turn_onnx_worker.py"
    )

    assert worker_path.is_file()
    assert (
        "from fa_turn_detector_py.backends.smart_turn_onnx_worker import main"
        in script_text
    )
    assert "onnxruntime" not in script_text


def test_onnxruntime_is_confined_to_worker_runtime_module() -> None:
    node_text = (
        PACKAGE_ROOT / "fa_turn_detector_py" / "turn_detector_node.py"
    ).read_text(encoding="utf-8")
    adapter_text = (
        PACKAGE_ROOT / "fa_turn_detector_py" / "backends" / "smart_turn_onnx.py"
    ).read_text(encoding="utf-8")
    worker_text = (
        PACKAGE_ROOT
        / "fa_turn_detector_py"
        / "backends"
        / "smart_turn_onnx_worker.py"
    ).read_text(encoding="utf-8")
    runtime_text = (
        PACKAGE_ROOT
        / "fa_turn_detector_py"
        / "backends"
        / "smart_turn_onnx_runtime.py"
    ).read_text(encoding="utf-8")

    assert "onnxruntime" not in node_text
    assert "onnxruntime" not in adapter_text
    assert "onnxruntime" not in worker_text
    assert "onnxruntime" in runtime_text
