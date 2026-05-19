from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]
LAUNCH_TIMEOUT_CODE = 124


def _run_fa_denoise_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    command = [
        ros2,
        "launch",
        "fa_denoise",
        "fa_denoise.launch.py",
        "node_name:=fa_denoise",
        f"config_file:={config_path}",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        stdout, _ = process.communicate(timeout=8)
        return subprocess.CompletedProcess(command, process.returncode, stdout, None)
    except subprocess.TimeoutExpired:
        process.terminate()
        stdout, _ = process.communicate(timeout=2)
        return subprocess.CompletedProcess(command, LAUNCH_TIMEOUT_CODE, stdout, None)
def test_default_launch_config_keeps_dtln_model_paths_explicit() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_denoise"]["ros__parameters"]

    assert params["backend.name"] == "dtln_onnx"
    assert params["input_topic"] == "fa_denoise/input"
    assert params["output_topic"] == "fa_denoise/output"
    assert params["input_stream_id"] == "audio/resample16k/mic"
    assert params["output"]["stream_id"] == "audio/denoised/mic"
    assert params["input_stream_id"] != params["input_topic"]
    assert params["output"]["stream_id"] != params["output_topic"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["dtln"]["model_1_path"] == ""
    assert params["dtln"]["model_2_path"] == ""
    assert params["dtln"]["block_len"] == 512
    assert params["dtln"]["block_shift"] == 128
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["output"]["encoding"] == "PCM16LE"
    assert params["output"]["bit_depth"] == 16
    assert "resample" not in params
    assert "gain" not in params
    assert "limiter" not in params


def test_launch_fails_closed_for_default_dtln_without_runtime_or_models(
    tmp_path: Path,
) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "default_dtln.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert (
        "fa_denoise was built without ONNX Runtime support" in result.stdout
        or "dtln.model_1_path is required for dtln_onnx backend" in result.stdout
    )


def test_launch_fails_closed_when_backend_is_unknown(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_denoise"]["ros__parameters"]["backend.name"] = "unknown"
    config_path = tmp_path / "unknown_backend.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "backend.name must be passthrough or dtln_onnx" in result.stdout


def test_launch_fails_closed_when_raw_topics_match(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_denoise"]["ros__parameters"]
    params["backend.name"] = "passthrough"
    params["output_topic"] = params["input_topic"]
    config_path = tmp_path / "same_raw_topics.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "input_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_resolved_topics_match(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_denoise"]["ros__parameters"]
    params["backend.name"] = "passthrough"
    params["input_topic"] = "audio/same"
    params["output_topic"] = "/audio/same"
    config_path = tmp_path / "same_resolved_topics.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "resolved input_topic and output_topic must be distinct" in result.stdout


def test_launch_fails_closed_when_stream_identity_collides_with_topic(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_denoise"]["ros__parameters"]
    params["backend.name"] = "passthrough"
    params["input_stream_id"] = params["input_topic"]
    config_path = tmp_path / "stream_topic_collision.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "input_stream_id must be distinct from ROS topics" in result.stdout


def test_launch_fails_closed_when_input_output_stream_identity_matches(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_denoise"]["ros__parameters"]
    params["backend.name"] = "passthrough"
    params["output"]["stream_id"] = params["input_stream_id"]
    config_path = tmp_path / "same_stream_ids.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode != LAUNCH_TIMEOUT_CODE
    assert "process has died" in result.stdout
    assert "input_stream_id and output.stream_id must be distinct" in result.stdout


def test_launch_accepts_explicit_passthrough_config(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_denoise"]["ros__parameters"]["backend.name"] = "passthrough"
    config_path = tmp_path / "passthrough.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_denoise_launch(config_path)

    assert result.returncode == LAUNCH_TIMEOUT_CODE
    assert "backend.name must be passthrough or dtln_onnx" not in result.stdout
    assert "fa_denoise was built without ONNX Runtime support" not in result.stdout
    assert "Starting FA Denoise node" in result.stdout
    assert "Exception:" not in result.stdout
