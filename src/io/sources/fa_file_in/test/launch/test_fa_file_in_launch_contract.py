from pathlib import Path
import shutil
import subprocess

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def _run_fa_file_in_launch(config_path: Path) -> subprocess.CompletedProcess[str]:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    return subprocess.run(
        [
            ros2,
            "launch",
            "fa_file_in",
            "fa_file_in.launch.py",
            "node_name:=fa_file_in",
            f"config_file:={config_path}",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )


def test_launch_uses_only_node_name_and_config_file_arguments() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_file_in.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'package="fa_file_in"' in launch_text
    assert 'executable="fa_file_in_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "file.path" not in launch_text
    assert "expected.sample_rate" not in launch_text
    assert "backend.name" not in launch_text


def test_default_launch_config_declares_raw_pcm_source_contract() -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_file_in"]["ros__parameters"]

    assert params["backend.name"] == "pcm_file_reader"
    assert params["file.path"] == ""
    assert params["output_topic"] == "audio/file_in"
    assert params["audio"]["source_id"] == "file_source"
    assert params["audio"]["stream_id"] == "audio/file_in"
    assert params["audio"]["frames_per_chunk"] == 160
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert params["expected"]["layout"] == "interleaved"


def test_launch_fails_closed_when_file_path_is_missing(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "missing_file_path.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_file_in_launch(config_path)

    assert "process has died" in result.stdout
    assert "file.path is required" in result.stdout


def test_launch_fails_closed_when_backend_is_unknown(tmp_path: Path) -> None:
    config = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    config["fa_file_in"]["ros__parameters"]["backend.name"] = "hidden_decoder"
    config["fa_file_in"]["ros__parameters"]["file.path"] = str(tmp_path / "fixture.pcm")
    (tmp_path / "fixture.pcm").write_bytes(b"\x00\x00")
    config_path = tmp_path / "unknown_backend.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = _run_fa_file_in_launch(config_path)

    assert "process has died" in result.stdout
    assert "unsupported fa_file_in backend.name: hidden_decoder" in result.stdout
