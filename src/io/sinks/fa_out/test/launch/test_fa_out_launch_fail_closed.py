import shutil
import subprocess
from pathlib import Path

import pytest
import yaml


def _run_fa_out_launch(*launch_args: str) -> str:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    result = subprocess.run(
        [ros2, "launch", "fa_out", "fa_out.launch.py", *launch_args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )
    return result.stdout


def _write_file_backend_config(path: Path, target: str, *, overwrite_enabled: bool) -> Path:
    config = {
        "fa_out": {
            "ros__parameters": {
                "backend.name": "pcm_file_writer",
                "file.path": target,
                "input_topic": "fa_out/file_input",
                "input_stream_id": "audio/file/output",
                "playback_done_topic": "fa_out/file_playback_done",
                "playback_control_service": "fa_out/file_playback_control",
                "audio.encoding": "PCM16LE",
                "audio.sample_rate": 16000,
                "audio.channels": 1,
                "audio.bit_depth": 16,
                "audio.chunk_duration_ms": 1,
                "queue.max_frames": 32,
                "audio.qos.depth": 10,
                "audio.qos.reliable": True,
                "lifecycle.qos.depth": 10,
                "lifecycle.qos.reliable": True,
                "overwrite.enabled": overwrite_enabled,
            }
        }
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_default_launch_fails_closed_without_explicit_sink_binding() -> None:
    default_config = Path(__file__).parents[2] / "config" / "default.yaml"
    output = _run_fa_out_launch(
        "node_name:=fa_out",
        f"config_file:={default_config}",
    )

    assert "process has died" in output
    assert "exit code 1" in output
    assert "audio.device_id is required for backend.name=alsa_playback" in output


def test_launch_fails_closed_when_params_file_is_missing(tmp_path: Path) -> None:
    missing_config = tmp_path / "missing.yaml"
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={missing_config}")

    assert f"Parameter file path is not a file: {missing_config}" in output
    assert "process has died" in output
    assert "exit code 1" in output
    assert "Statically typed parameter 'backend.name' must be initialized" in output


def test_file_backend_launch_fails_closed_when_file_path_is_missing(tmp_path: Path) -> None:
    config_path = _write_file_backend_config(
        tmp_path / "missing_file_path.yaml",
        "",
        overwrite_enabled=False,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "file.path is required for backend.name=pcm_file_writer" in output


def test_file_backend_launch_fails_closed_when_overwrite_disabled_target_exists(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.pcm"
    target.write_bytes(b"\x00\x00")
    config_path = _write_file_backend_config(
        tmp_path / "target_exists.yaml",
        str(target),
        overwrite_enabled=False,
    )
    output = _run_fa_out_launch("node_name:=fa_out", f"config_file:={config_path}")

    assert "process has died" in output
    assert "overwrite.enabled=false" in output
