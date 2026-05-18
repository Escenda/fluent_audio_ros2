import shutil
import subprocess
from pathlib import Path

import pytest


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
