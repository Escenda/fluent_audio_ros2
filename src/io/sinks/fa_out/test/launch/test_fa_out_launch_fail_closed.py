import shutil
import subprocess

import pytest


def test_default_launch_fails_closed_without_explicit_sink_binding() -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    result = subprocess.run(
        [ros2, "launch", "fa_out", "fa_out.launch.py"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )

    assert "process has died" in result.stdout
    assert "exit code 1" in result.stdout
    assert "audio.device_id is required for backend.name=alsa_playback" in result.stdout
