import shutil
import subprocess

import pytest


def test_default_launch_fails_closed_without_explicit_source_binding() -> None:
    ros2 = shutil.which("ros2")
    if ros2 is None:
        pytest.skip("ros2 executable is required for launch fail-closed verification")

    result = subprocess.run(
        [ros2, "launch", "fa_in", "fa_in.launch.py"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=8,
    )

    assert "process has died" in result.stdout
    assert "exit code 1" in result.stdout
    assert (
        "audio.device_selector.identifier is required when audio.device_selector.mode=name"
        in result.stdout
    )
