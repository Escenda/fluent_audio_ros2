from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_tts_launch_requires_explicit_node_name_and_config_file() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_tts.launch.py").read_text(
        encoding="utf-8"
    )

    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert 'LaunchConfiguration("config_file")' in launch_text
