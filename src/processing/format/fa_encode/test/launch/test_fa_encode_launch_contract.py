from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_file_runs_fa_encode_node_with_params_file() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_encode.launch.py").read_text(
        encoding="utf-8"
    )

    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'executable="fa_encode_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
