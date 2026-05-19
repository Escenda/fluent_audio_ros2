from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_launch_uses_explicit_node_name_and_config_file_arguments() -> None:
    package_name = PACKAGE_ROOT.name
    launch_text = (
        PACKAGE_ROOT / "launch" / f"{package_name}.launch.py"
    ).read_text(encoding="utf-8")

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'LaunchConfiguration("node_name")' in launch_text
    assert 'LaunchConfiguration("config_file")' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert f'package="{package_name}"' in launch_text
    if package_name == "fa_stream":
        expected_executable = "fa_stream_node.py"
    else:
        expected_executable = f"{package_name}_node"
    assert f'executable="{expected_executable}"' in launch_text
    assert "parameters=[config_file]" in launch_text
