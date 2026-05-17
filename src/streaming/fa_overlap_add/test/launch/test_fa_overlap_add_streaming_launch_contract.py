from pathlib import Path


def package_root() -> Path:
    return Path(__file__).parents[2]


def test_launch_exposes_only_node_name_and_config_file_arguments() -> None:
    package_name = package_root().name
    launch_text = (package_root() / "launch" / f"{package_name}.launch.py").read_text(encoding="utf-8")

    assert "DeclareLaunchArgument" in launch_text
    assert '"node_name"' in launch_text
    assert '"config_file"' in launch_text
    assert f'FindPackageShare("{package_name}")' in launch_text
    assert f'package="{package_name}"' in launch_text
    assert f'executable="{package_name}_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "backend.name" not in launch_text
