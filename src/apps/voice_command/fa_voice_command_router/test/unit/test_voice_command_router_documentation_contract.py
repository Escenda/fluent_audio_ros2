from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def read_package_file(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_voice_command_router_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "no_runtime_backend.md").is_file()


def test_voice_command_router_has_standard_test_directories() -> None:
    assert (PACKAGE_ROOT / "test" / "unit").is_dir()
    assert (PACKAGE_ROOT / "test" / "integration").is_dir()
    assert (PACKAGE_ROOT / "test" / "launch").is_dir()
    assert (PACKAGE_ROOT / "test" / "fixtures").is_dir()


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    launch_text = read_package_file("launch/fa_voice_command_router.launch.py")

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert "parameters=[config_file]" in launch_text
