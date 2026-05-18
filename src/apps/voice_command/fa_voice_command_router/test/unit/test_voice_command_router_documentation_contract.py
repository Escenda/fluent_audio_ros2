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


def test_voice_command_router_runtime_backend_boundary_is_explicit() -> None:
    spec = read_package_file("docs/仕様書.md")
    backend = read_package_file("docs/backends/no_runtime_backend.md")
    algorithm = read_package_file("docs/アルゴリズム詳細説明書.md")

    assert "TTS 合成自体は `fa_tts` の責務" in spec
    assert "`tts_service` is a ROS service name, not a backend selector" in backend
    assert "No TTS engine is imported or selected here" in backend
    assert "音声 frame は扱わない" in algorithm
