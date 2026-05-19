from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_audio_embedding_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "external_worker.md").is_file()


def test_fa_audio_embedding_is_declared_as_ros_package() -> None:
    assert (PACKAGE_ROOT / "package.xml").is_file()
    assert (PACKAGE_ROOT / "CMakeLists.txt").is_file()
    assert (PACKAGE_ROOT / "config" / "default.yaml").is_file()
    assert (PACKAGE_ROOT / "launch" / "fa_audio_embedding.launch.py").is_file()
    assert (PACKAGE_ROOT / "scripts" / "fa_audio_embedding_node").is_file()
