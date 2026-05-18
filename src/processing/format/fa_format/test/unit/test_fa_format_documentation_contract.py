from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_format_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "explicit_format_pipeline.md").is_file()


def test_fa_format_is_not_declared_as_ros_package_before_contract_completion() -> None:
    assert not (PACKAGE_ROOT / "package.xml").exists()
    assert not (PACKAGE_ROOT / "CMakeLists.txt").exists()


def test_fa_format_documents_current_fluent_audio_system_boundary() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_design = (PACKAGE_ROOT / "docs" / "テスト設計.md").read_text(encoding="utf-8")

    assert "fluent_audio_system" in readme
    assert "package: fa_format" in readme
    assert "leaf package" in spec
    assert "generic launch orchestration" in spec
    assert "format-stage adjacency validation" in algorithm
    assert "FA-FORMAT-SPEC-005" in test_design
