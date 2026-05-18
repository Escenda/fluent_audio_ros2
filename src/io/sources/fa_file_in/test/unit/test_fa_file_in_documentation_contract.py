from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_file_in_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "pcm_file_reader.md").is_file()


def test_fa_file_in_is_declared_as_ros_package_after_contract_completion() -> None:
    assert (PACKAGE_ROOT / "package.xml").is_file()
    assert (PACKAGE_ROOT / "CMakeLists.txt").is_file()
    assert (PACKAGE_ROOT / "launch" / "fa_file_in.launch.py").is_file()
    assert (PACKAGE_ROOT / "config" / "default.yaml").is_file()


def test_fa_file_in_docs_keep_file_source_adapter_boundary() -> None:
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "codec decode" in spec
    assert "resample" in spec
    assert "gain" in spec
    assert "Those steps must be explicit processing nodes" in readme
