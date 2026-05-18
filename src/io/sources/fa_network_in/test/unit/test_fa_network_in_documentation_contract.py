from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_network_in_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "network_pcm_receiver.md").is_file()


def test_fa_network_in_is_declared_as_ros_package_after_contract_completion() -> None:
    assert (PACKAGE_ROOT / "package.xml").is_file()
    assert (PACKAGE_ROOT / "CMakeLists.txt").is_file()
    assert (PACKAGE_ROOT / "launch" / "fa_network_in.launch.py").is_file()
    assert (PACKAGE_ROOT / "config" / "default.yaml").is_file()


def test_fa_network_in_docs_keep_streaming_stability_out_of_source_adapter() -> None:
    specification = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )

    assert "jitter buffer / PLC / clock drift correction" in specification
    assert "codec decode" in specification
    assert "transport instability はこの node で補正しない" in algorithm
