from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_wiener_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "stft_wiener_filter.md").is_file()


def test_fa_wiener_is_not_declared_as_ros_package_before_contract_completion() -> None:
    assert not (PACKAGE_ROOT / "package.xml").exists()
    assert not (PACKAGE_ROOT / "CMakeLists.txt").exists()


def test_fa_wiener_separates_topics_from_stream_identity() -> None:
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")

    assert "`input_topic`: ROS 搬送路" in spec
    assert "`input_stream_id`" in spec
    assert "`output_topic`: ROS 搬送路" in spec
    assert "`output.stream_id`" in spec
    assert "`stream_id` は `output.stream_id` に更新する" in spec
    assert "`stream_id` は `output_topic` に更新する" not in spec
