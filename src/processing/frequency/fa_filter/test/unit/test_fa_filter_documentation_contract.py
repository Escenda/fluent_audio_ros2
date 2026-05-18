from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_filter_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "explicit_filter_pipeline.md").is_file()


def test_fa_filter_is_not_declared_as_ros_package_before_contract_completion() -> None:
    assert not (PACKAGE_ROOT / "package.xml").exists()
    assert not (PACKAGE_ROOT / "CMakeLists.txt").exists()


def test_fa_filter_separates_stage_topics_from_stream_identity() -> None:
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_doc = (PACKAGE_ROOT / "docs" / "backends" / "explicit_filter_pipeline.md").read_text(
        encoding="utf-8"
    )

    assert "`stage.input_topic`" in spec
    assert "`stage.output_topic`" in spec
    assert "`stage.input_stream_id`" in spec
    assert "`stage.output.stream_id`" in spec
    assert "ROS 搬送路の identity" in spec
    assert "`AudioFrame.stream_id` ではない" in spec
    assert "stage topic と stream identity の衝突" in spec
    assert "stage input stream id" in backend_doc
    assert "stage output stream id" in backend_doc
    assert "ROS topic と `AudioFrame.stream_id` の兼用" in backend_doc
