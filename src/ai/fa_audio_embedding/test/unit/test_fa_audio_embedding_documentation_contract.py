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


def test_fa_audio_embedding_readme_uses_current_package_contract() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")

    assert "ROS 2 package" in readme
    assert "AudioEmbeddingFrame" in readme
    assert "現時点では ROS 2 package ではありません" not in readme
    assert "package 化前" not in readme


def test_fa_audio_embedding_docs_forbid_zero_or_stale_fallbacks() -> None:
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    backend_doc = (
        PACKAGE_ROOT / "docs" / "backends" / "external_worker.md"
    ).read_text(encoding="utf-8")

    assert "fallback" in spec
    assert "zero vector" in algorithm
    assert "stale embedding" in algorithm
    assert "zero vector fallback" in backend_doc
    assert "stale embedding reuse" in backend_doc


def test_fa_audio_embedding_backend_docs_forbid_ros_dependency() -> None:
    backend_doc = (
        PACKAGE_ROOT / "docs" / "backends" / "external_worker.md"
    ).read_text(encoding="utf-8")

    assert "ROS2 topic/message dependency inside backend" in backend_doc
    assert "backend.command" in backend_doc
    assert "backend.name: external_worker" in backend_doc
    assert "backend.payload_encoding" in backend_doc
    assert "`payload_encoding`" not in backend_doc
    assert "backend.endpoint" not in backend_doc
