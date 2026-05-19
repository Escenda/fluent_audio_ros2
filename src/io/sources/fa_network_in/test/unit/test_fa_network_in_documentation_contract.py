from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_network_in_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (
        PACKAGE_ROOT / "docs" / "backends" / "network_pcm_receiver_adapter.md"
    ).is_file()


def test_fa_network_in_is_not_declared_as_ros_package() -> None:
    assert not (PACKAGE_ROOT / "package.xml").exists()
    assert not (PACKAGE_ROOT / "CMakeLists.txt").exists()


def test_fa_network_in_documents_current_backend_boundary() -> None:
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend = (
        PACKAGE_ROOT / "docs" / "backends" / "network_pcm_receiver_adapter.md"
    ).read_text(encoding="utf-8")

    assert "package: fa_network_in" in readme
    assert "fa_in" in readme
    assert "backend.name: network_pcm_receiver" in spec
    assert "current implementation lives in `fa_in`" in backend


def test_fa_network_in_documents_streaming_boundary() -> None:
    combined = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (
            PACKAGE_ROOT / "docs" / "仕様書.md",
            PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md",
            PACKAGE_ROOT / "docs" / "backends" / "network_pcm_receiver_adapter.md",
        )
    )

    assert "jitter" in combined
    assert "packet loss concealment" in combined
    assert "clock drift" in combined
    assert "src/streaming" in combined
