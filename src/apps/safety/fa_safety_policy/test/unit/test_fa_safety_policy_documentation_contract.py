from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def read_package_file(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_fa_safety_policy_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "no_runtime_backend.md").is_file()


def test_fa_safety_policy_documents_absence_of_runtime_backend() -> None:
    spec = read_package_file("docs/仕様書.md")
    readme = read_package_file("README.md")
    backend = read_package_file("docs/backends/no_runtime_backend.md")

    assert "backend が無いこと自体を `docs/backends/no_runtime_backend.md` で明示する" in spec
    assert "具体 backend contract を追加します" in readme
    assert "No external runtime backend is selected by this package" in backend
    assert "No model backend fallback is allowed" in backend


def test_fa_safety_policy_is_not_declared_as_ros_package_before_contract_completion() -> None:
    assert not (PACKAGE_ROOT / "package.xml").exists()
    assert not (PACKAGE_ROOT / "CMakeLists.txt").exists()


def test_fa_safety_policy_fails_closed_for_missing_or_stale_state() -> None:
    spec = read_package_file("docs/仕様書.md")
    algorithm = read_package_file("docs/アルゴリズム詳細説明書.md")

    assert "unknown action、missing state、stale state は default allow にしない" in spec
    assert "空 decision、default allow、警告のみで継続はしない" in spec
    assert "stale robot state を current state として扱う" in algorithm
    assert "unknown action を low-risk action と推測する" in algorithm
