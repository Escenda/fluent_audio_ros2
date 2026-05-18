from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def read_package_file(relative_path: str) -> str:
    return (PACKAGE_ROOT / relative_path).read_text(encoding="utf-8")


def test_fa_dialogue_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (
        PACKAGE_ROOT / "docs" / "backends" / "external_dialogue_service.md"
    ).is_file()


def test_fa_dialogue_is_not_declared_as_ros_package_before_contract_completion() -> None:
    assert not (PACKAGE_ROOT / "package.xml").exists()
    assert not (PACKAGE_ROOT / "CMakeLists.txt").exists()


def test_fa_dialogue_backend_and_safety_boundaries_are_explicit() -> None:
    spec = read_package_file("docs/仕様書.md")
    backend = read_package_file("docs/backends/external_dialogue_service.md")
    algorithm = read_package_file("docs/アルゴリズム詳細説明書.md")

    assert "未設定の backend を default model や canned response へ置き換えない" in spec
    assert "robot command の最終 accept / reject は `fa_safety_policy` の責務" in spec
    assert "backend failure を canned response で隠す" in algorithm
    assert "ROS2 topic/message dependency inside backend" in backend
    assert "direct robot actuation" in backend
