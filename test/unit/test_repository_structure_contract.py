from pathlib import Path


REPO_ROOT = Path(__file__).parents[2]
SRC_ROOT = REPO_ROOT / "src"


REQUIRED_PACKAGE_PATHS = (
    "README.md",
    "docs/仕様書.md",
    "docs/アルゴリズム詳細説明書.md",
    "docs/テスト設計.md",
    "docs/backends",
    "test/unit",
    "test/integration",
    "test/launch",
    "test/fixtures",
)


BACKEND_CODE_SUFFIXES = (".cpp", ".hpp", ".py")
FORBIDDEN_BACKEND_TOKENS = (
    "#include <rclcpp",
    '#include "rclcpp',
    "import rclpy",
    "from rclpy",
    "#include <fa_interfaces",
    '#include "fa_interfaces',
    "from fa_interfaces",
    "import fa_interfaces",
)


def _package_roots() -> list[Path]:
    return sorted(path.parent for path in SRC_ROOT.rglob("package.xml"))


def _backend_code_files() -> list[Path]:
    backend_dirs = [
        path
        for path in SRC_ROOT.rglob("backends")
        if path.is_dir() and "docs" not in path.parts
    ]
    files: list[Path] = []
    for backend_dir in backend_dirs:
        files.extend(
            path
            for path in backend_dir.rglob("*")
            if path.is_file() and path.suffix in BACKEND_CODE_SUFFIXES
        )
    return sorted(files)


def test_all_ros_packages_use_standard_documented_layout() -> None:
    missing: list[str] = []

    for package_root in _package_roots():
        for relative_path in REQUIRED_PACKAGE_PATHS:
            expected = package_root / relative_path
            if not expected.exists():
                missing.append(f"{package_root.relative_to(REPO_ROOT)}/{relative_path}")

    assert missing == []


def test_legacy_fa_capture_and_fa_output_paths_are_not_present() -> None:
    legacy_paths = [
        str(path.relative_to(REPO_ROOT))
        for path in SRC_ROOT.rglob("*")
        if "fa_capture" in path.parts or "fa_output" in path.parts
    ]

    assert legacy_paths == []


def test_runtime_backends_do_not_import_ros2_or_audio_messages() -> None:
    violations: list[str] = []

    for code_file in _backend_code_files():
        source = code_file.read_text(encoding="utf-8")
        for forbidden_token in FORBIDDEN_BACKEND_TOKENS:
            if forbidden_token in source:
                violations.append(
                    f"{code_file.relative_to(REPO_ROOT)} contains {forbidden_token}"
                )

    assert violations == []
