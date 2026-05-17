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


PROCESSING_CATEGORIES = (
    "format",
    "dynamics",
    "frequency",
    "temporal",
    "correction",
    "spatial",
    "analysis",
    "generation",
    "routing",
    "streaming",
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


FORBIDDEN_PYTHON_TYPE_ESCAPE_TOKENS = (
    "from typing import Any",
    "typing.Any",
    "dict[str, Any]",
    "Dict[str, Any]",
    ": Any",
    "-> Any",
    ": object",
    "-> object",
    "list[object]",
    "dict[str, object]",
    "tuple[object",
    "# type: ignore",
)


def _package_roots() -> list[Path]:
    return sorted(path.parent for path in SRC_ROOT.rglob("package.xml"))


def _processing_package_roots() -> list[Path]:
    processing_root = SRC_ROOT / "processing"
    return sorted(path.parent for path in processing_root.rglob("package.xml"))


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


def _production_python_files() -> list[Path]:
    return sorted(
        path
        for path in SRC_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts and "test" not in path.parts
    )


def test_all_ros_packages_use_standard_documented_layout() -> None:
    missing: list[str] = []

    for package_root in _package_roots():
        for relative_path in REQUIRED_PACKAGE_PATHS:
            expected = package_root / relative_path
            if not expected.exists():
                missing.append(f"{package_root.relative_to(REPO_ROOT)}/{relative_path}")

    assert missing == []


def test_processing_taxonomy_has_all_design_categories() -> None:
    missing: list[str] = []

    for category in PROCESSING_CATEGORIES:
        category_path = SRC_ROOT / "processing" / category
        if not category_path.is_dir():
            missing.append(f"src/processing/{category}/")
        if not (category_path / "README.md").is_file():
            missing.append(f"src/processing/{category}/README.md")

    assert missing == []


def test_processing_ros_packages_live_under_taxonomy_categories() -> None:
    invalid: list[str] = []
    allowed_parents = {
        SRC_ROOT / "processing" / category for category in PROCESSING_CATEGORIES
    }

    for package_root in _processing_package_roots():
        if package_root.parent not in allowed_parents:
            invalid.append(str(package_root.relative_to(REPO_ROOT)))

    assert invalid == []


def test_all_ros_packages_have_backend_documentation_file() -> None:
    missing: list[str] = []

    for package_root in _package_roots():
        backend_docs = sorted((package_root / "docs" / "backends").glob("*.md"))
        if not backend_docs:
            missing.append(
                f"{package_root.relative_to(REPO_ROOT)}/docs/backends/*.md"
            )

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


def test_production_python_does_not_use_ambiguous_type_escapes() -> None:
    violations: list[str] = []

    for code_file in _production_python_files():
        source = code_file.read_text(encoding="utf-8")
        for forbidden_token in FORBIDDEN_PYTHON_TYPE_ESCAPE_TOKENS:
            if forbidden_token in source:
                violations.append(
                    f"{code_file.relative_to(REPO_ROOT)} contains {forbidden_token}"
                )

    assert violations == []
