import importlib
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

config_schema = importlib.import_module("fluent_audio_system.config_schema")

SRC_ROOT = PACKAGE_ROOT.parents[1]
BASE_PACKAGES = frozenset(("fa_interfaces", "fluent_audio_system"))
REQUIRED_PACKAGE_DOCS = (
    "README.md",
    "docs/仕様書.md",
    "docs/アルゴリズム詳細説明書.md",
    "docs/テスト設計.md",
)
REQUIRED_TEST_DIRS = (
    "test/unit",
    "test/integration",
    "test/launch",
    "test/fixtures",
)
AI_TEST_TRACE_PREFIXES = {
    "fa_asr": "FA-ASR",
    "fa_audio_embedding": "FA-AUDIO-EMBEDDING",
    "fa_kws": "FA-KWS",
    "fa_turn_detector": "FA-TD",
    "fa_vad": "FA-VAD",
}


def _buildable_package_dirs() -> list[Path]:
    return sorted({package_xml.parent for package_xml in SRC_ROOT.rglob("package.xml")})


def _relative_package_path(package_dir: Path) -> str:
    return str(package_dir.relative_to(SRC_ROOT))


def _path_category(package_dir: Path) -> str | None:
    relative_parts = package_dir.relative_to(SRC_ROOT).parts
    if not relative_parts:
        return None
    if relative_parts[0] == "processing" and len(relative_parts) >= 2:
        return relative_parts[1]
    if relative_parts[0] in {"ai", "streaming", "apps"}:
        return relative_parts[0]
    if relative_parts[0] == "io":
        return "io"
    return None


def test_package_category_map_covers_every_buildable_node_package() -> None:
    buildable_package_names = {
        package_dir.name
        for package_dir in _buildable_package_dirs()
        if package_dir.name not in BASE_PACKAGES
    }
    mapped_package_names = set(config_schema._PACKAGE_CATEGORIES)

    assert buildable_package_names - mapped_package_names == set()
    assert mapped_package_names - buildable_package_names == set()


def test_package_category_map_matches_repository_layout() -> None:
    package_categories: dict[str, frozenset[str]] = config_schema._PACKAGE_CATEGORIES

    for package_dir in _buildable_package_dirs():
        package_name = package_dir.name
        if package_name in BASE_PACKAGES:
            continue
        expected_category = _path_category(package_dir)
        if expected_category is None:
            continue

        assert expected_category in package_categories[package_name]


def test_buildable_packages_have_standard_documentation_layout() -> None:
    missing_paths: list[str] = []

    for package_dir in _buildable_package_dirs():
        for required_doc in REQUIRED_PACKAGE_DOCS:
            if not (package_dir / required_doc).is_file():
                missing_paths.append(f"{_relative_package_path(package_dir)}/{required_doc}")

        backend_docs_dir = package_dir / "docs" / "backends"
        if not backend_docs_dir.is_dir():
            missing_paths.append(f"{_relative_package_path(package_dir)}/docs/backends")
            continue
        if not any(backend_doc.is_file() for backend_doc in backend_docs_dir.glob("*.md")):
            missing_paths.append(f"{_relative_package_path(package_dir)}/docs/backends/*.md")

    assert missing_paths == []


def test_buildable_packages_have_standard_test_layout() -> None:
    missing_paths: list[str] = []

    for package_dir in _buildable_package_dirs():
        for required_test_dir in REQUIRED_TEST_DIRS:
            if not (package_dir / required_test_dir).is_dir():
                missing_paths.append(f"{_relative_package_path(package_dir)}/{required_test_dir}")

    assert missing_paths == []


def test_buildable_ai_packages_have_spec_to_test_traceability() -> None:
    missing_traceability: list[str] = []

    for package_name, trace_prefix in AI_TEST_TRACE_PREFIXES.items():
        package_dir = SRC_ROOT / "ai" / package_name
        test_design_path = package_dir / "docs" / "テスト設計.md"
        mapped_lines = [
            line
            for line in test_design_path.read_text(encoding="utf-8").splitlines()
            if f"`{trace_prefix}-TC-" in line
            and "->" in line
            and f"`{trace_prefix}-SPEC-" in line
        ]
        if not mapped_lines:
            missing_traceability.append(_relative_package_path(package_dir))

    assert missing_traceability == []


def test_ai_and_streaming_packages_stay_out_of_processing_analysis() -> None:
    processing_analysis_packages = {
        package_dir.name
        for package_dir in (SRC_ROOT / "processing" / "analysis").iterdir()
        if package_dir.is_dir()
    }

    assert processing_analysis_packages & set(config_schema._AI_PACKAGE_NAMES) == set()
    assert processing_analysis_packages & set(config_schema._STREAMING_PACKAGE_NAMES) == set()


def test_ai_and_streaming_packages_have_top_level_categories() -> None:
    for package_name in config_schema._AI_PACKAGE_NAMES:
        assert (SRC_ROOT / "ai" / package_name).is_dir()
        assert config_schema._PACKAGE_CATEGORIES[package_name] == frozenset(("ai",))

    for package_name in config_schema._STREAMING_PACKAGE_NAMES:
        assert (SRC_ROOT / "streaming" / package_name).is_dir()
        assert config_schema._PACKAGE_CATEGORIES[package_name] == frozenset(("streaming",))


def test_ai_roadmap_placeholders_are_not_launchable_system_packages() -> None:
    for package_name in ("fa_sed", "fa_speaker"):
        assert (SRC_ROOT / "ai" / package_name).is_dir()
        assert not (SRC_ROOT / "ai" / package_name / "package.xml").exists()
        assert package_name not in config_schema._PACKAGE_CATEGORIES
        assert package_name not in config_schema._AI_PACKAGE_NAMES
