import importlib
from pathlib import Path
import sys

PACKAGE_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

config_schema = importlib.import_module("fluent_audio_system.config_schema")

SRC_ROOT = PACKAGE_ROOT.parents[1]
BASE_PACKAGES = frozenset(("fa_interfaces", "fluent_audio_system"))


def _documented_package_dirs() -> list[Path]:
    package_dirs: list[Path] = []
    for candidate in SRC_ROOT.rglob("fa_*"):
        if not candidate.is_dir():
            continue
        if not (candidate / "README.md").is_file():
            continue
        if not (candidate / "docs" / "仕様書.md").is_file():
            continue
        if not (candidate / "package.xml").is_file():
            continue
        package_dirs.append(candidate)
    package_dirs.append(PACKAGE_ROOT)
    return sorted(package_dirs)


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


def test_package_category_map_covers_every_documented_node_package() -> None:
    documented_package_names = {
        package_dir.name
        for package_dir in _documented_package_dirs()
        if package_dir.name not in BASE_PACKAGES
    }
    mapped_package_names = set(config_schema._PACKAGE_CATEGORIES)

    assert documented_package_names - mapped_package_names == set()
    assert mapped_package_names - documented_package_names == set()


def test_package_category_map_matches_repository_layout() -> None:
    package_categories: dict[str, frozenset[str]] = config_schema._PACKAGE_CATEGORIES

    for package_dir in _documented_package_dirs():
        package_name = package_dir.name
        if package_name in BASE_PACKAGES:
            continue
        expected_category = _path_category(package_dir)
        if expected_category is None:
            continue

        assert expected_category in package_categories[package_name]


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
