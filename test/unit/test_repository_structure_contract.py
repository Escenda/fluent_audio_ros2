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
)

ANALYSIS_PACKAGE_NAMES = (
    "fa_log_mel",
)


AI_PACKAGE_NAMES = (
    "fa_asr",
    "fa_kws",
    "fa_turn_detector",
    "fa_vad",
)


AI_PLACEHOLDER_NAMES = (
    "fa_audio_embedding",
    "fa_sed",
    "fa_speaker",
)


STREAMING_PACKAGE_NAMES = (
    "fa_chunk_overlap",
    "fa_clock_drift",
    "fa_frame_buffer",
    "fa_jitter_buffer",
    "fa_latency_compensation",
    "fa_overlap_add",
    "fa_packet_loss_concealment",
    "fa_time_alignment",
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


def _ai_package_roots() -> list[Path]:
    ai_root = SRC_ROOT / "ai"
    return sorted(path.parent for path in ai_root.rglob("package.xml"))


def _streaming_package_roots() -> list[Path]:
    streaming_root = SRC_ROOT / "streaming"
    return sorted(path.parent for path in streaming_root.rglob("package.xml"))


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


def test_analysis_category_contains_only_non_ai_feature_packages() -> None:
    invalid: list[str] = []
    analysis_root = SRC_ROOT / "processing" / "analysis"
    package_names: list[str] = []

    for package_root in sorted(path.parent for path in analysis_root.rglob("package.xml")):
        package_names.append(package_root.name)
        if package_root.name not in ANALYSIS_PACKAGE_NAMES:
            invalid.append(str(package_root.relative_to(REPO_ROOT)))

    assert invalid == []
    assert tuple(package_names) == ANALYSIS_PACKAGE_NAMES


def test_ai_ros_packages_live_under_src_ai() -> None:
    missing: list[str] = []

    if not (SRC_ROOT / "ai" / "README.md").is_file():
        missing.append("src/ai/README.md")

    for package_name in AI_PACKAGE_NAMES:
        package_path = SRC_ROOT / "ai" / package_name
        if not (package_path / "package.xml").is_file():
            missing.append(f"src/ai/{package_name}/package.xml")

    for placeholder_name in AI_PLACEHOLDER_NAMES:
        placeholder_path = SRC_ROOT / "ai" / placeholder_name
        if not placeholder_path.is_dir():
            missing.append(f"src/ai/{placeholder_name}/")
        if (placeholder_path / "package.xml").exists():
            missing.append(f"src/ai/{placeholder_name}/package.xml")

    for package_root in _ai_package_roots():
        if package_root.name not in AI_PACKAGE_NAMES:
            missing.append(str(package_root.relative_to(REPO_ROOT)))

    assert missing == []


def test_streaming_ros_packages_live_under_src_streaming() -> None:
    missing: list[str] = []

    if not (SRC_ROOT / "streaming" / "README.md").is_file():
        missing.append("src/streaming/README.md")

    for package_name in STREAMING_PACKAGE_NAMES:
        package_path = SRC_ROOT / "streaming" / package_name
        if not (package_path / "package.xml").is_file():
            missing.append(f"src/streaming/{package_name}/package.xml")

    for package_root in _streaming_package_roots():
        if package_root.name not in STREAMING_PACKAGE_NAMES:
            missing.append(str(package_root.relative_to(REPO_ROOT)))

    assert missing == []


def test_streaming_docs_do_not_describe_packages_as_processing_nodes() -> None:
    streaming_root = SRC_ROOT / "streaming"
    checked_files = [
        path
        for path in streaming_root.rglob("*")
        if path.is_file()
        and path.suffix in (".md", ".hpp", ".py")
        and "__pycache__" not in path.parts
    ]
    violations: list[str] = []

    forbidden_phrases = (
        "processing node",
        "processing package",
        "processing layout",
        "processing contract",
        "processing_node",
        "required_processing_layout",
        "standard_processing_layout",
        "processing responsibilities",
        "processing_responsibilities",
    )
    for path in sorted(checked_files):
        source = path.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            if phrase in source:
                violations.append(f"{path.relative_to(REPO_ROOT)} contains {phrase}")

    assert violations == []


def test_network_stream_sink_utility_is_not_transport_streaming() -> None:
    package_path = SRC_ROOT / "io" / "utilities" / "fa_stream"
    streaming_path = SRC_ROOT / "streaming" / "fa_stream"
    docs = (package_path / "docs" / "仕様書.md").read_text(encoding="utf-8")

    assert (package_path / "package.xml").is_file()
    assert not streaming_path.exists()
    assert "src/streaming" in docs
    assert "リアルタイム伝送安定化" in docs


def test_processing_does_not_contain_ai_or_streaming_packages() -> None:
    forbidden_paths = [
        "src/processing/streaming",
        *(f"src/processing/analysis/{name}" for name in AI_PACKAGE_NAMES),
        *(f"src/processing/analysis/{name}" for name in AI_PLACEHOLDER_NAMES),
    ]
    present = [path for path in forbidden_paths if (REPO_ROOT / path).exists()]

    assert present == []


def test_generation_category_is_audio_data_plane_not_dialogue_ai() -> None:
    required_docs = [
        REPO_ROOT / "docs" / "仕様書.md",
        REPO_ROOT / "docs" / "アルゴリズム詳細説明書.md",
        SRC_ROOT / "processing" / "generation" / "README.md",
        SRC_ROOT / "processing" / "generation" / "fa_tts" / "docs" / "仕様書.md",
    ]
    missing_terms: list[str] = []

    for doc_path in required_docs:
        source = doc_path.read_text(encoding="utf-8")
        for term in (
            "data-plane",
            "dialogue policy",
            "LLM",
            "VLM",
            "src/ai",
            "src/apps",
        ):
            if term not in source:
                missing_terms.append(f"{doc_path.relative_to(REPO_ROOT)} lacks {term}")

    assert missing_terms == []


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
