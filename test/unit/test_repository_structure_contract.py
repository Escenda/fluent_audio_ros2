from pathlib import Path
from typing import TypeAlias

import yaml


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

IO_ROADMAP_PACKAGE_PATHS = (
    "io/sources/fa_in",
    "io/sources/fa_file_in",
    "io/sources/fa_network_in",
    "io/sinks/fa_out",
    "io/sinks/fa_file_out",
    "io/sinks/fa_network_out",
    "io/utilities/fa_record",
    "io/utilities/fa_stream",
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

PROCESSING_ROADMAP_PACKAGE_NAMES = (
    (
        "format",
        (
            "fa_resample",
            "fa_bit_depth",
            "fa_channel_convert",
            "fa_interleave",
            "fa_sample_format",
            "fa_encode",
            "fa_decode",
            "fa_format",
        ),
    ),
    (
        "dynamics",
        (
            "fa_gain",
            "fa_normalize",
            "fa_compressor",
            "fa_limiter",
            "fa_expander",
            "fa_noise_gate",
            "fa_agc",
        ),
    ),
    (
        "frequency",
        (
            "fa_eq",
            "fa_low_pass",
            "fa_high_pass",
            "fa_band_pass",
            "fa_notch",
            "fa_deesser",
            "fa_spectral_subtraction",
            "fa_wiener",
            "fa_filter",
        ),
    ),
    (
        "temporal",
        (
            "fa_trim",
            "fa_silence_removal",
            "fa_time_stretch",
            "fa_pitch_shift",
            "fa_delay",
            "fa_echo",
            "fa_reverb",
            "fa_crossfade",
            "fa_fade",
            "fa_window",
        ),
    ),
    (
        "correction",
        (
            "fa_denoise",
            "fa_aec_linear",
            "fa_aec_nn",
            "fa_dereverb",
            "fa_declip",
            "fa_debreath",
            "fa_declick",
            "fa_wind",
            "fa_hum",
            "fa_dc_offset_removal",
        ),
    ),
    (
        "spatial",
        (
            "fa_pan",
            "fa_stereo_widening",
            "fa_downmix",
            "fa_upmix",
            "fa_beamforming",
            "fa_source_separation",
            "fa_binaural",
            "fa_ambisonics",
        ),
    ),
    (
        "analysis",
        (
            "fa_onset",
            "fa_pitch",
            "fa_tempo",
            "fa_stft",
            "fa_log_mel",
            "fa_mfcc",
            "fa_cqt",
            "fa_loudness",
        ),
    ),
    (
        "generation",
        (
            "fa_tts",
            "fa_voice_conversion",
            "fa_speech_enhancement",
            "fa_speech_separation",
            "fa_speech_translation",
            "fa_music_source_separation",
            "fa_neural_codec",
            "fa_neural_vocoder",
            "fa_super_resolution",
        ),
    ),
    (
        "routing",
        (
            "fa_mix",
            "fa_bus_router",
            "fa_sidechain",
            "fa_ducking",
            "fa_monitor_mix",
            "fa_loopback",
            "fa_patchbay",
        ),
    ),
)

ANALYSIS_PACKAGE_NAMES = (
    "fa_log_mel",
    "fa_loudness",
    "fa_mfcc",
    "fa_onset",
    "fa_pitch",
    "fa_stft",
    "fa_tempo",
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

YamlScalar: TypeAlias = str | int | float | bool | None
YamlMapping: TypeAlias = dict[str, "YamlValue"]
YamlSequence: TypeAlias = list["YamlValue"]
YamlValue: TypeAlias = YamlScalar | YamlMapping | YamlSequence


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


def _collect_yaml_keys(value: YamlValue) -> list[str]:
    keys: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            keys.append(key)
            keys.extend(_collect_yaml_keys(child))
    elif isinstance(value, list):
        for child in value:
            keys.extend(_collect_yaml_keys(child))
    return keys


def test_all_ros_packages_use_standard_documented_layout() -> None:
    missing: list[str] = []

    for package_root in _package_roots():
        for relative_path in REQUIRED_PACKAGE_PATHS:
            expected = package_root / relative_path
            if not expected.exists():
                missing.append(f"{package_root.relative_to(REPO_ROOT)}/{relative_path}")

    assert missing == []


def test_io_taxonomy_exposes_design_source_sink_utility_directories() -> None:
    missing: list[str] = []

    for package_path in IO_ROADMAP_PACKAGE_PATHS:
        package_root = SRC_ROOT / package_path
        if not package_root.is_dir():
            missing.append(f"src/{package_path}/")
        if not (package_root / "README.md").is_file():
            missing.append(f"src/{package_path}/README.md")

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


def test_processing_taxonomy_exposes_all_design_roadmap_directories() -> None:
    missing: list[str] = []

    for category, package_names in PROCESSING_ROADMAP_PACKAGE_NAMES:
        for package_name in package_names:
            package_path = SRC_ROOT / "processing" / category / package_name
            if not package_path.is_dir():
                missing.append(f"src/processing/{category}/{package_name}/")
            if not (package_path / "README.md").is_file():
                missing.append(f"src/processing/{category}/{package_name}/README.md")

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


def test_streaming_packages_have_executable_integration_and_launch_contracts() -> None:
    missing: list[str] = []

    for package_root in _streaming_package_roots():
        for relative_path in ("test/integration", "test/launch"):
            test_dir = package_root / relative_path
            test_files = sorted(test_dir.glob("test_*.py"))
            if not test_files:
                missing.append(f"{test_dir.relative_to(REPO_ROOT)}/test_*.py")

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
    checked_files.append(REPO_ROOT / "docs" / "仕様書.md")
    violations: list[str] = []

    forbidden_phrases = (
        "Frame Processing",
        "Processing Pipeline",
        "processing node",
        "processing package",
        "processing layout",
        "processing contract",
        "processing_node",
        "required_processing_layout",
        "standard_processing_layout",
        "processing responsibilities",
        "processing_responsibilities",
        "処理手順",
        "処理対象",
        "後段処理",
        "リアルタイム伝送処理",
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


def test_voice_command_docs_do_not_collapse_ai_events_into_analysis() -> None:
    source = (SRC_ROOT / "apps" / "voice_command" / "README.md").read_text(
        encoding="utf-8"
    )
    normalized_source = " ".join(source.split())

    assert "audio-analysis" not in normalized_source
    assert "analysis events" not in normalized_source
    assert "AI events" in normalized_source
    assert "non-AI feature events" in normalized_source


def test_all_ros_packages_have_backend_documentation_file() -> None:
    missing: list[str] = []

    for package_root in _package_roots():
        backend_docs = sorted((package_root / "docs" / "backends").glob("*.md"))
        if not backend_docs:
            missing.append(
                f"{package_root.relative_to(REPO_ROOT)}/docs/backends/*.md"
            )

    assert missing == []


def test_config_files_do_not_use_legacy_backend_mapping_key() -> None:
    violations: list[str] = []

    for config_path in sorted(SRC_ROOT.rglob("config/*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        keys = _collect_yaml_keys(config)
        if "backend" in keys:
            violations.append(str(config_path.relative_to(REPO_ROOT)))

    assert violations == []


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
