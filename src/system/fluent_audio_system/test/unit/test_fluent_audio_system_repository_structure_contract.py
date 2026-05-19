import importlib
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET

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
IO_TEST_TRACE_PREFIXES = {
    "io/sinks/fa_out": "FA-OUT",
    "io/sources/fa_in": "FA-IN",
    "io/utilities/fa_record": "FA-RECORD",
    "io/utilities/fa_stream": "FA-STREAM",
}
INTERFACE_TEST_TRACE_PREFIXES = {
    "interfaces/fa_interfaces": "FA-INTERFACES",
}
APP_TEST_TRACE_PREFIXES = {
    "apps/voice_command/fa_voice_command_router": "FA-VOICE-COMMAND-ROUTER",
}
STREAMING_TEST_TRACE_PREFIXES = {
    "streaming/fa_chunk_overlap": "FA-CHUNK-OVERLAP",
    "streaming/fa_clock_drift": "FA-CLOCK-DRIFT",
    "streaming/fa_frame_buffer": "FA-FRAME-BUFFER",
    "streaming/fa_jitter_buffer": "FA-JITTER-BUFFER",
    "streaming/fa_latency_compensation": "FA-LATENCY-COMPENSATION",
    "streaming/fa_overlap_add": "FA-OA",
    "streaming/fa_packet_loss_concealment": "FA-PLC",
    "streaming/fa_time_alignment": "FA-TIME-ALIGNMENT",
}
PROCESSING_TEST_TRACE_PREFIXES = {
    "processing/analysis/fa_cqt": "FA-CQT",
    "processing/analysis/fa_log_mel": "FA-LOG-MEL",
    "processing/analysis/fa_loudness": "FA-LOUDNESS",
    "processing/analysis/fa_mfcc": "FA-MFCC",
    "processing/analysis/fa_onset": "FA-ONSET",
    "processing/analysis/fa_pitch": "FA-PITCH",
    "processing/analysis/fa_stft": "FA-STFT",
    "processing/analysis/fa_tempo": "FA-TEMPO",
    "processing/correction/fa_aec_linear": "FA-AEC-LINEAR",
    "processing/correction/fa_aec_nn": "FA-AEC-NN",
    "processing/correction/fa_dc_offset_removal": "FA-DC-OFFSET-REMOVAL",
    "processing/correction/fa_declick": "FA-DECLICK",
    "processing/correction/fa_denoise": "FA-DENOISE",
    "processing/correction/fa_hum": "FA-HUM",
    "processing/dynamics/fa_agc": "FA-AGC",
    "processing/dynamics/fa_compressor": "FA-COMPRESSOR",
    "processing/dynamics/fa_expander": "FA-EXPANDER",
    "processing/dynamics/fa_gain": "FA-GAIN",
    "processing/dynamics/fa_limiter": "FA-LIMITER",
    "processing/dynamics/fa_noise_gate": "FA-NOISE-GATE",
    "processing/dynamics/fa_normalize": "FA-NORMALIZE",
    "processing/frequency/fa_band_pass": "FA-BAND-PASS",
    "processing/frequency/fa_deesser": "FA-DEESSER",
    "processing/frequency/fa_eq": "FA-EQ",
    "processing/frequency/fa_high_pass": "FA-HIGH-PASS",
    "processing/frequency/fa_low_pass": "FA-LOW-PASS",
    "processing/frequency/fa_notch": "FA-NOTCH",
    "processing/format/fa_bit_depth": "FA-BIT-DEPTH",
    "processing/format/fa_channel_convert": "FA-CHANNEL-CONVERT",
    "processing/format/fa_decode": "FA-DECODE",
    "processing/format/fa_encode": "FA-ENCODE",
    "processing/format/fa_interleave": "FA-INTERLEAVE",
    "processing/format/fa_resample": "FA-RESAMPLE",
    "processing/format/fa_sample_format": "FA-SAMPLE-FORMAT",
    "processing/generation/fa_tts": "FA-TTS",
    "processing/routing/fa_bus_router": "FA-BUS-ROUTER",
    "processing/routing/fa_ducking": "FA-DUCKING",
    "processing/routing/fa_loopback": "FA-LOOPBACK",
    "processing/routing/fa_mix": "FA-MIX",
    "processing/routing/fa_monitor_mix": "FA-MONITOR-MIX",
    "processing/routing/fa_patchbay": "FA-PATCHBAY",
    "processing/routing/fa_sidechain": "FA-SIDECHAIN",
    "processing/spatial/fa_beamforming": "FA-BEAMFORMING",
    "processing/spatial/fa_downmix": "FA-DOWNMIX",
    "processing/spatial/fa_pan": "FA-PAN",
    "processing/spatial/fa_stereo_widening": "FA-STEREO-WIDENING",
    "processing/spatial/fa_upmix": "FA-UPMIX",
    "processing/temporal/fa_crossfade": "FA-CROSSFADE",
    "processing/temporal/fa_delay": "FA-DELAY",
    "processing/temporal/fa_echo": "FA-ECHO",
    "processing/temporal/fa_fade": "FA-FADE",
    "processing/temporal/fa_reverb": "FA-REVERB",
    "processing/temporal/fa_silence_removal": "FA-SILENCE-REMOVAL",
    "processing/temporal/fa_trim": "FA-TRIM",
    "processing/temporal/fa_window": "FA-WINDOW",
}


def _buildable_package_dirs() -> list[Path]:
    return sorted({package_xml.parent for package_xml in SRC_ROOT.rglob("package.xml")})


def _relative_package_path(package_dir: Path) -> str:
    return str(package_dir.relative_to(SRC_ROOT))


def _traceability_mapped_package_paths() -> set[str]:
    mapped_paths: set[str] = set(IO_TEST_TRACE_PREFIXES)
    mapped_paths.update(INTERFACE_TEST_TRACE_PREFIXES)
    mapped_paths.update(APP_TEST_TRACE_PREFIXES)
    mapped_paths.update(STREAMING_TEST_TRACE_PREFIXES)
    mapped_paths.update(PROCESSING_TEST_TRACE_PREFIXES)
    mapped_paths.update(f"ai/{package_name}" for package_name in AI_TEST_TRACE_PREFIXES)
    mapped_paths.add("system/fluent_audio_system")
    return mapped_paths


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


def test_system_package_does_not_depend_on_optional_node_packages() -> None:
    package_xml = ET.parse(PACKAGE_ROOT / "package.xml")
    dependency_tags = {
        "depend",
        "build_depend",
        "build_export_depend",
        "exec_depend",
        "test_depend",
    }
    package_dependencies = {
        (element.text or "").strip()
        for element in package_xml.getroot()
        if element.tag in dependency_tags
    }
    forbidden_dependencies = {
        package_dir.name
        for package_dir in _buildable_package_dirs()
        if package_dir.name.startswith("fa_")
    }

    assert package_dependencies & forbidden_dependencies == set()


def _production_python_files_under(root: Path) -> list[Path]:
    production_files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        relative_parts = path.relative_to(root).parts
        if "test" in relative_parts or "__pycache__" in relative_parts:
            continue
        production_files.append(path)
    return production_files


def test_ai_production_python_keeps_explicit_backend_boundaries() -> None:
    forbidden_patterns = (
        ("Any import", r"\btyping\s+import\s+.*\bAny\b"),
        ("Any annotation", r"(:|->)\s*Any\b"),
        ("dict Any", r"\b(Dict|dict)\[str,\s*Any\]"),
        ("object annotation", r"(:|->)\s*object\b"),
        ("object container", r"\b(list|dict)\[[^\]]*object[^\]]*\]"),
        ("ImportError branch", r"\bImportError\b"),
    )
    violations: list[str] = []

    for path in _production_python_files_under(SRC_ROOT / "ai"):
        source = path.read_text(encoding="utf-8")
        for label, pattern in forbidden_patterns:
            if re.search(pattern, source):
                violations.append(f"{path.relative_to(SRC_ROOT)}: {label}")

    assert violations == []


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


def test_traceability_gate_covers_every_buildable_node_package() -> None:
    buildable_package_paths = {
        _relative_package_path(package_dir) for package_dir in _buildable_package_dirs()
    }

    assert buildable_package_paths - _traceability_mapped_package_paths() == set()


def test_fluent_audio_system_colcon_test_runs_pytest() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")

    assert 'extras_require={"test": ["pytest"]}' in setup_text
    assert "cmdclass=" not in setup_text
    assert "PytestCommand" not in setup_text


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
