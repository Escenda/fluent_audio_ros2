from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_expander_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_expander"
        / "backends"
        / "internal_static_expander.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_static_expander.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_expander"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/expanded/mic"
    assert params["expander"]["threshold_linear"] == 0.05
    assert params["expander"]["ratio"] == 2.0
    assert 0.0 < params["expander"]["threshold_linear"] < 1.0
    assert params["expander"]["ratio"] > 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_expander_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "std::clamp",
        "closed_gain",
        "target_peak",
        "silence_threshold",
        "lufs",
        "loudness",
        "legacy",
        "compat",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    node_source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = node_source.split("void FaExpanderNode::loadParameters")[1].split(
        "void FaExpanderNode::configureBackend"
    )[0]

    assert (
        'this->declare_parameter<double>("expander.threshold_linear", '
        "config_.threshold_linear);"
    ) in load_parameters
    assert 'this->declare_parameter<double>("expander.ratio", config_.ratio);' in load_parameters
    assert "!isFinite(config_.threshold_linear)" in load_parameters
    assert "config_.threshold_linear <= 0.0" in load_parameters
    assert "config_.threshold_linear >= 1.0" in load_parameters
    assert "!isFinite(config_.ratio)" in load_parameters
    assert "config_.ratio <= 1.0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters
    assert "config_.channels <= 0" in backend_source
    assert "config_.threshold_linear <= 0.0" in backend_source
    assert "config_.threshold_linear >= 1.0" in backend_source
    assert "config_.ratio <= 1.0" in backend_source


def test_expander_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    handle_frame = source.split("void FaExpanderNode::handleFrame")[1].split(
        "bool FaExpanderNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaExpanderNode::validateFrame")[1].split(
        "bool FaExpanderNode::applyExpansion"
    )[0]

    assert "if (!msg)" in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.channels == 0U" in validate_frame
    assert "static_cast<size_t>(msg.channels) * sizeof(float)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_expander_preserves_frame_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_expansion = source.split("bool FaExpanderNode::applyExpansion")[1].split(
        "void FaExpanderNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_expansion
    assert "out.stream_id = config_.output_topic;" in apply_expansion
    assert "backend_->process(in.data, out.data)" in apply_expansion
    assert ".rms" not in apply_expansion
    assert ".peak" not in apply_expansion
    assert ".vad" not in apply_expansion
    assert ".epoch" not in apply_expansion


def test_static_expansion_curve_is_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalStaticExpanderBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_expansion = node_source.split("bool FaExpanderNode::applyExpansion")[1].split(
        "void FaExpanderNode::publishDiagnostics"
    )[0]

    assert "class InternalStaticExpanderBackend" in header
    assert "struct ProcessResult" in header
    assert "uint64_t samples_expanded" in header
    assert "const double magnitude = std::abs(static_cast<double>(sample));" in process
    assert "if (magnitude < config_.threshold_linear)" in process
    assert "config_.threshold_linear *" in process
    assert "std::pow(magnitude / config_.threshold_linear, config_.ratio)" in process
    assert "std::copysign(expanded_abs, static_cast<double>(sample))" in process
    assert "++samples_expanded;" in process
    assert "backend_->process(in.data, out.data)" in apply_expansion
    assert "samples_expanded_.fetch_add(result.samples_expanded);" in apply_expansion
    assert "std::pow" not in node_source
    assert "std::copysign" not in node_source


def test_expander_drops_invalid_samples_and_outputs_instead_of_clamping_or_zeroing() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalStaticExpanderBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(expanded)" in process
    assert "expanded < kNormalizedMin || expanded > kNormalizedMax" in process
    assert "ProcessStatus::kNonFiniteInput" in process
    assert "ProcessStatus::kOutOfRangeInput" in process
    assert "ProcessStatus::kOutOfRangeOutput" in process
    assert "std::clamp" not in process
    assert "out_sample = 0.0" not in process
    assert "expanded = 0.0" not in process


def test_expander_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "ProcessStatus::kMisalignedInput" in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeInput" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_rejected_frame_does_not_overwrite_output() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalStaticExpanderBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    rejection_section = process.split("output = std::move(next_output);")[0]
    assert "std::vector<uint8_t> next_output(input.size());" in rejection_section
    assert "ProcessStatus::kOutOfRangeInput" in rejection_section
    assert "ProcessStatus::kOutOfRangeOutput" in rejection_section
    assert "output = std::move(next_output);" not in rejection_section
    assert "output = std::move(next_output);" in process


def test_diagnostics_publish_config_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaExpanderNode::publishDiagnostics")[1].split(
        "}  // namespace fa_expander"
    )[0]

    assert 'status.name = "fa_expander";' in diagnostics
    assert '"expander_threshold_linear"' in diagnostics
    assert '"expander_ratio"' in diagnostics
    assert "backend_->thresholdLinear()" in diagnostics
    assert "backend_->ratio()" in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_expanded"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_static_expander.md",
        "config/default.yaml",
        "launch/fa_expander.launch.py",
        "include/fa_expander/fa_expander_node.hpp",
        "include/fa_expander/backends/internal_static_expander.hpp",
        "src/fa_expander_node.cpp",
        "src/backends/internal_static_expander.cpp",
        "test/cpp/test_internal_static_expander_backend.cpp",
        "test/unit/test_fa_expander_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
