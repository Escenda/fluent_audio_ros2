from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_noise_gate_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (
        package_root() / "include" / "fa_noise_gate" / "fa_noise_gate_node.hpp"
    ).read_text(encoding="utf-8")


def read_main_source() -> str:
    return (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_noise_gate"
        / "backends"
        / "internal_threshold_gate.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_threshold_gate.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_noise_gate"]["ros__parameters"]

    assert params["input_topic"] == "audio/dc_offset_removed/mic"
    assert params["output_topic"] == "audio/noise_gated/mic"
    assert params["gate"]["threshold_linear"] == 0.02
    assert params["gate"]["closed_gain_linear"] == 0.0
    assert 0.0 <= params["gate"]["threshold_linear"] <= 1.0
    assert 0.0 <= params["gate"]["closed_gain_linear"] <= 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_noise_gate_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "normalize(",
        "std::clamp",
        "threshold.linear",
        "filter.",
        "denoise",
        "compress",
        "limiter",
        "limit",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_rejects_invalid_config_without_fallback() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = source.split("void FaNoiseGateNode::loadParameters")[1].split(
        "void FaNoiseGateNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<double>("gate.threshold_linear", config_.threshold_linear);' in load_parameters
    assert 'this->declare_parameter<double>("gate.closed_gain_linear", config_.closed_gain_linear);' in load_parameters
    assert "!isFinite(config_.threshold_linear)" in load_parameters
    assert "config_.threshold_linear < 0.0" in load_parameters
    assert "config_.threshold_linear > 1.0" in load_parameters
    assert "!isFinite(config_.closed_gain_linear)" in load_parameters
    assert "config_.closed_gain_linear < 0.0" in load_parameters
    assert "config_.closed_gain_linear > 1.0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters
    assert "config_.threshold_linear < 0.0" in backend_source
    assert "config_.closed_gain_linear < 0.0" in backend_source


def test_noise_gate_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaNoiseGateNode::validateFrame")[1].split(
        "bool FaNoiseGateNode::applyNoiseGate"
    )[0]
    handle_frame = source.split("void FaNoiseGateNode::handleFrame")[1].split(
        "bool FaNoiseGateNode::validateFrame"
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
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_noise_gate_preserves_frame_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_gate = source.split("bool FaNoiseGateNode::applyNoiseGate")[1].split(
        "void FaNoiseGateNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_gate
    assert "out.stream_id = config_.output_topic;" in apply_gate
    assert ".rms" not in apply_gate
    assert ".peak" not in apply_gate
    assert ".vad" not in apply_gate


def test_noise_gate_algorithm_uses_backend_threshold_and_closed_gain_only() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalThresholdGateBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_gate = node_source.split("bool FaNoiseGateNode::applyNoiseGate")[1].split(
        "void FaNoiseGateNode::publishDiagnostics"
    )[0]

    assert "class InternalThresholdGateBackend" in header
    assert "struct ProcessResult" in header
    assert "uint64_t samples_gated" in header
    assert "enum class ProcessStatus" in header
    assert "if (std::abs(sample) < config_.threshold_linear)" in process
    assert "output_sample = static_cast<double>(sample) * config_.closed_gain_linear;" in process
    assert "++gated_in_frame;" in process
    assert "output = std::move(next_output);" in process
    assert "backend_->process(in.data, out.data)" in apply_gate
    assert "samples_gated_.fetch_add(result.samples_gated);" in apply_gate
    assert "else" not in process
    assert "std::clamp" not in process


def test_noise_gate_drops_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalThresholdGateBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(output_sample)" in process
    assert "output_sample < kNormalizedMin || output_sample > kNormalizedMax" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeInput, 0};" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};" in process
    assert "std::clamp" not in process
    assert "normalize(" not in process
    assert "backends::processStatusMessage(result.status)" in node_source


def test_noise_gate_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "return ProcessResult{ProcessStatus::kMisalignedInput, 0};" in backend_source
    assert "return ProcessResult{ProcessStatus::kNonFiniteInput, 0};" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_diagnostics_publish_config_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaNoiseGateNode::publishDiagnostics")[1].split(
        "}  // namespace fa_noise_gate"
    )[0]

    assert 'status.name = "fa_noise_gate";' in diagnostics
    assert '"gate_threshold_linear"' in diagnostics
    assert '"gate_closed_gain_linear"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_gated"' in diagnostics
    assert '"output_topic"' in diagnostics


def test_node_core_is_constructible_with_node_options_for_graph_tests() -> None:
    header = read_node_header()
    node_source = read_node_source()
    main_source = read_main_source()

    assert (
        "explicit FaNoiseGateNode(const rclcpp::NodeOptions & options = "
        "rclcpp::NodeOptions());"
    ) in header
    assert "FaNoiseGateNode::FaNoiseGateNode(const rclcpp::NodeOptions & options)" in node_source
    assert ': rclcpp::Node("fa_noise_gate", options)' in node_source
    assert "int main(" not in node_source
    assert "auto node = std::make_shared<fa_noise_gate::FaNoiseGateNode>();" in main_source


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_threshold_gate.md",
        "config/default.yaml",
        "launch/fa_noise_gate.launch.py",
        "include/fa_noise_gate/fa_noise_gate_node.hpp",
        "include/fa_noise_gate/backends/internal_threshold_gate.hpp",
        "src/fa_noise_gate_node.cpp",
        "src/main.cpp",
        "src/backends/internal_threshold_gate.cpp",
        "test/cpp/test_internal_threshold_gate_backend.cpp",
        "test/cpp/test_noise_gate_graph.cpp",
        "test/launch/test_fa_noise_gate_launch_contract.py",
        "test/unit/test_fa_noise_gate_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "add_library(fa_noise_gate_node_core" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "test/cpp/test_noise_gate_graph.cpp" in cmake_text
    assert "fa_noise_gate_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
