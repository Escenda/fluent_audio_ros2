from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_compressor_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (
        package_root() / "include" / "fa_compressor" / "fa_compressor_node.hpp"
    ).read_text(encoding="utf-8")


def read_main_source() -> str:
    return (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_compressor"
        / "backends"
        / "internal_static_curve.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_static_curve.cpp").read_text(
        encoding="utf-8"
    )


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_compressor"]["ros__parameters"]

    assert params["input_topic"] == "fa_compressor/input"
    assert params["output_topic"] == "fa_compressor/output"
    assert params["input_stream_id"] == "audio/normalized/mic"
    assert params["output"]["stream_id"] == "audio/compressed/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["compressor"]["threshold_linear"] == 0.5
    assert params["compressor"]["ratio"] == 4.0
    assert params["compressor"]["makeup_gain_linear"] == 1.0
    assert 0.0 < params["compressor"]["threshold_linear"] < 1.0
    assert params["compressor"]["ratio"] > 1.0
    assert 0.0 < params["compressor"]["makeup_gain_linear"] <= 4.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_compressor_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::" + "clamp",
        "clip",
        "SND_PCM",
        "snd_pcm",
        "resample",
        "set_channels",
        "applyLimiter",
        "applyNoiseGate",
        "applyNormalize",
        "low_pass",
        "high_pass",
        "denoise",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = source.split("void FaCompressorNode::loadParameters")[1].split(
        "void FaCompressorNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"input_stream_id is required\")" in load_parameters
    assert "throw std::runtime_error(\"output.stream_id is required\")" in load_parameters
    assert "input_topic and output_topic must be distinct" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "config_.threshold_linear <= 0.0" in load_parameters
    assert "config_.threshold_linear >= 1.0" in load_parameters
    assert "compressor.threshold_linear must be finite and in (0.0, 1.0)" in load_parameters
    assert "config_.ratio <= 1.0" in load_parameters
    assert "compressor.ratio must be finite and > 1.0" in load_parameters
    assert "config_.makeup_gain_linear <= 0.0" in load_parameters
    assert "config_.makeup_gain_linear > 4.0" in load_parameters
    assert "compressor.makeup_gain_linear must be finite and in (0.0, 4.0]" in load_parameters
    assert "fa_compressor requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_compressor requires expected.bit_depth=32" in load_parameters
    assert "fa_compressor requires expected.layout=interleaved" in load_parameters
    assert "expected.sample_rate must satisfy 0 < value <= 384000" in load_parameters
    assert "expected.channels must satisfy 0 < value <= 64" in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "config_.threshold_linear <= 0.0" in backend_source
    assert "config_.ratio <= 1.0" in backend_source
    assert "config_.makeup_gain_linear > 4.0" in backend_source


def test_compressor_runtime_config_types_do_not_define_meaningful_defaults() -> None:
    header = read_node_header()
    backend_header = read_backend_header()
    source = read_node_source()

    forbidden_defaults = (
        "threshold_linear" + "{0.5}",
        "ratio" + "{4.0}",
        "makeup_gain_linear" + "{1.0}",
        "channels" + "{-1}",
        "qos_reliable" + "{false}",
        "diagnostics_qos_reliable" + "{false}",
    )
    combined = header + "\n" + backend_header
    for token in forbidden_defaults:
        assert token not in combined

    assert "InternalStaticCurveConfig() = delete;" in backend_header
    assert "InternalStaticCurveConfig(" in backend_header
    assert "config_.threshold_linear = readRequiredDouble(*this, \"compressor.threshold_linear\")" in source
    assert "config_.ratio = readRequiredDouble(*this, \"compressor.ratio\")" in source
    assert "config_.makeup_gain_linear = readRequiredDouble(*this, \"compressor.makeup_gain_linear\")" in source
    assert "config_.diagnostics_qos_reliable = readRequiredBool(*this, \"diagnostics.qos.reliable\")" in source


def test_runtime_frame_validation_drops_invalid_frames() -> None:
    header = read_node_header()
    source = read_node_source()
    validate_frame = source.split("bool FaCompressorNode::validateFrame")[1].split(
        "bool FaCompressorNode::applyCompressor"
    )[0]
    handle_frame = source.split("void FaCompressorNode::handleFrame")[1].split(
        "bool FaCompressorNode::validateFrame"
    )[0]

    assert "explicit FaCompressorNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());" in header
    assert ': rclcpp::Node("fa_compressor", options)' in source
    assert "if (!msg)" in handle_frame
    assert "frames_dropped_.fetch_add(1);" in handle_frame
    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame


def test_compressor_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_compressor = source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_compressor
    assert "out.stream_id = config_.output_stream_id;" in apply_compressor
    assert ".rms" not in apply_compressor
    assert ".peak" not in apply_compressor
    assert ".vad" not in apply_compressor


def test_static_compression_curve_is_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalStaticCurveBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_compressor = node_source.split("bool FaCompressorNode::applyCompressor")[1].split(
        "void FaCompressorNode::publishDiagnostics"
    )[0]

    assert "class InternalStaticCurveBackend" in header
    assert "struct ProcessResult" in header
    assert "uint64_t samples_compressed" in header
    assert "const double amplitude = std::abs(static_cast<double>(sample));" in process
    assert "if (amplitude > config_.threshold_linear)" in process
    assert "config_.threshold_linear + ((amplitude - config_.threshold_linear) / config_.ratio)" in process
    assert "std::signbit(sample) ? -compressed_abs : compressed_abs" in process
    assert "const double output_sample = signed_sample * config_.makeup_gain_linear;" in process
    assert "backend_->process(in.data, out.data)" in apply_compressor
    assert "samples_compressed_.fetch_add(result.samples_compressed);" in apply_compressor


def test_compressor_drops_non_finite_or_out_of_range_samples_instead_of_clamping() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalStaticCurveBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(output_sample)" in process
    assert "output_sample < kNormalizedMin || output_sample > kNormalizedMax" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeInput, 0};" in process
    assert "return ProcessResult{ProcessStatus::kOutOfRangeOutput, 0};" in process
    forbidden_clamp = "std::"
    forbidden_clamp += "clamp"
    assert forbidden_clamp not in process
    assert "backends::processStatusMessage(result.status)" in node_source


def test_compressor_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
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
    assert "return ProcessResult{ProcessStatus::kMisalignedInput, 0};" in backend_source
    assert "return ProcessResult{ProcessStatus::kNonFiniteInput, 0};" in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_diagnostics_include_parameters_and_counters() -> None:
    source = read_node_source()
    publish_diagnostics = source.split("void FaCompressorNode::publishDiagnostics")[1].split(
        "}  // namespace fa_compressor"
    )[0]

    assert 'status.name = "fa_compressor";' in publish_diagnostics
    assert '"threshold_linear"' in publish_diagnostics
    assert '"ratio"' in publish_diagnostics
    assert '"makeup_gain_linear"' in publish_diagnostics
    assert '"frames_in"' in publish_diagnostics
    assert '"frames_out"' in publish_diagnostics
    assert '"frames_dropped"' in publish_diagnostics
    assert '"samples_compressed"' in publish_diagnostics
    assert '"input_topic"' in publish_diagnostics
    assert '"output_topic"' in publish_diagnostics
    assert '"input_stream_id"' in publish_diagnostics
    assert '"output_stream_id"' in publish_diagnostics
    assert '"qos.depth"' in publish_diagnostics
    assert '"diagnostics.qos.depth"' in publish_diagnostics
    assert '"diagnostics.qos.reliable"' in publish_diagnostics


def test_diagnostics_qos_is_explicit_and_not_system_defaulted() -> None:
    source = read_node_source()
    setup_interfaces = source.split("void FaCompressorNode::setupInterfaces")[1].split(
        "void FaCompressorNode::handleFrame"
    )[0]

    assert "rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));" in setup_interfaces
    assert "config_.diagnostics_qos_reliable" in setup_interfaces
    assert "diagnostics_qos.reliable();" in setup_interfaces
    assert "diagnostics_qos.best_effort();" in setup_interfaces
    assert '"diagnostics"' in setup_interfaces
    assert "diagnostics_qos" in setup_interfaces
    forbidden_system_qos = "System"
    forbidden_system_qos += "Defaults"
    forbidden_system_qos += "QoS"
    assert forbidden_system_qos not in setup_interfaces


def test_forbidden_runtime_fallback_patterns_are_absent() -> None:
    files = (
        package_root() / "include" / "fa_compressor" / "fa_compressor_node.hpp",
        package_root() / "include" / "fa_compressor" / "backends" / "internal_static_curve.hpp",
        package_root() / "src" / "fa_compressor_node.cpp",
        package_root() / "src" / "backends" / "internal_static_curve.cpp",
        package_root() / "launch" / "fa_compressor.launch.py",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = (
        "System" + "Defaults" + "QoS",
        "std::" + "clamp",
        "FindPackage" + "Share",
        "PathJoin" + "Substitution",
        "default_" + "value",
        "dict[str, " + "A" + "ny]",
        "except " + "ImportError",
        "std::max",
    )
    for token in forbidden:
        assert token not in combined

    assert "declare_parameter<std::string>(\"input_topic\", config_" not in combined
    assert "declare_parameter<double>(\"compressor.threshold_linear\", config_" not in combined
    assert "declare_parameter<int>(\"diagnostics.qos.depth\", config_" not in combined


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_static_curve.md",
        "config/default.yaml",
        "launch/fa_compressor.launch.py",
        "include/fa_compressor/fa_compressor_node.hpp",
        "include/fa_compressor/backends/internal_static_curve.hpp",
        "src/fa_compressor_node.cpp",
        "src/main.cpp",
        "src/backends/internal_static_curve.cpp",
        "test/cpp/test_internal_static_curve_backend.cpp",
        "test/cpp/test_compressor_graph.cpp",
        "test/launch/test_fa_compressor_launch_contract.py",
        "test/unit/test_fa_compressor_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")
    node_source = read_node_source()
    main_source = read_main_source()

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "add_library(fa_compressor_node_core" in cmake_text
    assert "src/fa_compressor_node.cpp" in cmake_text
    assert "add_executable(fa_compressor_node" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "test/cpp/test_compressor_graph.cpp" in cmake_text
    assert "fa_compressor_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "int main(" not in node_source
    assert "int main(int argc, char ** argv)" in main_source
