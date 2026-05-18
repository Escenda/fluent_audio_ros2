from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_agc_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (package_root() / "include" / "fa_agc" / "fa_agc_node.hpp").read_text(
        encoding="utf-8"
    )


def read_main_source() -> str:
    return (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_agc" / "backends" / "internal_rms_agc.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_rms_agc.cpp").read_text(
        encoding="utf-8"
    )


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_agc"]["ros__parameters"]

    assert params["input_topic"] == "fa_agc/input"
    assert params["output_topic"] == "fa_agc/output"
    assert params["input_stream_id"] == "audio/compressed/mic"
    assert params["output"]["stream_id"] == "audio/agc/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["agc"]["target_rms"] == 0.1
    assert params["agc"]["min_gain"] == 0.25
    assert params["agc"]["max_gain"] == 4.0
    assert params["agc"]["attack_ms"] == 10.0
    assert params["agc"]["release_ms"] == 250.0
    assert 0.0 < params["agc"]["target_rms"] <= 1.0
    assert 0.0 < params["agc"]["min_gain"] <= 1.0
    assert params["agc"]["max_gain"] >= 1.0
    assert params["agc"]["min_gain"] <= params["agc"]["max_gain"]
    assert params["agc"]["attack_ms"] > 0.0
    assert params["agc"]["release_ms"] > 0.0
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


def test_agc_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "std::" + "clamp",
        "SND_PCM",
        "snd_pcm",
        "resample",
        "set_channels",
        "convert",
        "applyLimiter",
        "applyCompressor",
        "applyNormalize",
        "applyNoiseGate",
        "device_gain",
        "fa_in/",
        "fa_in::",
        'package="fa_in"',
        "low_pass",
        "high_pass",
        "denoise",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    node_source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = node_source.split("void FaAgcNode::loadParameters")[1].split(
        "void FaAgcNode::configureBackend"
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
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"input_stream_id is required\")" in load_parameters
    assert "throw std::runtime_error(\"output.stream_id is required\")" in load_parameters
    assert "input_topic and output_topic must be distinct" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "config_.target_rms <= 0.0" in load_parameters
    assert "config_.target_rms > 1.0" in load_parameters
    assert "agc.target_rms must be finite and in (0.0, 1.0]" in load_parameters
    assert "config_.min_gain <= 0.0" in load_parameters
    assert "agc.min_gain must be finite and > 0.0" in load_parameters
    assert "config_.max_gain < config_.min_gain" in load_parameters
    assert "agc.max_gain must be finite and >= agc.min_gain" in load_parameters
    assert "config_.min_gain > 1.0 || config_.max_gain < 1.0" in load_parameters
    assert "agc.min_gain <= 1.0 <= agc.max_gain is required for initial gain" in load_parameters
    assert "config_.attack_ms <= 0.0" in load_parameters
    assert "config_.release_ms <= 0.0" in load_parameters
    assert "fa_agc requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_agc requires expected.bit_depth=32" in load_parameters
    assert "fa_agc requires expected.layout=interleaved" in load_parameters
    assert "expected.sample_rate must satisfy 0 < value <= 384000" in load_parameters
    assert "expected.channels must satisfy 0 < value <= 64" in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "config_.sample_rate <= 0" in backend_source
    assert "config_.channels <= 0" in backend_source
    assert "config_.target_rms <= 0.0" in backend_source
    assert "config_.max_gain < config_.min_gain" in backend_source


def test_runtime_config_types_do_not_define_meaningful_defaults() -> None:
    node_header = read_node_header()
    backend_header = read_backend_header()

    forbidden_node_defaults = (
        "target_rms" + "{0.1}",
        "min_gain" + "{0.25}",
        "max_gain" + "{4.0}",
        "attack_ms" + "{10.0}",
        "release_ms" + "{250.0}",
        "qos_reliable" + "{false}",
        "diagnostics_qos_reliable" + "{false}",
    )
    for token in forbidden_node_defaults:
        assert token not in node_header

    forbidden_backend_defaults = (
        "sample_rate" + "{-1}",
        "channels" + "{-1}",
        "target_rms" + "{0.1}",
        "min_gain" + "{0.25}",
        "max_gain" + "{4.0}",
        "attack_ms" + "{10.0}",
        "release_ms" + "{250.0}",
    )
    for token in forbidden_backend_defaults:
        assert token not in backend_header

    assert "InternalRmsAgcConfig() = delete;" in backend_header
    assert "InternalRmsAgcConfig(" in backend_header


def test_runtime_frame_validation_drops_invalid_frames() -> None:
    source = read_node_source()
    handle_frame = source.split("void FaAgcNode::handleFrame")[1].split(
        "bool FaAgcNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaAgcNode::validateFrame")[1].split(
        "bool FaAgcNode::applyAgc"
    )[0]

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


def test_agc_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_agc = source.split("bool FaAgcNode::applyAgc")[1].split(
        "void FaAgcNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_agc
    assert "out.stream_id = config_.output_stream_id;" in apply_agc
    assert "backend_->process(in.data, out.data)" in apply_agc
    assert ".vad" not in apply_agc
    assert ".asr" not in apply_agc


def test_frame_rms_target_gain_and_smoothing_are_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "class InternalRmsAgcBackend" in header
    assert "struct ProcessResult" in header
    assert "enum class GainDirection" in header
    assert "double InternalRmsAgcBackend::calculateFrameRms" in backend_source
    assert "square_sum += value * value;" in backend_source
    assert "return std::sqrt(mean_square);" in backend_source
    assert "double InternalRmsAgcBackend::boundedTargetGain" in backend_source
    assert "target_gain = config_.target_rms / frame_rms;" in backend_source
    assert "target_gain < config_.min_gain" in backend_source
    assert "target_gain > config_.max_gain" in backend_source
    assert "double InternalRmsAgcBackend::smoothingAlpha" in backend_source
    assert "1.0 - std::exp(-frame_seconds / time_constant_seconds)" in backend_source
    assert "target_gain < current_gain_" in backend_source
    assert "current_gain_ + (alpha * (target_gain - current_gain_))" in backend_source
    assert "calculateFrameRms" not in node_source
    assert "boundedTargetGain" not in node_source
    assert "smoothingAlpha" not in node_source


def test_agc_drops_non_finite_or_out_of_range_samples_instead_of_clamping() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalRmsAgcBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(output_sample)" in process
    assert "output_sample < kNormalizedMin || output_sample > kNormalizedMax" in process
    assert "ProcessStatus::kNonFiniteInput" in process
    assert "ProcessStatus::kOutOfRangeInput" in process
    assert "ProcessStatus::kOutOfRangeOutput" in process
    forbidden_clamp = "std::"
    forbidden_clamp += "clamp"
    assert forbidden_clamp not in backend_source


def test_output_range_drop_does_not_commit_candidate_gain_or_output() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalRmsAgcBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    validation_before_commit = process.split("current_gain_ = candidate_gain;")[0]
    assert "std::vector<uint8_t> next_output(input.size());" in validation_before_commit
    assert "ProcessStatus::kOutOfRangeOutput" in validation_before_commit
    assert "output = std::move(next_output);" not in validation_before_commit
    assert "current_gain_ = candidate_gain;" in process
    assert "last_frame_rms_ = frame_rms;" in process
    assert "last_target_gain_ = target_gain;" in process
    assert "output = std::move(next_output);" in process


def test_agc_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "enum class ProcessStatus" in header
    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteGain" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "ProcessStatus::kMisalignedInput" in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeOutput" in backend_source
    assert 'throw std::logic_error("unhandled AGC backend process status")' in backend_source
    assert "unknown AGC backend status" not in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source
    assert "GainDirection::kReduction" in node_source
    assert "GainDirection::kIncrease" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_diagnostics_include_parameters_state_and_counters() -> None:
    source = read_node_source()
    publish_diagnostics = source.split("void FaAgcNode::publishDiagnostics")[1].split(
        "}  // namespace fa_agc"
    )[0]

    assert 'status.name = "fa_agc";' in publish_diagnostics
    assert '"target_rms"' in publish_diagnostics
    assert '"min_gain"' in publish_diagnostics
    assert '"max_gain"' in publish_diagnostics
    assert '"attack_ms"' in publish_diagnostics
    assert '"release_ms"' in publish_diagnostics
    assert '"current_gain"' in publish_diagnostics
    assert '"last_frame_rms"' in publish_diagnostics
    assert '"last_target_gain"' in publish_diagnostics
    assert '"frames_in"' in publish_diagnostics
    assert '"frames_out"' in publish_diagnostics
    assert '"frames_dropped"' in publish_diagnostics
    assert '"gain_reductions"' in publish_diagnostics
    assert '"gain_increases"' in publish_diagnostics
    assert '"input_topic"' in publish_diagnostics
    assert '"output_topic"' in publish_diagnostics
    assert '"input_stream_id"' in publish_diagnostics
    assert '"output_stream_id"' in publish_diagnostics
    assert '"qos.depth"' in publish_diagnostics
    assert '"diagnostics.qos.depth"' in publish_diagnostics
    assert '"diagnostics.qos.reliable"' in publish_diagnostics
    assert "backend_->currentGain()" in publish_diagnostics
    assert "backend_->lastFrameRms()" in publish_diagnostics
    assert "backend_->lastTargetGain()" in publish_diagnostics


def test_agc_uses_explicit_diagnostics_qos() -> None:
    source = read_node_source()
    setup_interfaces = source.split("void FaAgcNode::setupInterfaces")[1].split(
        "void FaAgcNode::handleFrame"
    )[0]

    assert "rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth))" in setup_interfaces
    assert "config_.diagnostics_qos_reliable" in setup_interfaces
    assert "diagnostics_qos.best_effort()" in setup_interfaces
    forbidden_system_qos = "rclcpp::SystemDefaults"
    forbidden_system_qos += "QoS()"
    assert forbidden_system_qos not in setup_interfaces


def test_agc_contract_forbidden_runtime_patterns_are_absent() -> None:
    relative_paths = (
        "include/fa_agc/fa_agc_node.hpp",
        "include/fa_agc/backends/internal_rms_agc.hpp",
        "src/fa_agc_node.cpp",
        "src/backends/internal_rms_agc.cpp",
        "src/main.cpp",
        "launch/fa_agc.launch.py",
    )
    forbidden_qos = "QoS("
    forbidden_qos += "std::"
    forbidden_qos += "max"
    forbidden_std_max = "std::"
    forbidden_std_max += "max("
    forbidden_declare = 'declare_parameter("input_topic", '
    forbidden_declare += "config_"
    forbidden_import_error = "except "
    forbidden_import_error += "Import"
    forbidden_import_error += "Error"
    forbidden_tokens = (
        "SystemDefaults" + "QoS",
        "dict[str, " + "An" + "y]",
        forbidden_import_error,
        forbidden_std_max,
        forbidden_qos,
        "std::" + "clamp",
        forbidden_declare,
        "PathJoin" + "Substitution",
        "FindPackage" + "Share",
        "default_" + "value=",
    )
    forbidden_whole_words = ("An" + "y", "ob" + "ject")

    for relative_path in relative_paths:
        text = (package_root() / relative_path).read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in text
        for token in forbidden_whole_words:
            assert f" {token} " not in text
            assert f"<{token}>" not in text
            assert f": {token}" not in text


def test_node_core_is_constructible_with_node_options_for_graph_tests() -> None:
    header = read_node_header()
    node_source = read_node_source()
    main_source = read_main_source()

    assert (
        "explicit FaAgcNode(const rclcpp::NodeOptions & options = "
        "rclcpp::NodeOptions());"
    ) in header
    assert "FaAgcNode::FaAgcNode(const rclcpp::NodeOptions & options)" in node_source
    assert ': rclcpp::Node("fa_agc", options)' in node_source
    assert "int main(" not in node_source
    assert "auto node = std::make_shared<fa_agc::FaAgcNode>();" in main_source


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_rms_agc.md",
        "config/default.yaml",
        "launch/fa_agc.launch.py",
        "include/fa_agc/fa_agc_node.hpp",
        "include/fa_agc/backends/internal_rms_agc.hpp",
        "src/fa_agc_node.cpp",
        "src/main.cpp",
        "src/backends/internal_rms_agc.cpp",
        "test/cpp/test_internal_rms_agc_backend.cpp",
        "test/cpp/test_agc_graph.cpp",
        "test/launch/test_fa_agc_launch_contract.py",
        "test/unit/test_fa_agc_audio_frame_contract.py",
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
    assert "add_library(fa_agc_node_core" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "test/cpp/test_agc_graph.cpp" in cmake_text
    assert "fa_agc_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
