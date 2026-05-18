from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_normalize_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (
        package_root() / "include" / "fa_normalize" / "fa_normalize_node.hpp"
    ).read_text(encoding="utf-8")


def read_main_source() -> str:
    return (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_normalize"
        / "backends"
        / "internal_peak_normalize.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_peak_normalize.cpp").read_text(
        encoding="utf-8"
    )


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_normalize"]["ros__parameters"]

    assert params["input_topic"] == "fa_normalize/input"
    assert params["output_topic"] == "fa_normalize/output"
    assert params["input_stream_id"] == "audio/noise_gated/mic"
    assert params["output"]["stream_id"] == "audio/normalized/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["normalize"]["target_peak_linear"] == 0.9
    assert params["normalize"]["silence_threshold_linear"] == 0.0001
    assert 0.0 < params["normalize"]["target_peak_linear"] <= 1.0
    assert 0.0 <= params["normalize"]["silence_threshold_linear"]
    assert params["normalize"]["silence_threshold_linear"] < params["normalize"]["target_peak_linear"]
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


def test_normalize_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "std::" + "clamp",
        "compress",
        "limiter",
        "gate.",
        "filter.",
        "denoise",
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
    load_parameters = node_source.split("void FaNormalizeNode::loadParameters")[1].split(
        "void FaNormalizeNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<double>("normalize.target_peak_linear");' in load_parameters
    assert 'this->declare_parameter<double>("normalize.silence_threshold_linear");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"normalize.target_peak_linear\")" in load_parameters
    assert "readRequiredDouble(*this, \"normalize.silence_threshold_linear\")" in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"input_stream_id is required\")" in load_parameters
    assert "throw std::runtime_error(\"output.stream_id is required\")" in load_parameters
    assert "input_topic and output_topic must be distinct" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.output_stream_id)" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "!isFinite(config_.target_peak_linear)" in load_parameters
    assert "config_.target_peak_linear <= 0.0" in load_parameters
    assert "config_.target_peak_linear > 1.0" in load_parameters
    assert "!isFinite(config_.silence_threshold_linear)" in load_parameters
    assert "config_.silence_threshold_linear < 0.0" in load_parameters
    assert "config_.silence_threshold_linear >= config_.target_peak_linear" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "requires expected.encoding=FLOAT32LE" in load_parameters
    assert "requires expected.bit_depth=32" in load_parameters
    assert "requires expected.layout=interleaved" in load_parameters
    assert "expected.sample_rate must satisfy 0 < value <= 384000" in load_parameters
    assert "expected.channels must satisfy 0 < value <= 64" in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "config_.channels <= 0" in backend_source
    assert "config_.target_peak_linear <= 0.0" in backend_source
    assert "config_.target_peak_linear > 1.0" in backend_source
    assert "config_.silence_threshold_linear >= config_.target_peak_linear" in backend_source


def test_normalize_validates_frame_contract_before_processing() -> None:
    header = read_node_header()
    source = read_node_source()
    handle_frame = source.split("void FaNormalizeNode::handleFrame")[1].split(
        "bool FaNormalizeNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaNormalizeNode::validateFrame")[1].split(
        "bool FaNormalizeNode::applyNormalize"
    )[0]

    assert "explicit FaNormalizeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());" in header
    assert ': rclcpp::Node("fa_normalize", options)' in source
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


def test_normalize_preserves_frame_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_normalize = source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_normalize
    assert "out.stream_id = config_.output_stream_id;" in apply_normalize
    assert "backend_->process(in.data, out.data)" in apply_normalize
    assert ".rms" not in apply_normalize
    assert ".vad" not in apply_normalize
    assert ".epoch" not in apply_normalize


def test_peak_normalize_algorithm_is_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_normalize = node_source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "class InternalPeakNormalizeBackend" in header
    assert "enum class ProcessMode" in header
    assert "struct ProcessResult" in header
    assert "float peak = 0.0F;" in process
    assert "peak = std::max(peak, std::abs(sample));" in process
    assert "config_.target_peak_linear / static_cast<double>(peak)" in process
    assert "const double normalized = static_cast<double>(samples[i]) * gain;" in process
    assert "backend_->process(in.data, out.data)" in apply_normalize
    assert "frames_normalized_.fetch_add(1);" in apply_normalize
    assert "last_gain_.store(result.gain);" in apply_normalize
    assert "std::max(peak, std::abs(sample))" not in node_source
    assert "std::memcpy" not in node_source
    forbidden_clamp = "std::"
    forbidden_clamp += "clamp"
    assert forbidden_clamp not in process


def test_silence_pass_through_changes_only_stream_identity_and_does_not_amplify() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_normalize = node_source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    assert "peak < static_cast<float>(config_.silence_threshold_linear)" in process
    assert "output = input;" in process
    assert "ProcessMode::kSilencePassthrough" in process
    assert "frames_silence_passthrough_.fetch_add(1);" in apply_normalize
    assert "last_gain_.store(1.0);" in apply_normalize


def test_normalize_drops_invalid_samples_and_outputs_instead_of_clamping() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalizedSample(sample)" in process
    assert "!isFinite(gain)" in process
    assert "!isFinite(normalized)" in process
    assert "normalized < kNormalizedMin || normalized > kNormalizedMax" in process
    assert "ProcessStatus::kNonFiniteInput" in process
    assert "ProcessStatus::kOutOfRangeInput" in process
    assert "ProcessStatus::kNonFiniteGain" in process
    assert "ProcessStatus::kOutOfRangeOutput" in process
    forbidden_clamp = "std::"
    forbidden_clamp += "clamp"
    assert forbidden_clamp not in process


def test_normalize_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
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
    assert "ProcessStatus::kOutOfRangeInput" in backend_source
    assert 'throw std::logic_error("unhandled normalize backend process status")' in backend_source
    assert "unknown normalize backend status" not in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_rejected_frame_does_not_overwrite_output_or_commit_last_gain() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalPeakNormalizeBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_normalize = node_source.split("bool FaNormalizeNode::applyNormalize")[1].split(
        "void FaNormalizeNode::publishDiagnostics"
    )[0]

    rejection_section = process.split("output = std::move(next_output);")[0]
    assert "std::vector<uint8_t> next_output(input.size());" in rejection_section
    assert "ProcessStatus::kOutOfRangeInput" in rejection_section
    assert "ProcessStatus::kOutOfRangeOutput" in rejection_section
    assert "output = std::move(next_output);" not in rejection_section
    assert "last_gain_.store" not in apply_normalize.split("return false;")[0]
    assert "last_gain_.store(result.gain);" in apply_normalize


def test_diagnostics_publish_config_last_gain_and_counters() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaNormalizeNode::publishDiagnostics")[1].split(
        "}  // namespace fa_normalize"
    )[0]

    assert 'status.name = "fa_normalize";' in diagnostics
    assert '"target_peak_linear"' in diagnostics
    assert '"silence_threshold_linear"' in diagnostics
    assert "backend_->targetPeakLinear()" in diagnostics
    assert "backend_->silenceThresholdLinear()" in diagnostics
    assert '"last_gain"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"frames_silence_passthrough"' in diagnostics
    assert '"frames_normalized"' in diagnostics
    assert '"input_topic"' in diagnostics
    assert '"output_topic"' in diagnostics
    assert '"input_stream_id"' in diagnostics
    assert '"output_stream_id"' in diagnostics
    assert '"qos.depth"' in diagnostics
    assert '"diagnostics.qos.depth"' in diagnostics
    assert '"diagnostics.qos.reliable"' in diagnostics


def test_normalize_runtime_config_types_do_not_define_meaningful_defaults() -> None:
    header = read_node_header()
    backend_header = read_backend_header()

    forbidden_target_default = "double target_peak_linear"
    forbidden_target_default += "{0.9}"
    forbidden_silence_default = "double silence_threshold_linear"
    forbidden_silence_default += "{0.0001}"
    forbidden_channels_default = "expected_channels"
    forbidden_channels_default += "{-1}"
    forbidden_qos_reliable_default = "qos_reliable"
    forbidden_qos_reliable_default += "{false}"
    forbidden_diagnostics_reliable_default = "diagnostics_qos_reliable"
    forbidden_diagnostics_reliable_default += "{false}"
    forbidden_backend_channels_default = "int channels"
    forbidden_backend_channels_default += "{-1}"

    assert forbidden_target_default not in header
    assert forbidden_silence_default not in header
    assert "expected_sample_rate{-1}" not in header
    assert forbidden_channels_default not in header
    assert forbidden_qos_reliable_default not in header
    assert forbidden_diagnostics_reliable_default not in header
    assert forbidden_backend_channels_default not in backend_header
    assert forbidden_target_default not in backend_header
    assert forbidden_silence_default not in backend_header
    assert "InternalPeakNormalizeConfig() = delete;" in backend_header


def test_diagnostics_qos_is_explicit_and_not_system_defaulted() -> None:
    source = read_node_source()
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_normalize"]["ros__parameters"]

    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in source
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in source
    assert "readRequiredInt(*this, \"diagnostics.qos.depth\")" in source
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in source
    assert "diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth))" in source
    assert "diagnostics_qos.reliable();" in source
    assert "diagnostics_qos.best_effort();" in source
    assert "System" + "Defaults" + "QoS" not in source


def test_forbidden_runtime_fallback_patterns_are_absent() -> None:
    source = read_node_source()
    header = read_node_header()
    backend_header = read_backend_header()
    combined = "\n".join([source, header, backend_header])

    assert "PathJoin" + "Substitution" not in (
        package_root() / "launch" / "fa_normalize.launch.py"
    ).read_text(encoding="utf-8")
    assert "FindPackage" + "Share" not in (
        package_root() / "launch" / "fa_normalize.launch.py"
    ).read_text(encoding="utf-8")
    assert "System" + "Defaults" + "QoS" not in combined
    forbidden_clamp = "std::"
    forbidden_clamp += "clamp"
    forbidden_qos_fallback = "std::"
    forbidden_qos_fallback += "max<int>(1, config_.qos_depth)"
    assert forbidden_clamp not in combined
    assert forbidden_qos_fallback not in combined
    forbidden_typed_declare = "declare_parameter<std::string>(\"input_topic\", "
    forbidden_typed_declare += "config_"
    forbidden_untyped_declare = "declare_parameter(\"input_topic\", "
    forbidden_untyped_declare += "config_"
    assert forbidden_typed_declare not in combined
    assert forbidden_untyped_declare not in combined


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_peak_normalize.md",
        "config/default.yaml",
        "launch/fa_normalize.launch.py",
        "include/fa_normalize/fa_normalize_node.hpp",
        "include/fa_normalize/backends/internal_peak_normalize.hpp",
        "src/fa_normalize_node.cpp",
        "src/main.cpp",
        "src/backends/internal_peak_normalize.cpp",
        "test/cpp/test_internal_peak_normalize_backend.cpp",
        "test/cpp/test_normalize_graph.cpp",
        "test/launch/test_fa_normalize_launch_contract.py",
        "test/unit/test_fa_normalize_audio_frame_contract.py",
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
    assert "add_library(fa_normalize_node_core" in cmake_text
    assert "src/fa_normalize_node.cpp" in cmake_text
    assert "add_executable(fa_normalize_node" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "test/cpp/test_normalize_graph.cpp" in cmake_text
    assert "fa_normalize_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "int main(" not in node_source
    assert "int main(int argc, char ** argv)" in main_source
