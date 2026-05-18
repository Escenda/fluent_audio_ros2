from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_limiter_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (
        package_root() / "include" / "fa_limiter" / "fa_limiter_node.hpp"
    ).read_text(encoding="utf-8")


def read_main_source() -> str:
    return (package_root() / "src" / "main.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root() / "include" / "fa_limiter" / "backends" / "internal_limiter.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_limiter.cpp").read_text(
        encoding="utf-8"
    )


def test_example_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_limiter"]["ros__parameters"]

    assert params["input_topic"] == "fa_limiter/input"
    assert params["output_topic"] == "fa_limiter/output"
    assert params["input_stream_id"] == "audio/gain/mic"
    assert params["output"]["stream_id"] == "audio/limit/mic"
    assert params["input_topic"] != params["input_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_id"] != params["output"]["stream_id"]
    assert params["threshold"]["linear"] == 1.0
    assert 0.0 < params["threshold"]["linear"] <= 1.0
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


def test_limiter_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "gain_",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_limiter_validates_frame_contract_before_processing() -> None:
    header = read_node_header()
    source = read_node_source()
    validate_frame = source.split("bool FaLimiterNode::validateFrame")[1].split(
        "bool FaLimiterNode::applyLimiter"
    )[0]
    handle_frame = source.split("void FaLimiterNode::handleFrame")[1].split(
        "bool FaLimiterNode::validateFrame"
    )[0]

    assert "explicit FaLimiterNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());" in header
    assert ': rclcpp::Node("fa_limiter", options)' in source
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


def test_limiter_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_limiter = source.split("bool FaLimiterNode::applyLimiter")[1].split(
        "void FaLimiterNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_limiter
    assert "out.stream_id = config_.output_stream_id;" in apply_limiter
    assert ".rms" not in apply_limiter
    assert ".peak" not in apply_limiter
    assert ".vad" not in apply_limiter


def test_limiter_algorithm_uses_backend_for_explicit_threshold_limiting() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalLimiterBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]
    apply_limiter = node_source.split("bool FaLimiterNode::applyLimiter")[1].split(
        "void FaLimiterNode::publishDiagnostics"
    )[0]

    assert "class InternalLimiterBackend" in header
    assert "struct ProcessResult" in header
    assert "uint64_t samples_limited" in header
    assert "enum class ProcessStatus" in header
    assert "out_sample = threshold_;" in process
    assert "out_sample = -threshold_;" in process
    assert "++limited_in_frame;" in process
    assert "output = std::move(next_output);" in process
    assert "backend_->process(in.data, out.data)" in apply_limiter
    assert "samples_limited_.fetch_add(result.samples_limited);" in apply_limiter
    forbidden_clamp = "std::"
    forbidden_clamp += "clamp"
    assert forbidden_clamp not in process


def test_limiter_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kNonFiniteOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "return ProcessResult{ProcessStatus::kMisalignedInput, 0};" in backend_source
    assert "return ProcessResult{ProcessStatus::kNonFiniteInput, 0};" in backend_source
    assert 'throw std::logic_error("unhandled limiter backend process status")' in backend_source
    assert "unknown limiter backend status" not in backend_source
    assert "backends::processStatusMessage(result.status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_threshold_parameter_is_required_and_range_checked() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = source.split("void FaLimiterNode::loadParameters")[1].split(
        "void FaLimiterNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<double>("threshold.linear");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"input_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDouble(*this, \"threshold.linear\")" in load_parameters
    assert "throw std::runtime_error(\"input_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"input_stream_id is required\")" in load_parameters
    assert "throw std::runtime_error(\"output.stream_id is required\")" in load_parameters
    assert "input_topic and output_topic must be distinct" in load_parameters
    assert "sameIdentityString(config_.input_stream_id, config_.output_stream_id)" in load_parameters
    assert "input_stream_id and output.stream_id must be distinct" in load_parameters
    assert "input_stream_id must be distinct from ROS topics" in load_parameters
    assert "output.stream_id must be distinct from ROS topics" in load_parameters
    assert "config_.threshold_linear <= 0.0" in load_parameters
    assert "config_.threshold_linear > 1.0" in load_parameters
    assert "threshold.linear must be finite and in (0.0, 1.0]" in load_parameters
    assert "expected.sample_rate must satisfy 0 < value <= 384000" in load_parameters
    assert "expected.channels must satisfy 0 < value <= 64" in load_parameters
    assert "diagnostics.qos.depth must be > 0" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "config_.threshold_linear <= 0.0" in backend_source
    assert "config_.threshold_linear > 1.0" in backend_source


def test_limiter_runtime_config_types_do_not_define_meaningful_defaults() -> None:
    header = read_node_header()
    backend_header = read_backend_header()
    source = read_node_source()

    forbidden_defaults = (
        "threshold_linear" + "{-1.0}",
        "threshold_" + "{1.0F}",
        "channels" + "{-1}",
        "expected_channels" + "{-1}",
        "qos_reliable" + "{false}",
        "diagnostics_qos_reliable" + "{false}",
    )
    combined = header + "\n" + backend_header
    for token in forbidden_defaults:
        assert token not in combined

    assert "InternalLimiterConfig() = delete;" in backend_header
    assert "InternalLimiterConfig(int channels_value, double threshold_linear_value);" in backend_header
    assert "config_.threshold_linear = readRequiredDouble(*this, \"threshold.linear\")" in source
    assert "config_.diagnostics_qos_reliable = readRequiredBool(*this, \"diagnostics.qos.reliable\")" in source


def test_diagnostics_include_parameters_counters_and_explicit_qos() -> None:
    source = read_node_source()
    publish_diagnostics = source.split("void FaLimiterNode::publishDiagnostics")[1].split(
        "}  // namespace fa_limiter"
    )[0]
    setup_interfaces = source.split("void FaLimiterNode::setupInterfaces")[1].split(
        "void FaLimiterNode::handleFrame"
    )[0]

    assert '"threshold_linear"' in publish_diagnostics
    assert '"frames_in"' in publish_diagnostics
    assert '"frames_out"' in publish_diagnostics
    assert '"frames_dropped"' in publish_diagnostics
    assert '"samples_limited"' in publish_diagnostics
    assert '"input_topic"' in publish_diagnostics
    assert '"output_topic"' in publish_diagnostics
    assert '"input_stream_id"' in publish_diagnostics
    assert '"output_stream_id"' in publish_diagnostics
    assert '"qos.depth"' in publish_diagnostics
    assert '"diagnostics.qos.depth"' in publish_diagnostics
    assert '"diagnostics.qos.reliable"' in publish_diagnostics
    assert "rclcpp::QoS diagnostics_qos(static_cast<size_t>(config_.diagnostics_qos_depth));" in setup_interfaces
    assert "config_.diagnostics_qos_reliable" in setup_interfaces
    assert "diagnostics_qos.reliable();" in setup_interfaces
    assert "diagnostics_qos.best_effort();" in setup_interfaces


def test_forbidden_runtime_fallback_patterns_are_absent() -> None:
    files = (
        package_root() / "include" / "fa_limiter" / "fa_limiter_node.hpp",
        package_root() / "include" / "fa_limiter" / "backends" / "internal_limiter.hpp",
        package_root() / "src" / "fa_limiter_node.cpp",
        package_root() / "src" / "backends" / "internal_limiter.cpp",
        package_root() / "launch" / "fa_limiter.launch.py",
    )
    combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

    forbidden = (
        "System" + "Defaults" + "QoS",
        "std::max",
        "FindPackage" + "Share",
        "PathJoin" + "Substitution",
        "default_" + "value",
        "dict[str, " + "A" + "ny]",
        "except " + "ImportError",
    )
    for token in forbidden:
        assert token not in combined

    assert "declare_parameter<std::string>(\"input_topic\", config_" not in combined
    assert "declare_parameter<double>(\"threshold.linear\", config_" not in combined
    assert "declare_parameter<int>(\"diagnostics.qos.depth\", config_" not in combined


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_limiter.md",
        "config/default.yaml",
        "launch/fa_limiter.launch.py",
        "include/fa_limiter/fa_limiter_node.hpp",
        "include/fa_limiter/backends/internal_limiter.hpp",
        "src/fa_limiter_node.cpp",
        "src/main.cpp",
        "src/backends/internal_limiter.cpp",
        "test/cpp/test_internal_limiter_backend.cpp",
        "test/cpp/test_limiter_graph.cpp",
        "test/launch/test_fa_limiter_launch_contract.py",
        "test/unit/test_fa_limiter_audio_frame_contract.py",
        "test/integration",
        "test/launch",
        "test/fixtures",
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
    assert "add_library(fa_limiter_node_core" in cmake_text
    assert "src/fa_limiter_node.cpp" in cmake_text
    assert "add_executable(fa_limiter_node" in cmake_text
    assert "src/main.cpp" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "test/cpp/test_limiter_graph.cpp" in cmake_text
    assert "fa_limiter_node_core" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
    assert "int main(" not in node_source
    assert "int main(int argc, char ** argv)" in main_source
