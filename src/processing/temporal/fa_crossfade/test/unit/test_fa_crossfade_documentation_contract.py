from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def read_node_source() -> str:
    return (PACKAGE_ROOT / "src" / "fa_crossfade_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        PACKAGE_ROOT
        / "include"
        / "fa_crossfade"
        / "backends"
        / "internal_crossfade.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (PACKAGE_ROOT / "src" / "backends" / "internal_crossfade.cpp").read_text(
        encoding="utf-8"
    )


def test_fa_crossfade_has_standard_package_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_crossfade.md",
        "config/default.yaml",
        "launch/fa_crossfade.launch.py",
        "package.xml",
        "CMakeLists.txt",
        "include/fa_crossfade/fa_crossfade_node.hpp",
        "include/fa_crossfade/backends/internal_crossfade.hpp",
        "src/fa_crossfade_node.cpp",
        "src/backends/internal_crossfade.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_crossfade_backend.cpp",
        "test/cpp/test_crossfade_graph.cpp",
        "test/unit/test_fa_crossfade_documentation_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (PACKAGE_ROOT / relative_path).exists()


def test_example_config_declares_pairwise_float32_contract() -> None:
    config = yaml.safe_load((PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_crossfade"]["ros__parameters"]

    assert params["input_a_topic"] == "fa_crossfade/input_a"
    assert params["input_b_topic"] == "fa_crossfade/input_b"
    assert params["output_topic"] == "fa_crossfade/output"
    assert params["input_a_stream_id"] == "audio/segment/a"
    assert params["input_b_stream_id"] == "audio/segment/b"
    assert params["output"]["stream_id"] == "audio/crossfaded/segment"
    assert params["input_a_topic"] != params["input_a_stream_id"]
    assert params["input_b_topic"] != params["input_b_stream_id"]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert len(
        {
            params["input_a_stream_id"],
            params["input_b_stream_id"],
            params["output"]["stream_id"],
        }
    ) == 3
    assert params["crossfade"]["overlap_frames"] == 1600
    assert params["crossfade"]["curve"] == "linear"
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


def test_launch_requires_explicit_config_file_without_package_default() -> None:
    launch_source = (PACKAGE_ROOT / "launch" / "fa_crossfade.launch.py").read_text(
        encoding="utf-8"
    )
    config_argument = launch_source.split('DeclareLaunchArgument(\n            "config_file"')[1].split(
        "        ),",
        1,
    )[0]

    assert "default_value" not in launch_source
    assert "FindPackageShare" not in launch_source
    assert "PathJoinSubstitution" not in launch_source
    assert "config/default.yaml" not in launch_source
    assert "parameters=[config_file]" in launch_source


def test_node_requires_parameters_without_runtime_defaults_and_validates_identity() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaCrossfadeNode::loadParameters")[1].split(
        "void FaCrossfadeNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::string>("input_a_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_b_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_a_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("input_b_stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<int>("crossfade.overlap_frames");' in load_parameters
    assert 'this->declare_parameter<std::string>("crossfade.curve");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredString(*this, \"input_a_topic\")" in load_parameters
    assert "readRequiredString(*this, \"input_a_stream_id\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredInt(*this, \"crossfade.overlap_frames\")" in load_parameters
    assert "backends::fadeCurveFromName(config_.curve)" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "resolve_topic_name(config_.input_a_topic)" in load_parameters
    assert "resolved_input_a == resolved_input_b" in load_parameters
    assert "streamMatchesTopic(" in load_parameters
    assert "config_.input_a_stream_id == config_.input_b_stream_id" in load_parameters
    assert "config_.overlap_frames <= 0" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "rclcpp::SystemDefaultsQoS" not in source


def test_node_pairs_adjacent_segments_by_epoch_and_metadata_before_backend() -> None:
    source = read_node_source()
    try_publish = source.split("void FaCrossfadeNode::tryPublishPairLocked")[1].split(
        "void FaCrossfadeNode::clearPendingLocked"
    )[0]
    validate_frame = source.split("bool FaCrossfadeNode::validateFrame")[1].split(
        "bool FaCrossfadeNode::pairMetadataMatches"
    )[0]

    assert "pending_a_.epoch != pending_b_.epoch" in try_publish
    assert "pairMetadataMatches(pending_a_, pending_b_)" in try_publish
    assert "backend_->process(pending_a_.data, pending_b_.data, output_data)" in try_publish
    assert try_publish.index("pending_a_.epoch != pending_b_.epoch") < try_publish.index(
        "backend_->process(pending_a_.data, pending_b_.data, output_data)"
    )
    assert "msg.stream_id != expected_stream_id" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytesPerFrame()) != 0" in validate_frame
    assert "segment_a.source_id == segment_b.source_id" in source


def test_backend_is_ros_free_and_owns_sample_crossfade() -> None:
    backend_header = read_backend_header()
    backend_source = read_backend_source()

    forbidden = ("rclcpp", "fa_interfaces", "AudioFrame", "diagnostic_msgs", "topic")
    for text in (backend_header, backend_source):
        for token in forbidden:
            assert token not in text
    assert "enum class FadeCurve" in backend_header
    assert "FadeCurve::kLinear" in backend_source
    assert "FadeCurve::kEqualPower" in backend_source
    assert "a_gain = 1.0 - position;" in backend_source
    assert "b_gain = position;" in backend_source
    assert "std::cos(position * kHalfPi)" in backend_source
    assert "std::sin(position * kHalfPi)" in backend_source
    assert "output = std::move(candidate);" in backend_source


def test_backend_fails_closed_without_clamp_or_hidden_limiter() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalCrossfadeBackend::process")[1].split(
        "FadeCurve fadeCurveFromName"
    )[0]

    assert "std::clamp" not in backend_source
    assert "limiter" not in backend_source
    assert "normalize(" not in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kOutOfRangeInput" in backend_source
    assert "ProcessStatus::kNonFiniteOutput" in process
    assert "ProcessStatus::kOutOfRangeOutput" in process
    assert "throw std::logic_error(\"unhandled crossfade fade curve\")" in backend_source
    assert "throw std::logic_error(\"unhandled crossfade backend process status\")" in backend_source
    assert "throw std::logic_error(\"crossfade.curve must be one of: linear, equal_power\")" in backend_source


def test_colcon_runs_pytest_and_backend_gtest_contracts() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
