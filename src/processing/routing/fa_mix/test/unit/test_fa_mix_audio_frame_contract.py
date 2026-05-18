from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_mix_node.cpp").read_text(encoding="utf-8")


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_mix"
        / "backends"
        / "internal_pcm16_mixer.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_pcm16_mixer.cpp").read_text(
        encoding="utf-8"
    )


def test_example_config_separates_topics_from_stream_ids() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_mix"]["ros__parameters"]

    assert params["input_topics"] == ["fa_mix/tts"]
    assert params["input_stream_ids"] == ["audio/tts/frame"]
    assert params["output_topic"] == "fa_mix/output"
    assert params["output"]["stream_id"] == "audio/mix/output"
    assert params["input_topics"][0] != params["input_stream_ids"][0]
    assert params["output_topic"] != params["output"]["stream_id"]
    assert params["input_stream_ids"][0] != params["output"]["stream_id"]
    assert params["input_gains_db"] == [0.0]
    assert params["master_index"] == 0
    assert params["expected"]["sample_rate"] == 48000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["bit_depth"] == 16
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["layout"] == "interleaved"
    assert params["max_frame_age_ms"] == 60
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is True
    assert params["diagnostics"]["qos"]["depth"] == 10
    assert params["diagnostics"]["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_launch_requires_explicit_config_file_and_node_name_without_package_default() -> None:
    launch_source = (package_root() / "launch" / "fa_mix.launch.py").read_text(encoding="utf-8")
    node_name_argument = launch_source.split('DeclareLaunchArgument(\n            "node_name"')[1].split(
        "        ),",
        1,
    )[0]
    config_argument = launch_source.split('DeclareLaunchArgument(\n            "config_file"')[1].split(
        "        ),",
        1,
    )[0]

    assert "default_value" not in node_name_argument
    assert "default_value" not in config_argument
    assert "FindPackageShare" not in launch_source
    assert "PathJoinSubstitution" not in launch_source
    assert "config/default.yaml" not in launch_source
    assert "parameters=[config_file]" in launch_source


def test_node_requires_parameters_without_runtime_defaults_and_validates_identity() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaMixNode::loadParameters")[1].split(
        "void FaMixNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<std::vector<std::string>>("input_topics");' in load_parameters
    assert 'this->declare_parameter<std::vector<std::string>>("input_stream_ids");' in load_parameters
    assert 'this->declare_parameter<std::string>("output_topic");' in load_parameters
    assert 'this->declare_parameter<std::string>("output.stream_id");' in load_parameters
    assert 'this->declare_parameter<std::string>("expected.layout");' in load_parameters
    assert 'this->declare_parameter<int>("diagnostics.qos.depth");' in load_parameters
    assert 'this->declare_parameter<bool>("diagnostics.qos.reliable");' in load_parameters
    assert "readRequiredStringArray(*this, \"input_topics\")" in load_parameters
    assert "readRequiredStringArray(*this, \"input_stream_ids\")" in load_parameters
    assert "readRequiredString(*this, \"output.stream_id\")" in load_parameters
    assert "readRequiredDoubleArray(*this, \"input_gains_db\")" in load_parameters
    assert "readRequiredBool(*this, \"diagnostics.qos.reliable\")" in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line
    assert "input_stream_ids must match input_topics length" in load_parameters
    assert "resolve_topic_name(topic)" in load_parameters
    assert "ensureUniqueIdentities(raw_topics" in load_parameters
    assert "ensureUniqueIdentities(stream_ids" in load_parameters
    assert "ensureStreamDoesNotMatchTopic(" in load_parameters
    assert "config_.expected_sample_rate > kMaxExpectedSampleRate" in load_parameters
    assert "config_.expected_channels > kMaxExpectedChannels" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "rclcpp::SystemDefaultsQoS" not in source
    assert "std::max<int>(1, config_.qos_depth)" not in source


def test_node_validates_frame_contract_before_backend() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaMixNode::validateFrame")[1].split(
        "void FaMixNode::onInputFrame"
    )[0]
    on_input = source.split("void FaMixNode::onInputFrame")[1].split(
        "void FaMixNode::mixAndPublish"
    )[0]
    mix_and_publish = source.split("void FaMixNode::mixAndPublish")[1].split(
        "void FaMixNode::publishDiagnostics"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != expected_stream_id" in validate_frame
    assert "hasValidFrameStamp(msg)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty()" not in validate_frame
    assert "bytesPerFrame()" not in validate_frame
    assert "validateFrame(*msg, config_.input_stream_ids[index])" in on_input
    assert "backend_->mix(input_data, out_bytes)" in mix_and_publish
    assert "input_data.push_back(frame->data);" in mix_and_publish


def test_backend_is_ros_free_and_owns_pcm16_mixing() -> None:
    node_source = read_node_source()
    backend_header = read_backend_header()
    backend_source = read_backend_source()

    forbidden = ("rclcpp", "fa_interfaces", "AudioFrame", "diagnostic_msgs", "topic")
    for text in (backend_header, backend_source):
        for token in forbidden:
            assert token not in text
    assert "decodePcm16ToFloat" not in node_source
    assert "encodeFloatToPcm16" not in node_source
    assert "std::vector<std::vector<uint8_t>>" in backend_header
    assert "readPcm16Le" in backend_source
    assert "writePcm16Le" in backend_source
    assert "mixed[sample_index] +=" in backend_source


def test_backend_drops_overflow_without_clamp_and_commits_only_on_success() -> None:
    backend_source = read_backend_source()
    mix_code = backend_source.split("MixResult InternalPcm16MixerBackend::mix")[1].split(
        "double dbToLinear"
    )[0]
    encode_code = backend_source.split("MixStatus InternalPcm16MixerBackend::encodePcm16Le")[1].split(
        "MixResult InternalPcm16MixerBackend::mix"
    )[0]

    assert "std::clamp" not in backend_source
    assert "MixStatus::kSampleCountMismatch" in mix_code
    assert "MixStatus::kOutOfRangeOutput" in encode_code
    assert "MixStatus::kPcm16RangeOutput" in encode_code
    assert "output = std::move(candidate);" in encode_code
    assert "output = std::move(candidate);" in backend_source
    assert "last_sample_count_ = expected_sample_count;" in mix_code
    assert mix_code.index("output = std::move(candidate);") < mix_code.index(
        "last_sample_count_ = expected_sample_count;"
    )
    assert "throw std::logic_error(\"unhandled PCM16 mixer backend status\")" in backend_source


def test_node_output_metadata_and_diagnostics_use_backend_state() -> None:
    source = read_node_source()
    mix_and_publish = source.split("void FaMixNode::mixAndPublish")[1].split(
        "void FaMixNode::publishDiagnostics"
    )[0]
    diagnostics = source.split("void FaMixNode::publishDiagnostics")[1].split(
        "}  // namespace fa_mix"
    )[0]

    assert "out.header = base.header;" in mix_and_publish
    assert "out.source_id = base.source_id;" in mix_and_publish
    assert "out.stream_id = config_.output_stream_id;" in mix_and_publish
    assert "out.layout = config_.expected_layout;" in mix_and_publish
    assert "out.data = std::move(out_bytes);" in mix_and_publish
    assert "backend_->inputCount()" in diagnostics
    assert "backend_->lastSampleCount()" in diagnostics
    assert "output_stream_id" in diagnostics
    assert "diagnostics_qos_depth" in diagnostics


def test_package_layout_and_colcon_test_contracts() -> None:
    required_paths = (
        "docs/backends/internal_pcm16_mixer.md",
        "include/fa_mix/backends/internal_pcm16_mixer.hpp",
        "src/backends/internal_pcm16_mixer.cpp",
        "test/cpp/test_internal_pcm16_mixer_backend.cpp",
    )
    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()

    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "add_library(fa_mix_internal_pcm16_mixer STATIC" in cmake_text
    assert "target_link_libraries(fa_mix_node_core" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
