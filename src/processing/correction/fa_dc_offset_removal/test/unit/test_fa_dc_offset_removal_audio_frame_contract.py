from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_dc_offset_removal_node.cpp").read_text(
        encoding="utf-8"
    )


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_dc_offset_removal"
        / "backends"
        / "internal_frame_mean.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (package_root() / "src" / "backends" / "internal_frame_mean.cpp").read_text(
        encoding="utf-8"
    )


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_dc_offset_removal"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/dc_offset_removed/mic"
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_dc_offset_removal_does_not_hide_unrelated_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "convertPcm",
        "gain.linear",
        "threshold.linear",
        "filter.",
        "cutoff_hz",
        "center_hz",
        "std::clamp",
        "normalize(",
        ".rms",
        ".peak",
        ".vad",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    node_source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = node_source.split("void FaDcOffsetRemovalNode::loadParameters")[1].split(
        "void FaDcOffsetRemovalNode::configureBackend"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "config_.resolved_input_topic =" in load_parameters
    assert "resolve_topic_name(config_.input_topic)" in load_parameters
    assert "resolve_topic_name(config_.output_topic)" in load_parameters
    assert "config_.resolved_input_topic == config_.resolved_output_topic" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "config_.channels <= 0" in backend_source


def test_qos_depth_is_not_clamped_after_startup_validation() -> None:
    node_source = read_node_source()
    setup_interfaces = node_source.split("void FaDcOffsetRemovalNode::setupInterfaces")[1].split(
        "void FaDcOffsetRemovalNode::handleFrame"
    )[0]

    assert "config_.qos_depth <= 0" in node_source
    assert "rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));" in setup_interfaces
    assert "std::max<int>(1, config_.qos_depth)" not in node_source


def test_dc_offset_removal_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    handle_frame = source.split("void FaDcOffsetRemovalNode::handleFrame")[1].split(
        "bool FaDcOffsetRemovalNode::validateFrame"
    )[0]
    validate_frame = source.split("bool FaDcOffsetRemovalNode::validateFrame")[1].split(
        "bool FaDcOffsetRemovalNode::removeDcOffset"
    )[0]

    assert "if (!msg)" in handle_frame
    assert 'throw std::logic_error("received null AudioFrame pointer")' in handle_frame
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


def test_dc_offset_removal_preserves_identity_and_updates_stream_identity() -> None:
    source = read_node_source()
    remove_dc_offset = source.split("bool FaDcOffsetRemovalNode::removeDcOffset")[1].split(
        "void FaDcOffsetRemovalNode::publishDiagnostics"
    )[0]

    assert "out = in;" in remove_dc_offset
    assert "out.stream_id = config_.output_topic;" in remove_dc_offset
    assert "backend_->process(in.data, out.data)" in remove_dc_offset
    assert "out.encoding =" not in remove_dc_offset
    assert "out.bit_depth =" not in remove_dc_offset
    assert "out.sample_rate =" not in remove_dc_offset
    assert "out.channels =" not in remove_dc_offset
    assert "out.layout =" not in remove_dc_offset


def test_dc_offset_algorithm_is_backend_owned() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalFrameMeanBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "class InternalFrameMeanBackend" in header
    assert "enum class ProcessStatus" in header
    assert "const size_t channel_count = static_cast<size_t>(config_.channels);" in process
    assert "const size_t sample_count = input.size() / sizeof(float);" in process
    assert "const size_t frame_count = sample_count / channel_count;" in process
    assert "std::vector<double> channel_sums(channel_count, 0.0);" in process
    assert "std::vector<float> samples(sample_count, 0.0F);" in process
    assert "channel_sums.at(i % channel_count) += static_cast<double>(sample);" in process
    assert "std::vector<double> channel_means(channel_count, 0.0);" in process
    assert "channel_sums.at(channel) / static_cast<double>(frame_count)" in process
    assert "static_cast<double>(samples.at(i)) - channel_means.at(i % channel_count)" in process
    assert "std::memcpy(next_output.data() + (i * sizeof(float)), &out_sample, sizeof(float));" in process
    assert "channel_sums" not in node_source
    assert "channel_means" not in node_source
    assert "std::memcpy" not in node_source


def test_float32le_native_memcpy_is_little_endian_only() -> None:
    backend_source = read_backend_source()

    assert "#if !defined(__BYTE_ORDER__)" in backend_source
    assert "__ORDER_LITTLE_ENDIAN__" in backend_source
    assert (
        '#error "fa_dc_offset_removal internal_frame_mean requires a little-endian target '
        'for FLOAT32LE"'
    ) in backend_source


def test_dc_offset_removal_drops_non_finite_input_and_output_samples() -> None:
    backend_source = read_backend_source()
    node_source = read_node_source()
    process = backend_source.split("ProcessResult InternalFrameMeanBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!std::isfinite(mean)" in process
    assert "!std::isfinite(corrected)" in process
    assert "!std::isfinite(out_sample)" in process
    assert "ProcessStatus::kNonFiniteInput" in process
    assert "ProcessStatus::kNonFiniteMean" in process
    assert "ProcessStatus::kNonFiniteOutput" in process
    assert "backends::processStatusMessage(result.status)" in node_source


def test_dc_offset_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    header = read_backend_header()
    backend_source = read_backend_source()

    assert "kEmptyInput" in header
    assert "kMisalignedInput" in header
    assert "kNonFiniteInput" in header
    assert "kNonFiniteMean" in header
    assert "kNonFiniteOutput" in header
    assert "processStatusMessage(ProcessStatus status)" in header
    assert "ProcessStatus::kMisalignedInput" in backend_source
    assert "ProcessStatus::kNonFiniteInput" in backend_source
    assert "ProcessStatus::kNonFiniteOutput" in backend_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in header
        assert token not in backend_source


def test_rejected_frame_does_not_overwrite_output() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalFrameMeanBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    rejection_section = process.split("output = std::move(next_output);")[0]
    assert "std::vector<uint8_t> next_output(input.size());" in rejection_section
    assert "ProcessStatus::kNonFiniteInput" in rejection_section
    assert "ProcessStatus::kNonFiniteOutput" in rejection_section
    assert "output = std::move(next_output);" not in rejection_section
    assert "output = std::move(next_output);" in process


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_frame_mean.md",
        "config/default.yaml",
        "launch/fa_dc_offset_removal.launch.py",
        "include/fa_dc_offset_removal/fa_dc_offset_removal_node.hpp",
        "include/fa_dc_offset_removal/backends/internal_frame_mean.hpp",
        "src/fa_dc_offset_removal_node.cpp",
        "src/backends/internal_frame_mean.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_frame_mean_backend.cpp",
        "test/cpp/test_dc_offset_removal_graph.cpp",
        "test/unit/test_fa_dc_offset_removal_audio_frame_contract.py",
        "test/launch/test_fa_dc_offset_removal_launch_contract.py",
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
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    graph_deps = cmake_text.split(
        "ament_target_dependencies(${PROJECT_NAME}_graph_smoke_test"
    )[1]
    assert "diagnostic_msgs" in graph_deps
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml


def test_diagnostics_publish_resolved_topic_identity() -> None:
    source = read_node_source()
    publish_diagnostics = source.split("void FaDcOffsetRemovalNode::publishDiagnostics")[1]

    assert 'pushKeyValue(status, "input_topic", config_.input_topic);' in publish_diagnostics
    assert 'pushKeyValue(status, "output_topic", config_.output_topic);' in publish_diagnostics
    assert (
        'pushKeyValue(status, "resolved_input_topic", config_.resolved_input_topic);'
        in publish_diagnostics
    )
    assert (
        'pushKeyValue(status, "resolved_output_topic", config_.resolved_output_topic);'
        in publish_diagnostics
    )
