from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_declick_node.cpp").read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_impulse_declick.cpp"
    ).read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_declick"]["ros__parameters"]

    assert params["input_topic"] == "audio/noise_gated/mic"
    assert params["output_topic"] == "audio/declicked/mic"
    assert params["threshold"]["delta"] == 0.25
    assert 0.0 < params["threshold"]["delta"] <= 2.0
    assert params["window"]["max_samples"] == 1
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_declick_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = read_node_source() + read_backend_source()

    forbidden = (
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "normalize(",
        "std::clamp",
        "gain.linear",
        "threshold.linear",
        "cutoff_hz",
        "center_hz",
        "denoise",
        "decrackle",
        "declip",
        "limiter",
        "reverb",
        "echo",
    )
    for token in forbidden:
        assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaDeclickNode::loadParameters")[1].split(
        "void FaDeclickNode::configureBackend"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "resolve_topic_name(config_.input_topic)" in load_parameters
    assert "resolve_topic_name(config_.output_topic)" in load_parameters
    assert "config_.resolved_input_topic == config_.resolved_output_topic" in load_parameters
    assert "!isFinite(config_.threshold_delta)" in load_parameters
    assert "config_.threshold_delta <= 0.0" in load_parameters
    assert "config_.threshold_delta > 2.0" in load_parameters
    assert "config_.window_max_samples <= 0" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "config_.diagnostics_qos_depth <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters


def test_required_parameters_are_declared_without_runtime_defaults() -> None:
    source = read_node_source()
    load_parameters = source.split("void FaDeclickNode::loadParameters")[1].split(
        "if (config_.input_topic.empty())"
    )[0]

    required_reads = (
        'readRequiredString(*this, "input_topic")',
        'readRequiredString(*this, "output_topic")',
        'readRequiredDouble(*this, "threshold.delta")',
        'readRequiredInt(*this, "window.max_samples")',
        'readRequiredInt(*this, "expected.sample_rate")',
        'readRequiredInt(*this, "expected.channels")',
        'readRequiredString(*this, "expected.encoding")',
        'readRequiredInt(*this, "expected.bit_depth")',
        'readRequiredString(*this, "expected.layout")',
        'readRequiredInt(*this, "qos.depth")',
        'readRequiredInt(*this, "diagnostics.qos.depth")',
        'readRequiredBool(*this, "qos.reliable")',
        'readRequiredBool(*this, "diagnostics.qos.reliable")',
    )
    for read in required_reads:
        assert read in load_parameters

    assert "readRequiredInt(" in load_parameters
    assert '"diagnostics.publish_period_ms"' in load_parameters
    assert "this->get_parameter(" not in load_parameters
    assert "SystemDefaultsQoS" not in source
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert "config_." not in line


def test_qos_uses_validated_depth_without_clamping_fallback() -> None:
    source = read_node_source()
    setup = source.split("void FaDeclickNode::setupInterfaces")[1].split(
        "void FaDeclickNode::handleFrame"
    )[0]

    assert "rclcpp::QoS qos(static_cast<size_t>(config_.qos_depth));" in setup
    assert "std::max" not in setup


def test_backend_is_ros_free_and_node_owns_ros_boundary() -> None:
    backend_header = (
        package_root() / "include" / "fa_declick" / "backends" / "internal_impulse_declick.hpp"
    ).read_text(encoding="utf-8")
    backend_source = read_backend_source()
    node_source = read_node_source()

    assert "InternalImpulseDeclickBackend" in backend_header
    assert "ProcessStatus" in backend_header
    assert "ProcessResult" in backend_header
    assert "fa_interfaces/msg/audio_frame" not in backend_header
    assert "rclcpp" not in backend_header
    assert "AudioFrame" not in backend_source
    assert "create_publisher" in node_source
    assert "create_subscription" in node_source
    assert "backend_->process(in.data, processed_data)" in node_source


def test_declick_validates_frame_contract_before_backend_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaDeclickNode::validateFrame")[1].split(
        "bool FaDeclickNode::applyDeclick"
    )[0]
    handle_frame = source.split("void FaDeclickNode::handleFrame")[1].split(
        "bool FaDeclickNode::validateFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "if (!validateFrame(*msg))" in handle_frame
    assert "if (!applyDeclick(*msg, out))" in handle_frame
    assert handle_frame.index("if (!validateFrame(*msg))") < handle_frame.index(
        "if (!applyDeclick(*msg, out))"
    )


def test_backend_rejects_invalid_samples_instead_of_clamping_or_normalizing() -> None:
    backend = read_backend_source()

    assert "ProcessStatus::kNonFiniteInput" in backend
    assert "ProcessStatus::kOutOfRangeInput" in backend
    assert "ProcessStatus::kNonFiniteOutput" in backend
    assert "ProcessStatus::kOutOfRangeOutput" in backend
    assert "std::clamp" not in backend
    assert "normalize(" not in backend


def test_declick_preserves_metadata_and_updates_stream_identity() -> None:
    source = read_node_source()
    apply_declick = source.split("bool FaDeclickNode::applyDeclick")[1].split(
        "void FaDeclickNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_declick
    assert "out.stream_id = config_.output_topic;" in apply_declick
    assert "out.data = std::move(processed_data);" in apply_declick
    assert "out.encoding =" not in apply_declick
    assert "out.bit_depth =" not in apply_declick
    assert "out.sample_rate =" not in apply_declick
    assert "out.channels =" not in apply_declick
    assert "out.layout =" not in apply_declick


def test_diagnostics_publish_config_counters_backend_and_resolved_topics() -> None:
    source = read_node_source()
    diagnostics = source.split("void FaDeclickNode::publishDiagnostics")[1].split(
        "}  // namespace fa_declick"
    )[0]

    assert 'status.name = "fa_declick";' in diagnostics
    assert '"threshold_delta"' in diagnostics
    assert '"window_max_samples"' in diagnostics
    assert '"resolved_input_topic"' in diagnostics
    assert '"resolved_output_topic"' in diagnostics
    assert '"expected_encoding"' in diagnostics
    assert '"expected_bit_depth"' in diagnostics
    assert '"expected_layout"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"samples_corrected"' in diagnostics
    assert '"click_runs_corrected"' in diagnostics
    assert '"backend.name"' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_impulse_declick.md",
        "config/default.yaml",
        "launch/fa_declick.launch.py",
        "include/fa_declick/backends/internal_impulse_declick.hpp",
        "include/fa_declick/fa_declick_node.hpp",
        "src/backends/internal_impulse_declick.cpp",
        "src/fa_declick_node.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_impulse_declick_backend.cpp",
        "test/cpp/test_fa_declick_graph.cpp",
        "test/unit/test_fa_declick_audio_frame_contract.py",
        "test/launch/test_fa_declick_launch_contract.py",
        "test/integration/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_cpp_and_pytest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
