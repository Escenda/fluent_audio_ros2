from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_node_source() -> str:
    return (package_root() / "src" / "fa_hum_node.cpp").read_text(encoding="utf-8")


def read_node_header() -> str:
    return (package_root() / "include" / "fa_hum" / "fa_hum_node.hpp").read_text(
        encoding="utf-8"
    )


def read_backend_header() -> str:
    return (
        package_root()
        / "include"
        / "fa_hum"
        / "backends"
        / "internal_notch_cascade.hpp"
    ).read_text(encoding="utf-8")


def read_backend_source() -> str:
    return (
        package_root() / "src" / "backends" / "internal_notch_cascade.cpp"
    ).read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load(
        (package_root() / "config" / "default.yaml").read_text(encoding="utf-8")
    )
    params = config["fa_hum"]["ros__parameters"]

    assert params["input_topic"] == "audio/dc_offset_removed/mic"
    assert params["output_topic"] == "audio/hum_removed/mic"
    assert params["hum"]["frequency_hz"] == 60.0
    assert params["hum"]["harmonics"] == 4
    assert params["hum"]["q"] == 30.0
    assert 0.0 < params["hum"]["frequency_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["hum"]["harmonics"] >= 1
    assert params["hum"]["q"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_hum_removal_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [read_node_source(), read_backend_source()]

    forbidden = (
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "gain.linear",
        "threshold.linear",
        "std::clamp",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_startup_validation_fails_closed_for_invalid_config() -> None:
    source = read_node_source()
    backend_source = read_backend_source()
    load_parameters = source.split("void FaHumNode::loadParameters")[1].split(
        "void FaHumNode::configureBackend"
    )[0]

    assert "config_.input_topic.empty()" in load_parameters
    assert "config_.output_topic.empty()" in load_parameters
    assert "resolve_topic_name(config_.input_topic)" in load_parameters
    assert "resolve_topic_name(config_.output_topic)" in load_parameters
    assert "config_.resolved_input_topic == config_.resolved_output_topic" in load_parameters
    assert "config_.expected_sample_rate <= 0" in load_parameters
    assert "!isFinite(config_.frequency_hz)" in load_parameters
    assert "config_.frequency_hz <= 0.0" in load_parameters
    assert "config_.frequency_hz >= nyquist_hz" in load_parameters
    assert "config_.harmonics < 1" in load_parameters
    assert "!isFinite(config_.q)" in load_parameters
    assert "config_.q <= 0.0" in load_parameters
    assert "config_.expected_channels <= 0" in load_parameters
    assert "config_.expected_encoding != kEncodingFloat32" in load_parameters
    assert "config_.expected_bit_depth != 32" in load_parameters
    assert "config_.expected_layout != kInterleavedLayout" in load_parameters
    assert "config_.qos_depth <= 0" in load_parameters
    assert "config_.diagnostics_publish_period_ms <= 0" in load_parameters
    assert "throw std::runtime_error" in load_parameters
    assert "throw std::runtime_error" in backend_source


def test_hum_validates_frame_contract_before_processing() -> None:
    source = read_node_source()
    validate_frame = source.split("bool FaHumNode::validateFrame")[1].split(
        "bool FaHumNode::isStaleFrame"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "isStaleFrame(msg)" in validate_frame


def test_stateful_iir_stream_order_contract_is_explicit() -> None:
    source = read_node_source()
    header = read_node_header()
    backend_header = read_backend_header()
    backend_source = read_backend_source()

    assert "bool isStaleFrame(const fa_interfaces::msg::AudioFrame & msg) const;" in header
    assert "void rememberAcceptedFrame(const fa_interfaces::msg::AudioFrame & msg);" in header
    assert "last_source_id_" in header
    assert "last_epoch_" in header
    assert "last_stamp_sec_" in header
    assert "last_stamp_nanosec_" in header
    assert "msg.source_id != last_source_id_ || msg.epoch != last_epoch_" in source
    assert "msg.header.stamp.sec < last_stamp_sec_" in source
    assert "msg.header.stamp.nanosec < last_stamp_nanosec_" in source
    assert "rememberAcceptedFrame(*msg);" in source
    assert "uint32_t epoch" in backend_header
    assert "kStaleEpoch" in backend_header
    assert "epoch < active_epoch_" in backend_source
    assert "epoch > active_epoch_" in backend_source


def test_hum_preserves_metadata_and_updates_stream_identity_only() -> None:
    source = read_node_source()
    apply_hum = source.split("bool FaHumNode::applyHumRemoval")[1].split(
        "void FaHumNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_hum
    assert "out.stream_id = config_.output_topic;" in apply_hum
    assert "out.data = std::move(processed_data);" in apply_hum
    assert "out.encoding =" not in apply_hum
    assert "out.bit_depth =" not in apply_hum
    assert "out.sample_rate =" not in apply_hum
    assert "out.channels =" not in apply_hum
    assert "out.layout =" not in apply_hum
    assert ".rms" not in apply_hum
    assert ".peak" not in apply_hum
    assert ".vad" not in apply_hum


def test_hum_backend_owns_notch_biquad_cascade() -> None:
    backend_header = read_backend_header()
    backend_source = read_backend_source()
    node_source = read_node_source()
    configure_backend = node_source.split("void FaHumNode::configureBackend")[1].split(
        "void FaHumNode::setupInterfaces"
    )[0]

    assert "class InternalNotchCascadeBackend" in backend_header
    assert "enum class ProcessStatus" in backend_header
    assert "struct BiquadCoefficients" in backend_header
    assert "struct BiquadState" in backend_header
    assert "for (int harmonic = 1; harmonic <= config_.harmonics; ++harmonic)" in backend_source
    assert "const double center_hz = config_.frequency_hz * static_cast<double>(harmonic);" in backend_source
    assert "const double alpha = std::sin(omega) / (2.0 * config_.q);" in backend_source
    assert "coefficients.b0 = 1.0 / a0;" in backend_source
    assert "coefficients.b1 = (-2.0 * cos_omega) / a0;" in backend_source
    assert "coefficients.b2 = 1.0 / a0;" in backend_source
    assert "coefficients.a1 = (-2.0 * cos_omega) / a0;" in backend_source
    assert "coefficients.a2 = (1.0 - alpha) / a0;" in backend_source
    assert "channel_states_ = std::move(next_states);" in backend_source
    assert "std::make_unique<backends::InternalNotchCascadeBackend>" in configure_backend
    assert "BiquadCoefficients" not in node_source
    assert "std::memcpy" not in node_source


def test_backend_rejects_bad_samples_without_clamping_or_output_overwrite() -> None:
    backend_source = read_backend_source()
    process = backend_source.split("ProcessResult InternalNotchCascadeBackend::process")[1].split(
        "size_t InternalNotchCascadeBackend::stageCount"
    )[0]

    assert "!std::isfinite(sample)" in process
    assert "!isNormalized(static_cast<double>(sample))" in process
    assert "!isFinite(stage_output)" in process
    assert "!isNormalized(filtered)" in process
    assert "!std::isfinite(out_sample)" in process
    assert "!isNormalized(static_cast<double>(out_sample))" in process
    assert "std::clamp" not in process
    rejection_section = process.split("output = std::move(next_output);")[0]
    assert "output = std::move(next_output);" not in rejection_section
    assert "output = std::move(next_output);" in process


def test_backend_reports_reason_and_keeps_ros_boundary() -> None:
    backend_header = read_backend_header()
    backend_source = read_backend_source()

    assert "kEmptySourceId" in backend_header
    assert "kMisalignedInput" in backend_header
    assert "kStaleEpoch" in backend_header
    assert "kOutOfRangeInput" in backend_header
    assert "kOutOfRangeOutput" in backend_header
    assert "processStatusMessage(ProcessStatus status)" in backend_header
    assert "ProcessStatus::kStaleEpoch" in backend_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in backend_header
        assert token not in backend_source


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_notch_cascade.md",
        "config/default.yaml",
        "launch/fa_hum.launch.py",
        "include/fa_hum/fa_hum_node.hpp",
        "include/fa_hum/backends/internal_notch_cascade.hpp",
        "src/fa_hum_node.cpp",
        "src/backends/internal_notch_cascade.cpp",
        "src/main.cpp",
        "test/cpp/test_internal_notch_cascade_backend.cpp",
        "test/cpp/test_fa_hum_graph.cpp",
        "test/unit/test_fa_hum_audio_frame_contract.py",
        "test/launch/test_fa_hum_launch_contract.py",
        "test/integration/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_and_gtest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "add_library(fa_hum_internal_notch_cascade STATIC" in cmake_text
    assert "add_library(fa_hum_node_core" in cmake_text
    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
