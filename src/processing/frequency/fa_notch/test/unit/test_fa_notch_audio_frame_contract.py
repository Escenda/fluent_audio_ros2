from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_notch_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_notch"]["ros__parameters"]

    assert params["input_topic"] == "audio/high_pass/mic"
    assert params["output_topic"] == "audio/notch/mic"
    assert params["filter"]["center_hz"] == 60.0
    assert params["filter"]["q"] == 30.0
    assert 0.0 < params["filter"]["center_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["filter"]["q"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_notch_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [
        read_source(),
        (package_root() / "src" / "backends" / "internal_notch.cpp").read_text(
            encoding="utf-8"
        ),
    ]

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


def test_notch_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaNotchNode::validateFrame")[1].split(
        "bool FaNotchNode::applyNotch"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.source_id != active_source_id_" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_notch_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_notch = source.split("bool FaNotchNode::applyNotch")[1].split(
        "void FaNotchNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_notch
    assert "out = in;" in apply_notch
    assert "out.stream_id = config_.output_topic;" in apply_notch
    assert ".rms" not in apply_notch
    assert ".peak" not in apply_notch
    assert ".vad" not in apply_notch


def test_notch_uses_second_order_biquad_per_channel_state() -> None:
    header = (
        package_root() / "include" / "fa_notch" / "backends" / "internal_notch.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root() / "src" / "backends" / "internal_notch.cpp").read_text(
        encoding="utf-8"
    )
    constructor = source.split("InternalNotchBackend::InternalNotchBackend")[1].split(
        "double InternalNotchBackend::centerHz"
    )[0]
    process = source.split("ProcessStatus InternalNotchBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "struct BiquadCoefficients" in header
    assert "struct ChannelFilterState" in header
    assert "double previous_input_1" in header
    assert "double previous_input_2" in header
    assert "double previous_output_1" in header
    assert "double previous_output_2" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "const double alpha = std::sin(omega) / (2.0 * config_.q);" in constructor
    assert "const double a0 = 1.0 + alpha;" in constructor
    assert "coefficients_.b0 = 1.0 / a0;" in constructor
    assert "coefficients_.b1 = (-2.0 * cos_omega) / a0;" in constructor
    assert "coefficients_.b2 = 1.0 / a0;" in constructor
    assert "coefficients_.a1 = (-2.0 * cos_omega) / a0;" in constructor
    assert "coefficients_.a2 = (1.0 - alpha) / a0;" in constructor
    assert "std::vector<ChannelFilterState> next_channel_states = channel_states_;" in process
    assert "ChannelFilterState & state =" in process
    assert "next_channel_states.at(i % static_cast<size_t>(config_.channels));" in process
    assert "coefficients_.b0 * input_sample +" in process
    assert "coefficients_.b1 * state.previous_input_1 +" in process
    assert "coefficients_.b2 * state.previous_input_2 -" in process
    assert "coefficients_.a1 * state.previous_output_1 -" in process
    assert "coefficients_.a2 * state.previous_output_2;" in process
    assert "channel_states_ = std::move(next_channel_states);" in process


def test_notch_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    backend_header = (
        package_root() / "include" / "fa_notch" / "backends" / "internal_notch.hpp"
    ).read_text(encoding="utf-8")
    backend_source = (package_root() / "src" / "backends" / "internal_notch.cpp").read_text(
        encoding="utf-8"
    )
    node_source = read_source()

    assert "enum class ProcessStatus" in backend_header
    assert "kEmptyInput" in backend_header
    assert "kMisalignedInput" in backend_header
    assert "kNonFiniteInput" in backend_header
    assert "kNonFiniteOutput" in backend_header
    assert "processStatusMessage(ProcessStatus status)" in backend_header
    assert "return ProcessStatus::kMisalignedInput;" in backend_source
    assert "std::numeric_limits<float>::max()" in backend_source
    assert "backends::processStatusMessage(status)" in node_source

    forbidden_backend_tokens = ("rclcpp", "fa_interfaces", "AudioFrame")
    for token in forbidden_backend_tokens:
        assert token not in backend_header
        assert token not in backend_source


def test_filter_parameters_are_required_and_range_checked() -> None:
    source = read_source()
    load_parameters = source.split("void FaNotchNode::loadParameters")[1].split(
        "void FaNotchNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<double>("filter.center_hz", config_.center_hz);' in load_parameters
    assert 'this->declare_parameter<double>("filter.q", config_.q);' in load_parameters
    assert "const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;" in load_parameters
    assert "!isFinite(config_.center_hz)" in load_parameters
    assert "config_.center_hz <= 0.0" in load_parameters
    assert "config_.center_hz >= nyquist_hz" in load_parameters
    assert "!isFinite(config_.q)" in load_parameters
    assert "config_.q <= 0.0" in load_parameters
    assert "filter.center_hz must be finite, > 0.0, and < expected.sample_rate / 2.0" in load_parameters
    assert "filter.q must be finite and > 0.0" in load_parameters


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_notch.md",
        "config/default.yaml",
        "launch/fa_notch.launch.py",
        "include/fa_notch/fa_notch_node.hpp",
        "include/fa_notch/backends/internal_notch.hpp",
        "src/fa_notch_node.cpp",
        "src/backends/internal_notch.cpp",
        "test/cpp/test_internal_notch_backend.cpp",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/fixtures",
    )

    for relative_path in required_paths:
        assert (package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
