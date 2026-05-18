from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_high_pass"]["ros__parameters"]

    assert params["input_topic"] == "audio/resample16k/mic"
    assert params["output_topic"] == "audio/high_pass/mic"
    assert params["filter"]["cutoff_hz"] == 80.0
    assert 0.0 < params["filter"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_high_pass_does_not_hide_other_processing_or_io_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    sources = [
        (package_root / "src" / "fa_high_pass_node.cpp").read_text(encoding="utf-8"),
        (package_root / "src" / "backends" / "internal_high_pass.cpp").read_text(
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


def test_high_pass_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_high_pass_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaHighPassNode::validateFrame")[1].split(
        "bool FaHighPassNode::applyHighPass"
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


def test_high_pass_preserves_source_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_high_pass_node.cpp").read_text(encoding="utf-8")
    apply_high_pass = source.split("bool FaHighPassNode::applyHighPass")[1].split(
        "void FaHighPassNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_high_pass
    assert "out = in;" in apply_high_pass
    assert "out.stream_id = config_.output_topic;" in apply_high_pass
    assert ".rms" not in apply_high_pass
    assert ".peak" not in apply_high_pass
    assert ".vad" not in apply_high_pass


def test_high_pass_binds_source_only_after_backend_accepts_frame() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_high_pass_node.cpp").read_text(encoding="utf-8")
    apply_high_pass = source.split("bool FaHighPassNode::applyHighPass")[1].split(
        "void FaHighPassNode::publishDiagnostics"
    )[0]

    assert "const bool should_bind_source = active_source_id_.empty();" in apply_high_pass
    assert "const backends::ProcessStatus status = backend_->process(in.data, out.data);" in apply_high_pass
    assert "if (status != backends::ProcessStatus::kOk)" in apply_high_pass
    assert "if (should_bind_source)" in apply_high_pass
    assert apply_high_pass.index("backend_->process") < apply_high_pass.index(
        "if (status != backends::ProcessStatus::kOk)"
    )
    assert apply_high_pass.index("if (status != backends::ProcessStatus::kOk)") < apply_high_pass.index(
        "active_source_id_ = in.source_id;"
    )


def test_high_pass_uses_first_order_per_channel_state() -> None:
    package_root = Path(__file__).parents[2]
    header = (
        package_root / "include" / "fa_high_pass" / "backends" / "internal_high_pass.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root / "src" / "backends" / "internal_high_pass.cpp").read_text(
        encoding="utf-8"
    )
    process = source.split("ProcessStatus InternalHighPassBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "struct ChannelFilterState" in header
    assert "float previous_input" in header
    assert "float previous_output" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "alpha_ = rc_sec / (rc_sec + sample_interval_sec);" in source
    assert "std::vector<ChannelFilterState> next_channel_states = channel_states_;" in process
    assert "ChannelFilterState & state =" in process
    assert "next_channel_states.at(i % static_cast<size_t>(config_.channels));" in process
    assert "state.previous_output" in process
    assert "state.previous_input" in process
    assert "alpha_ * (static_cast<double>(state.previous_output) +" in process
    assert "static_cast<double>(sample) - static_cast<double>(state.previous_input))" in process
    assert "state.previous_input = sample;" in process
    assert "state.previous_output = out_sample;" in process
    assert "channel_states_ = std::move(next_channel_states);" in process


def test_high_pass_backend_reports_rejection_reason_and_keeps_ros_boundary() -> None:
    package_root = Path(__file__).parents[2]
    backend_header = (
        package_root / "include" / "fa_high_pass" / "backends" / "internal_high_pass.hpp"
    ).read_text(encoding="utf-8")
    backend_source = (
        package_root / "src" / "backends" / "internal_high_pass.cpp"
    ).read_text(encoding="utf-8")
    node_source = (package_root / "src" / "fa_high_pass_node.cpp").read_text(encoding="utf-8")

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


def test_cutoff_parameter_is_required_and_range_checked() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_high_pass_node.cpp").read_text(encoding="utf-8")
    load_parameters = source.split("void FaHighPassNode::loadParameters")[1].split(
        "void FaHighPassNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<double>("filter.cutoff_hz", config_.cutoff_hz);' in load_parameters
    assert "const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;" in load_parameters
    assert "!isFinite(config_.cutoff_hz)" in load_parameters
    assert "config_.cutoff_hz <= 0.0" in load_parameters
    assert "config_.cutoff_hz >= nyquist_hz" in load_parameters
    assert "filter.cutoff_hz must be finite, > 0.0, and < expected.sample_rate / 2.0" in load_parameters


def test_package_layout_matches_standard_processing_layout() -> None:
    package_root = Path(__file__).parents[2]
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_high_pass.md",
        "config/default.yaml",
        "launch/fa_high_pass.launch.py",
        "include/fa_high_pass/fa_high_pass_node.hpp",
        "src/fa_high_pass_node.cpp",
        "test/cpp/test_internal_high_pass_backend.cpp",
        "test/unit",
        "test/integration",
        "test/launch",
        "test/fixtures",
    )

    for relative_path in required_paths:
        assert (package_root / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
