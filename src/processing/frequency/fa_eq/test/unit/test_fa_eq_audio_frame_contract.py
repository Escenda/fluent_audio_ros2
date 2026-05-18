from pathlib import Path

import yaml


def package_root() -> Path:
    return Path(__file__).parents[2]


def read_source() -> str:
    return (package_root() / "src" / "fa_eq_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_three_band_eq_contract() -> None:
    config = yaml.safe_load((package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_eq"]["ros__parameters"]

    assert params["input_topic"] == "audio/sample_format/mic"
    assert params["output_topic"] == "audio/eq/mic"
    assert params["low"]["cutoff_hz"] == 250.0
    assert params["high"]["cutoff_hz"] == 4000.0
    assert 0.0 < params["low"]["cutoff_hz"] < params["high"]["cutoff_hz"]
    assert params["high"]["cutoff_hz"] < params["expected"]["sample_rate"] / 2.0
    assert params["gains"]["low_db"] == 0.0
    assert params["gains"]["mid_db"] == 0.0
    assert params["gains"]["high_db"] == 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_eq_does_not_hide_other_processing_or_io_responsibilities() -> None:
    sources = [
        read_source(),
        (package_root() / "src" / "backends" / "internal_three_band_eq.cpp").read_text(
            encoding="utf-8"
        ),
    ]

    forbidden = (
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
        "threshold.linear",
        "std::clamp",
        "limiter",
        "denoise",
    )
    for source in sources:
        for token in forbidden:
            assert token not in source


def test_eq_validates_startup_config_fail_closed() -> None:
    source = read_source()
    load_parameters = source.split("void FaEqNode::loadParameters")[1].split(
        "void FaEqNode::configureBackend"
    )[0]

    assert 'this->declare_parameter<double>("low.cutoff_hz", config_.low_cutoff_hz);' in load_parameters
    assert 'this->declare_parameter<double>("high.cutoff_hz", config_.high_cutoff_hz);' in load_parameters
    assert 'this->declare_parameter<double>("gains.low_db", config_.gain_low_db);' in load_parameters
    assert 'this->declare_parameter<double>("gains.mid_db", config_.gain_mid_db);' in load_parameters
    assert 'this->declare_parameter<double>("gains.high_db", config_.gain_high_db);' in load_parameters
    assert 'throw std::runtime_error("input_topic is required");' in load_parameters
    assert 'throw std::runtime_error("output_topic is required");' in load_parameters
    assert "const double nyquist_hz = static_cast<double>(config_.expected_sample_rate) / 2.0;" in load_parameters
    assert "!isFinite(config_.low_cutoff_hz)" in load_parameters
    assert "config_.low_cutoff_hz <= 0.0" in load_parameters
    assert "!isFinite(config_.high_cutoff_hz)" in load_parameters
    assert "config_.high_cutoff_hz <= config_.low_cutoff_hz" in load_parameters
    assert "config_.high_cutoff_hz >= nyquist_hz" in load_parameters
    assert "high.cutoff_hz must be finite, > low.cutoff_hz, and < expected.sample_rate / 2.0" in load_parameters
    assert "!isFinite(config_.gain_low_db)" in load_parameters
    assert "!isFinite(config_.gain_mid_db)" in load_parameters
    assert "!isFinite(config_.gain_high_db)" in load_parameters
    assert "gains.*_db must be finite" in load_parameters
    assert "fa_eq requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_eq requires expected.bit_depth=32" in load_parameters
    assert "fa_eq requires expected.layout=interleaved" in load_parameters


def test_eq_validates_frame_contract_before_processing() -> None:
    source = read_source()
    validate_frame = source.split("bool FaEqNode::validateFrame")[1].split(
        "bool FaEqNode::applyEq"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.source_id != active_source_id_" not in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_eq_preserves_source_identity_and_updates_stream_identity() -> None:
    source = read_source()
    apply_eq = source.split("bool FaEqNode::applyEq")[1].split(
        "void FaEqNode::publishDiagnostics"
    )[0]

    assert "active_source_id_ = in.source_id;" in apply_eq
    assert "out = in;" in apply_eq
    assert "out.stream_id = config_.output_topic;" in apply_eq
    assert ".rms" not in apply_eq
    assert ".peak" not in apply_eq
    assert ".vad" not in apply_eq


def test_eq_binds_source_only_after_backend_accepts_frame() -> None:
    source = read_source()
    apply_eq = source.split("bool FaEqNode::applyEq")[1].split(
        "void FaEqNode::publishDiagnostics"
    )[0]

    assert "const bool source_changed = !active_source_id_.empty() && in.source_id != active_source_id_;" in apply_eq
    assert (
        "const backends::ProcessStatus status = backend_->process(in.data, out.data, source_changed);"
        in apply_eq
    )
    assert "if (status != backends::ProcessStatus::kOk)" in apply_eq
    assert apply_eq.index("backend_->process") < apply_eq.index(
        "if (status != backends::ProcessStatus::kOk)"
    )
    assert apply_eq.index("if (status != backends::ProcessStatus::kOk)") < apply_eq.index(
        "active_source_id_ = in.source_id;"
    )


def test_eq_uses_first_order_splits_mid_residual_and_db_gains_per_channel() -> None:
    header = (
        package_root() / "include" / "fa_eq" / "backends" / "internal_three_band_eq.hpp"
    ).read_text(encoding="utf-8")
    source = (package_root() / "src" / "backends" / "internal_three_band_eq.cpp").read_text(
        encoding="utf-8"
    )
    process = source.split("ProcessStatus InternalThreeBandEqBackend::process")[1].split(
        "const char * processStatusMessage"
    )[0]

    assert "struct ChannelFilterState" in header
    assert "float previous_low_output" in header
    assert "float previous_hp_input" in header
    assert "float previous_hp_output" in header
    assert "std::vector<ChannelFilterState> channel_states_" in header
    assert "low_alpha_ = dt / (low_rc + dt);" in source
    assert "high_alpha_ = high_rc / (high_rc + dt);" in source
    assert "return std::pow(10.0, gain_db / 20.0);" in source
    assert "gain_low_linear_ = dbToLinear(config_.gain_low_db);" in source
    assert "gain_mid_linear_ = dbToLinear(config_.gain_mid_db);" in source
    assert "gain_high_linear_ = dbToLinear(config_.gain_high_db);" in source
    assert "reset_state ?" in process
    assert "channel_states_;" in process
    assert "ChannelFilterState & state =" in process
    assert "next_channel_states.at(i % static_cast<size_t>(config_.channels));" in process
    assert "low_sample = sample;" in process
    assert "high_sample = 0.0F;" in process
    assert "static_cast<double>(state.previous_low_output) +" in process
    assert "low_alpha_ * (static_cast<double>(sample) -" in process
    assert "high_alpha_ * (static_cast<double>(state.previous_hp_output) +" in process
    assert "static_cast<double>(sample) - static_cast<double>(state.previous_hp_input)" in process
    assert "static_cast<double>(sample) - static_cast<double>(low_sample) -" in process
    assert "static_cast<double>(low_sample) * gain_low_linear_" in process
    assert "mid_sample * gain_mid_linear_" in process
    assert "static_cast<double>(high_sample) * gain_high_linear_" in process
    assert "state.previous_low_output = low_sample;" in process
    assert "state.previous_hp_input = sample;" in process
    assert "state.previous_hp_output = high_sample;" in process
    assert "state.initialized = true;" in process
    assert "channel_states_ = std::move(next_channel_states);" in process


def test_eq_rejects_non_finite_or_out_of_range_samples_without_clamping() -> None:
    source = (package_root() / "src" / "backends" / "internal_three_band_eq.cpp").read_text(
        encoding="utf-8"
    )
    header = (
        package_root() / "include" / "fa_eq" / "backends" / "internal_three_band_eq.hpp"
    ).read_text(encoding="utf-8")

    assert "bool isNormalizedSample(float value)" in source
    assert "value >= kNormalizedMin && value <= kNormalizedMax" in source
    assert "kNonFiniteInput" in header
    assert "kOutOfRangeInput" in header
    assert "kNonFiniteOutput" in header
    assert "kOutOfRangeOutput" in header
    assert "return ProcessStatus::kOutOfRangeInput;" in source
    assert "return ProcessStatus::kOutOfRangeOutput;" in source
    assert "std::clamp" not in source


def test_eq_resets_state_on_source_change() -> None:
    source = read_source()
    apply_eq = source.split("bool FaEqNode::applyEq")[1].split(
        "void FaEqNode::publishDiagnostics"
    )[0]

    assert "const bool source_changed = !active_source_id_.empty() && in.source_id != active_source_id_;" in apply_eq
    assert "backend_->process(in.data, out.data, source_changed)" in apply_eq
    assert "source_resets_.fetch_add(1);" in apply_eq


def test_eq_diagnostics_include_filter_gain_state_and_counters() -> None:
    source = read_source()
    diagnostics = source.split("void FaEqNode::publishDiagnostics")[1]

    assert 'status.name = "fa_eq";' in diagnostics
    assert 'pushKeyValue(status, "low_cutoff_hz", std::to_string(config_.low_cutoff_hz));' in diagnostics
    assert 'pushKeyValue(status, "high_cutoff_hz", std::to_string(config_.high_cutoff_hz));' in diagnostics
    assert 'pushKeyValue(status, "low_alpha", std::to_string(backend_->lowAlpha()));' in diagnostics
    assert 'pushKeyValue(status, "high_alpha", std::to_string(backend_->highAlpha()));' in diagnostics
    assert 'pushKeyValue(status, "gain_low_db", std::to_string(config_.gain_low_db));' in diagnostics
    assert 'pushKeyValue(status, "gain_mid_db", std::to_string(config_.gain_mid_db));' in diagnostics
    assert 'pushKeyValue(status, "gain_high_db", std::to_string(config_.gain_high_db));' in diagnostics
    assert 'pushKeyValue(status, "state_source_id", active_source_id_);' in diagnostics
    assert 'pushKeyValue(status, "source_resets", std::to_string(source_resets_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_in", std::to_string(frames_in_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_out", std::to_string(frames_out_.load()));' in diagnostics
    assert 'pushKeyValue(status, "frames_dropped", std::to_string(frames_dropped_.load()));' in diagnostics


def test_package_layout_matches_standard_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_three_band_eq.md",
        "config/default.yaml",
        "launch/fa_eq.launch.py",
        "include/fa_eq/fa_eq_node.hpp",
        "include/fa_eq/backends/internal_three_band_eq.hpp",
        "src/fa_eq_node.cpp",
        "src/backends/internal_three_band_eq.cpp",
        "test/cpp/test_internal_three_band_eq_backend.cpp",
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
