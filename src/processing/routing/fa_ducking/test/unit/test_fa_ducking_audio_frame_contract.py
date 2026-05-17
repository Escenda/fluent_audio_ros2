from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _source_text() -> str:
    return (_package_root() / "src" / "fa_ducking_node.cpp").read_text(encoding="utf-8")


def test_default_config_requires_float32_interleaved_contract() -> None:
    config = yaml.safe_load((_package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_ducking"]["ros__parameters"]

    assert params["program_topic"] == "audio/program/frame"
    assert params["sidechain_topic"] == "audio/sidechain/frame"
    assert params["output_topic"] == "audio/ducked/frame"
    assert params["sidechain"]["threshold_rms"] == 0.05
    assert params["sidechain"]["max_age_ms"] == 100
    assert params["ducking"]["gain_db"] == -12.0
    assert params["ducking"]["attack_ms"] == 10.0
    assert params["ducking"]["release_ms"] == 250.0
    assert 0.0 < params["sidechain"]["threshold_rms"] <= 1.0
    assert params["sidechain"]["max_age_ms"] > 0
    assert -96.0 <= params["ducking"]["gain_db"] < 0.0
    assert params["ducking"]["attack_ms"] > 0.0
    assert params["ducking"]["release_ms"] > 0.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000


def test_ducking_does_not_hide_other_processing_or_io_responsibilities() -> None:
    source = _source_text()

    forbidden = (
        "std::clamp",
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
        "resample",
        "set_channels",
        "convert",
        "applyLimiter",
        "applyCompressor",
        "applyNormalize",
        "applyNoiseGate",
        "low_pass",
        "high_pass",
        "denoise",
    )
    for token in forbidden:
        assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    source = _source_text()
    load_parameters = source.split("void FaDuckingNode::loadParameters")[1].split(
        "void FaDuckingNode::setupInterfaces"
    )[0]

    assert "throw std::runtime_error(\"program_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"sidechain_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"output_topic is required\")" in load_parameters
    assert "config_.sidechain_threshold_rms <= 0.0" in load_parameters
    assert "config_.sidechain_threshold_rms > 1.0" in load_parameters
    assert "sidechain.threshold_rms must be finite and in (0.0, 1.0]" in load_parameters
    assert "config_.sidechain_max_age_ms <= 0" in load_parameters
    assert "sidechain.max_age_ms must be > 0" in load_parameters
    assert "config_.ducking_gain_db < -96.0" in load_parameters
    assert "config_.ducking_gain_db >= 0.0" in load_parameters
    assert "ducking.gain_db must be finite and in [-96.0, 0.0)" in load_parameters
    assert "ducking.attack_ms must be finite and > 0.0" in load_parameters
    assert "ducking.release_ms must be finite and > 0.0" in load_parameters
    assert "fa_ducking requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_ducking requires expected.bit_depth=32" in load_parameters
    assert "fa_ducking requires expected.layout=interleaved" in load_parameters


def test_runtime_frame_validation_drops_invalid_program_and_sidechain_frames() -> None:
    source = _source_text()
    validate_frame = source.split("bool FaDuckingNode::validateFrame")[1].split(
        "bool FaDuckingNode::readSamples"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != expected_stream_id" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "return false;" in validate_frame

    handle_program = source.split("void FaDuckingNode::handleProgramFrame")[1].split(
        "void FaDuckingNode::handleSidechainFrame"
    )[0]
    handle_sidechain = source.split("void FaDuckingNode::handleSidechainFrame")[1].split(
        "bool FaDuckingNode::validateFrame"
    )[0]

    assert "if (!msg)" in handle_program
    assert "program_frames_dropped_.fetch_add(1);" in handle_program
    assert "if (!msg)" in handle_sidechain
    assert "sidechain_frames_dropped_.fetch_add(1);" in handle_sidechain
    assert "invalidateSidechainState();" in handle_sidechain
    assert "validateFrame(*msg, config_.program_topic, \"program\")" in source
    assert "validateFrame(*msg, config_.sidechain_topic, \"sidechain\")" in source


def test_invalid_sidechain_frames_invalidate_recent_state() -> None:
    source = _source_text()
    handle_sidechain = source.split("void FaDuckingNode::handleSidechainFrame")[1].split(
        "bool FaDuckingNode::validateFrame"
    )[0]
    invalidate = source.split("void FaDuckingNode::invalidateSidechainState")[1].split(
        "SidechainSnapshot FaDuckingNode::sidechainSnapshot"
    )[0]

    assert "invalidateSidechainState();" in handle_sidechain
    assert "has_sidechain_ = false;" in invalidate
    assert "latest_sidechain_rms_ = 0.0;" in invalidate
    assert "sidechain_state_invalidations_.fetch_add(1);" in invalidate


def test_sidechain_rms_state_is_used_only_as_control_signal() -> None:
    source = _source_text()
    handle_sidechain = source.split("void FaDuckingNode::handleSidechainFrame")[1].split(
        "bool FaDuckingNode::validateFrame"
    )[0]
    apply_ducking = source.split("bool FaDuckingNode::applyDucking")[1].split(
        "void FaDuckingNode::publishDiagnostics"
    )[0]

    assert "const double rms = calculateFrameRms(samples);" in handle_sidechain
    assert "latest_sidechain_rms_ = rms;" in handle_sidechain
    assert "latest_sidechain_received_at_ = this->now();" in handle_sidechain
    assert "out = in;" in apply_ducking
    assert "out.stream_id = config_.output_topic;" in apply_ducking
    assert "out.data.resize(in.data.size());" in apply_ducking
    output_loop = apply_ducking.split("for (size_t i = 0; i < samples.size(); ++i)")[1].split(
        "current_gain_.store(candidate_gain);"
    )[0]
    assert "sidechain" not in output_loop.lower()
    assert "latest_sidechain_rms_" not in apply_ducking


def test_sidechain_active_requires_recent_rms_at_or_above_threshold() -> None:
    source = _source_text()
    sidechain_active = source.split("bool FaDuckingNode::sidechainIsActive")[1].split(
        "double FaDuckingNode::smoothingAlpha"
    )[0]

    assert "if (!snapshot.available)" in sidechain_active
    assert "const int64_t age_ms = (now - snapshot.received_at).nanoseconds() / 1000000;" in sidechain_active
    assert "age_ms < 0 || age_ms > config_.sidechain_max_age_ms" in sidechain_active
    assert "snapshot.rms >= config_.sidechain_threshold_rms" in sidechain_active
    assert "last_sidechain_active_.store(active);" in sidechain_active


def test_ducking_gain_target_and_smoothing_are_explicit() -> None:
    source = _source_text()

    assert "double dbToLinear(double db)" in source
    assert "std::pow(10.0, db / 20.0)" in source
    assert "config_.ducking_gain_linear = dbToLinear(config_.ducking_gain_db);" in source
    assert "double FaDuckingNode::smoothingAlpha" in source
    assert "1.0 - std::exp(-frame_seconds / time_constant_seconds)" in source
    assert "double FaDuckingNode::smoothedGain" in source
    assert "target_gain < current_gain ? config_.attack_ms : config_.release_ms" in source
    assert "current_gain + (alpha * (target_gain - current_gain))" in source

    apply_ducking = source.split("bool FaDuckingNode::applyDucking")[1].split(
        "void FaDuckingNode::publishDiagnostics"
    )[0]
    assert "const bool active_sidechain = sidechainIsActive(this->now());" in apply_ducking
    assert "const double target_gain = active_sidechain ? config_.ducking_gain_linear : 1.0;" in apply_ducking
    assert "const double candidate_gain = smoothedGain(target_gain, samples.size());" in apply_ducking


def test_ducking_drops_non_finite_or_out_of_range_samples_instead_of_clamping() -> None:
    source = _source_text()
    read_samples = source.split("bool FaDuckingNode::readSamples")[1].split(
        "double FaDuckingNode::calculateFrameRms"
    )[0]
    apply_ducking = source.split("bool FaDuckingNode::applyDucking")[1].split(
        "void FaDuckingNode::publishDiagnostics"
    )[0]

    assert "!std::isfinite(sample)" in read_samples
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in read_samples
    assert "output < kMinNormalizedSample" in apply_ducking
    assert "output > kMaxNormalizedSample" in apply_ducking
    assert "Dropping program frame because ducking output is outside normalized FLOAT32LE range" in apply_ducking
    assert "return false;" in apply_ducking
    assert "std::clamp" not in source


def test_output_range_drop_does_not_commit_candidate_gain() -> None:
    source = _source_text()
    apply_ducking = source.split("bool FaDuckingNode::applyDucking")[1].split(
        "void FaDuckingNode::publishDiagnostics"
    )[0]

    overflow_section = apply_ducking.split(
        "Dropping program frame because ducking output is outside normalized FLOAT32LE range"
    )[0]
    assert "current_gain_.store(candidate_gain);" not in overflow_section
    assert "current_gain_.store(candidate_gain);" in apply_ducking
    assert "last_target_gain_.store(target_gain);" in apply_ducking


def test_diagnostics_include_parameters_state_and_counters() -> None:
    source = _source_text()
    diagnostics = source.split("void FaDuckingNode::publishDiagnostics")[1].split(
        "}  // namespace fa_ducking"
    )[0]

    assert 'status.name = "fa_ducking";' in diagnostics
    assert '"program_topic"' in diagnostics
    assert '"sidechain_topic"' in diagnostics
    assert '"output_topic"' in diagnostics
    assert '"sidechain_threshold_rms"' in diagnostics
    assert '"sidechain_max_age_ms"' in diagnostics
    assert '"ducking_gain_db"' in diagnostics
    assert '"ducking_gain_linear"' in diagnostics
    assert '"attack_ms"' in diagnostics
    assert '"release_ms"' in diagnostics
    assert '"current_gain"' in diagnostics
    assert '"last_target_gain"' in diagnostics
    assert '"last_sidechain_rms"' in diagnostics
    assert '"last_sidechain_age_ms"' in diagnostics
    assert '"last_sidechain_active"' in diagnostics
    assert '"program_frames_in"' in diagnostics
    assert '"program_frames_out"' in diagnostics
    assert '"program_frames_dropped"' in diagnostics
    assert '"sidechain_frames_in"' in diagnostics
    assert '"sidechain_frames_valid"' in diagnostics
    assert '"sidechain_frames_dropped"' in diagnostics
    assert '"sidechain_state_invalidations"' in diagnostics
    assert '"ducked_program_frames"' in diagnostics
    assert '"released_program_frames"' in diagnostics
    assert '"stale_sidechain_checks"' in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/internal_sidechain_ducking.md",
        "config/default.yaml",
        "launch/fa_ducking.launch.py",
        "include/fa_ducking/fa_ducking_node.hpp",
        "src/fa_ducking_node.cpp",
        "test/unit/test_fa_ducking_audio_frame_contract.py",
        "test/integration/.gitkeep",
        "test/launch/.gitkeep",
        "test/fixtures/.gitkeep",
    )

    for relative_path in required_paths:
        assert (_package_root() / relative_path).exists()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (_package_root() / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (_package_root() / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
