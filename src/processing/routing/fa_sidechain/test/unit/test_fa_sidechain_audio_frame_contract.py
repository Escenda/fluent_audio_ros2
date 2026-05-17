import math
import struct
from pathlib import Path

import yaml


def _package_root() -> Path:
    return Path(__file__).parents[2]


def _source_text() -> str:
    return (_package_root() / "src" / "fa_sidechain_node.cpp").read_text(encoding="utf-8")


def _header_text() -> str:
    return (_package_root() / "include" / "fa_sidechain" / "fa_sidechain_node.hpp").read_text(
        encoding="utf-8"
    )


def _config_params():
    config = yaml.safe_load((_package_root() / "config" / "default.yaml").read_text(encoding="utf-8"))
    return config["fa_sidechain"]["ros__parameters"]


def _db_to_linear(db: float) -> float:
    return math.pow(10.0, db / 20.0)


def _rms(samples: list[float]) -> float:
    return math.sqrt(sum(sample * sample for sample in samples) / len(samples))


def _gain_for_samples(samples: list[float], threshold: float, active_db: float, inactive_db: float) -> float:
    return _db_to_linear(active_db if _rms(samples) >= threshold else inactive_db)


def _encode_control_gain(gain: float) -> bytes:
    return struct.pack("<f", gain)


def test_default_config_requires_float32_interleaved_contract() -> None:
    params = _config_params()
    launch_text = (_package_root() / "launch" / "fa_sidechain.launch.py").read_text(encoding="utf-8")

    assert params["sidechain_topic"] == "audio/sidechain/frame"
    assert params["control_topic"] == "audio/sidechain/control"
    assert params["detector"]["threshold_rms"] == 0.05
    assert params["detector"]["active_gain_db"] == -12.0
    assert params["detector"]["inactive_gain_db"] == 0.0
    assert params["control"]["sample_rate"] == 1000
    assert 0.0 < params["detector"]["threshold_rms"] <= 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"
    assert params["qos"]["depth"] == 10
    assert params["qos"]["reliable"] is False
    assert params["diagnostics"]["publish_period_ms"] == 1000
    assert 'default_value="fa_sidechain"' in launch_text


def test_gain_math_publishes_active_or_inactive_linear_gain() -> None:
    params = _config_params()
    threshold = params["detector"]["threshold_rms"]
    active_db = params["detector"]["active_gain_db"]
    inactive_db = params["detector"]["inactive_gain_db"]

    assert math.isclose(_rms([0.03, -0.04]), 0.03535533905932738)
    assert math.isclose(_gain_for_samples([0.03, -0.04], threshold, active_db, inactive_db), 1.0)
    assert math.isclose(
        _gain_for_samples([0.05, -0.05], threshold, active_db, inactive_db),
        _db_to_linear(active_db),
    )
    assert _encode_control_gain(1.0) == b"\x00\x00\x80?"


def test_sidechain_does_not_modify_program_audio_or_hide_processing() -> None:
    source = _source_text()

    forbidden = (
        "program_topic",
        "output_topic",
        "applyDucking",
        "std::clamp",
        "resample",
        "normalize(",
        "applyNormalize",
        "applyLimiter",
        "applyCompressor",
        "applyNoiseGate",
        "SND_PCM",
        "snd_pcm",
        "Pa_OpenStream",
    )
    for token in forbidden:
        assert token not in source


def test_startup_config_validation_fails_closed() -> None:
    source = _source_text()
    header = _header_text()
    load_parameters = source.split("void FaSidechainNode::loadParameters")[1].split(
        "void FaSidechainNode::setupInterfaces"
    )[0]

    assert "#include <limits>" in header
    assert "active_gain_db{std::numeric_limits<double>::quiet_NaN()}" in header
    assert "inactive_gain_db{std::numeric_limits<double>::quiet_NaN()}" in header
    assert "throw std::runtime_error(\"sidechain_topic is required\")" in load_parameters
    assert "throw std::runtime_error(\"control_topic is required\")" in load_parameters
    assert "control_topic must differ from sidechain_topic" in load_parameters
    assert "config_.threshold_rms <= 0.0" in load_parameters
    assert "config_.threshold_rms > 1.0" in load_parameters
    assert "detector.threshold_rms must be finite and in (0.0, 1.0]" in load_parameters
    assert "detector.active_gain_db must be finite" in load_parameters
    assert "detector.inactive_gain_db must be finite" in load_parameters
    assert "active_gain_db must resolve to finite linear gain in [0.0, 4.0]" in load_parameters
    assert "inactive_gain_db must resolve to finite linear gain in [0.0, 4.0]" in load_parameters
    assert "control.sample_rate must be > 0" in load_parameters
    assert "fa_sidechain requires expected.encoding=FLOAT32LE" in load_parameters
    assert "fa_sidechain requires expected.bit_depth=32" in load_parameters
    assert "fa_sidechain requires expected.layout=interleaved" in load_parameters


def test_runtime_frame_validation_drops_invalid_sidechain_frames() -> None:
    source = _source_text()
    validate_frame = source.split("bool FaSidechainNode::validateFrame")[1].split(
        "bool FaSidechainNode::readSamples"
    )[0]
    handle_sidechain = source.split("void FaSidechainNode::handleSidechainFrame")[1].split(
        "bool FaSidechainNode::validateFrame"
    )[0]

    assert "msg.source_id.empty()" in validate_frame
    assert "msg.stream_id != config_.sidechain_topic" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame
    assert "if (!msg)" in handle_sidechain
    assert "frames_dropped_.fetch_add(1);" in handle_sidechain
    assert "validateFrame(*msg)" in handle_sidechain


def test_non_finite_or_out_of_range_samples_are_dropped_without_clamp() -> None:
    source = _source_text()
    read_samples = source.split("bool FaSidechainNode::readSamples")[1].split(
        "double FaSidechainNode::calculateFrameRms"
    )[0]

    assert "!std::isfinite(sample)" in read_samples
    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in read_samples
    assert "return false;" in read_samples
    assert "std::clamp" not in source


def test_rms_threshold_and_gain_control_frame_are_explicit() -> None:
    source = _source_text()
    rms_code = source.split("double FaSidechainNode::calculateFrameRms")[1].split(
        "double FaSidechainNode::targetGainForRms"
    )[0]
    target_code = source.split("double FaSidechainNode::targetGainForRms")[1].split(
        "bool FaSidechainNode::buildControlFrame"
    )[0]
    control_code = source.split("bool FaSidechainNode::buildControlFrame")[1].split(
        "void FaSidechainNode::publishDiagnostics"
    )[0]

    assert "square_sum += value * value;" in rms_code
    assert "std::sqrt(mean_square)" in rms_code
    assert "rms >= config_.threshold_rms ? config_.active_gain_linear : config_.inactive_gain_linear" in target_code
    assert "output.header = input.header;" in control_code
    assert "output.source_id = input.source_id;" in control_code
    assert "output.stream_id = config_.control_topic;" in control_code
    assert "output.sample_rate = static_cast<uint32_t>(config_.control_sample_rate);" in control_code
    assert "output.channels = 1;" in control_code
    assert "output.encoding = kEncodingFloat32;" in control_code
    assert "output.bit_depth = 32;" in control_code
    assert "output.layout = kInterleavedLayout;" in control_code
    assert "output.epoch = input.epoch;" in control_code
    assert "output.data.resize(sizeof(float));" in control_code
    assert "std::memcpy(output.data.data(), &gain_sample, sizeof(float));" in control_code


def test_output_gain_range_is_validated_before_publish() -> None:
    source = _source_text()
    control_code = source.split("bool FaSidechainNode::buildControlFrame")[1].split(
        "void FaSidechainNode::publishDiagnostics"
    )[0]

    assert "gain_linear < kMinControlGain" in control_code
    assert "gain_linear > kMaxControlGain" in control_code
    assert "gain_sample < static_cast<float>(kMinControlGain)" in control_code
    assert "gain_sample > static_cast<float>(kMaxControlGain)" in control_code
    assert "return false;" in control_code
    assert "control_pub_->publish(control_frame);" in source


def test_diagnostics_include_required_counters_and_last_values() -> None:
    source = _source_text()
    diagnostics = source.split("void FaSidechainNode::publishDiagnostics")[1].split(
        "}  // namespace fa_sidechain"
    )[0]

    assert 'status.name = "fa_sidechain";' in diagnostics
    assert '"sidechain_topic"' in diagnostics
    assert '"control_topic"' in diagnostics
    assert '"threshold_rms"' in diagnostics
    assert '"active_gain_db"' in diagnostics
    assert '"active_gain_linear"' in diagnostics
    assert '"inactive_gain_db"' in diagnostics
    assert '"inactive_gain_linear"' in diagnostics
    assert '"frames_in"' in diagnostics
    assert '"frames_out"' in diagnostics
    assert '"frames_dropped"' in diagnostics
    assert '"last_rms"' in diagnostics
    assert '"last_gain_linear"' in diagnostics
    assert '"active_frames"' in diagnostics
    assert '"inactive_frames"' in diagnostics


def test_package_layout_matches_required_processing_layout() -> None:
    required_paths = (
        "README.md",
        "docs/仕様書.md",
        "docs/アルゴリズム詳細説明書.md",
        "docs/テスト設計.md",
        "docs/backends/no_runtime_backend.md",
        "config/default.yaml",
        "launch/fa_sidechain.launch.py",
        "include/fa_sidechain/fa_sidechain_node.hpp",
        "src/fa_sidechain_node.cpp",
        "test/unit/test_fa_sidechain_audio_frame_contract.py",
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
