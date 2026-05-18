from pathlib import Path

import yaml


def test_default_config_requires_float32_interleaved_contract() -> None:
    package_root = Path(__file__).parents[2]
    config = yaml.safe_load((package_root / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_gain"]["ros__parameters"]

    assert params["input_topic"] == "audio/resample16k/mic"
    assert params["output_topic"] == "audio/gain/mic"
    assert params["gain"]["linear"] == 1.0
    assert params["expected"]["sample_rate"] == 16000
    assert params["expected"]["channels"] == 1
    assert params["expected"]["encoding"] == "FLOAT32LE"
    assert params["expected"]["bit_depth"] == 32
    assert params["expected"]["layout"] == "interleaved"


def test_gain_does_not_hide_other_processing_responsibilities() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_gain_node.cpp").read_text(encoding="utf-8")

    forbidden = (
        "std::clamp",
        "clip",
        "normalize(",
        "resample",
        "SND_PCM",
        "snd_pcm",
        "set_channels",
    )
    for token in forbidden:
        assert token not in source


def test_gain_validates_frame_contract_before_processing() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_gain_node.cpp").read_text(encoding="utf-8")
    validate_frame = source.split("bool FaGainNode::validateFrame")[1].split(
        "bool FaGainNode::applyGain"
    )[0]

    assert "msg.source_id.empty() || msg.stream_id.empty()" in validate_frame
    assert "msg.stream_id != config_.input_topic" in validate_frame
    assert "msg.layout != config_.expected_layout" in validate_frame
    assert "msg.encoding != config_.expected_encoding" in validate_frame
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in validate_frame
    assert "msg.sample_rate != static_cast<uint32_t>(config_.expected_sample_rate)" in validate_frame
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in validate_frame
    assert "msg.data.empty() || (msg.data.size() % bytes_per_frame) != 0" in validate_frame


def test_gain_preserves_source_identity_and_updates_stream_identity() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_gain_node.cpp").read_text(encoding="utf-8")
    apply_gain = source.split("bool FaGainNode::applyGain")[1].split(
        "void FaGainNode::publishDiagnostics"
    )[0]

    assert "out = in;" in apply_gain
    assert "out.stream_id = config_.output_topic;" in apply_gain
    assert ".rms" not in apply_gain
    assert ".peak" not in apply_gain
    assert ".vad" not in apply_gain


def test_gain_drops_out_of_range_samples_instead_of_limiting() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_gain_node.cpp").read_text(encoding="utf-8")
    apply_gain = source.split("bool FaGainNode::applyGain")[1].split(
        "void FaGainNode::publishDiagnostics"
    )[0]

    assert "sample < kMinNormalizedSample || sample > kMaxNormalizedSample" in apply_gain
    assert "gained < kMinNormalizedSample || gained > kMaxNormalizedSample" in apply_gain
    assert "return false;" in apply_gain
    assert "std::clamp" not in apply_gain


def test_colcon_runs_pytest_contracts() -> None:
    package_root = Path(__file__).parents[2]
    cmake_text = (package_root / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
