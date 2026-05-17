from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_defines_required_stream_parameters() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_stream"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["ffmpeg_path"] == "ffmpeg"
    assert params["output_url"] == ""
    assert params["audio_codec"] == "libmp3lame"
    assert params["bitrate"] == "128k"
    assert params["container_format"] == "mp3"
    assert params["content_type"] == "audio/mpeg"
    assert params["loglevel"] == "warning"


def test_required_parameters_have_no_runtime_defaults() -> None:
    source = (PACKAGE_ROOT / "scripts" / "fa_stream_node.py").read_text(
        encoding="utf-8"
    )

    required_parameters = (
        "input_topic",
        "ffmpeg_path",
        "output_url",
        "audio_codec",
        "bitrate",
        "container_format",
        "content_type",
        "loglevel",
    )
    for parameter_name in required_parameters:
        assert f'self.declare_parameter("{parameter_name}")' in source
        assert f'self.declare_parameter("{parameter_name}",' not in source

    assert "_required_string_parameter" in source
    assert "is required" in source
    assert "shutil.which(self._ffmpeg_path)" in source


def test_streamer_rejects_frame_contract_mismatch_without_conversion() -> None:
    source = (PACKAGE_ROOT / "scripts" / "fa_stream_node.py").read_text(
        encoding="utf-8"
    )

    assert 'msg.layout != "interleaved"' in source
    assert "msg.bit_depth != 16" in source
    assert "Sample rate changed during stream" in source
    assert "Channel count changed during stream" in source
    assert '"s16le"' in source
    assert "resample" not in source
    assert "normalize" not in source
    assert "gain" not in source.lower()


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
