from pathlib import Path

import pytest
import yaml

from fa_stream_py.backends.network_streamer import (
    AudioStreamFormat,
    _validate_audio_format,
)


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


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_stream.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'package="fa_stream"' in launch_text
    assert 'executable="fa_stream_node.py"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


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
    assert "Parameter.Type.NOT_SET" in source
    assert "Parameter.Type.STRING" in source
    assert "must be a string parameter" in source
    assert "get_parameter(name).get_parameter_value().string_value" not in source
    assert "NetworkStreamerBackend" in source


def test_streamer_rejects_frame_contract_mismatch_without_conversion() -> None:
    source = (PACKAGE_ROOT / "scripts" / "fa_stream_node.py").read_text(
        encoding="utf-8"
    )

    assert 'msg.layout != "interleaved"' in source
    assert 'msg.encoding != "PCM16LE"' in source
    assert "msg.bit_depth != 16" in source
    backend_source = (
        PACKAGE_ROOT
        / "fa_stream_py"
        / "backends"
        / "network_streamer.py"
    ).read_text(encoding="utf-8")

    assert "Audio stream format changed during stream" in backend_source
    assert 'audio_format.encoding != "PCM16LE"' in backend_source
    assert '"s16le"' in backend_source
    assert "resample" not in source
    assert "normalize" not in source
    assert "gain" not in source.lower()


def test_network_streamer_backend_rejects_non_pcm16le_encoding() -> None:
    with pytest.raises(
        RuntimeError,
        match="Only PCM16LE audio stream encoding is supported: PCM16BE",
    ):
        _validate_audio_format(
            AudioStreamFormat(
                sample_rate=48000,
                channels=1,
                encoding="PCM16BE",
                bit_depth=16,
                layout="interleaved",
            )
        )


def test_network_streamer_backend_accepts_pcm16le_s16le_contract() -> None:
    _validate_audio_format(
        AudioStreamFormat(
            sample_rate=48000,
            channels=1,
            encoding="PCM16LE",
            bit_depth=16,
            layout="interleaved",
        )
    )


def test_codec_settings_are_sink_packaging_not_general_format_conversion() -> None:
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    source = (PACKAGE_ROOT / "scripts" / "fa_stream_node.py").read_text(
        encoding="utf-8"
    )
    backend_source = (
        PACKAGE_ROOT
        / "fa_stream_py"
        / "backends"
        / "network_streamer.py"
    ).read_text(encoding="utf-8")

    assert "network endpoint protocol packaging" in spec
    assert "汎用的な encode/decode" in spec
    assert "src/processing/format" in spec
    assert "_audio_codec" in source
    assert "config.audio_codec" in backend_source


def test_network_streamer_backend_is_ros_free() -> None:
    backend_source = (
        PACKAGE_ROOT
        / "fa_stream_py"
        / "backends"
        / "network_streamer.py"
    ).read_text(encoding="utf-8")
    node_source = (PACKAGE_ROOT / "scripts" / "fa_stream_node.py").read_text(
        encoding="utf-8"
    )

    assert "import rclpy" not in backend_source
    assert "fa_interfaces" not in backend_source
    assert "AudioFrame" not in backend_source
    assert "from fa_stream_py.backends.network_streamer import" in node_source
    assert "subprocess.Popen" not in node_source
    assert "signal.SIGINT" not in node_source
    assert "shutil.which" in backend_source


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_python REQUIRED)" in cmake_text
    assert "ament_python_install_package(fa_stream_py PACKAGE_DIR fa_stream_py)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<buildtool_depend>ament_cmake_python</buildtool_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
