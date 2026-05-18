from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_defines_explicit_input_topic() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_record"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"


def test_launch_requires_explicit_node_name_and_config_file() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_record.launch.py").read_text(
        encoding="utf-8"
    )

    assert 'DeclareLaunchArgument(\n            "node_name"' in launch_text
    assert 'DeclareLaunchArgument(\n            "config_file"' in launch_text
    assert 'package="fa_record"' in launch_text
    assert 'executable="fa_record_node"' in launch_text
    assert "parameters=[config_file]" in launch_text
    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text


def test_input_topic_has_no_runtime_default() -> None:
    source = (PACKAGE_ROOT / "src" / "fa_record_node.cpp").read_text(
        encoding="utf-8"
    )

    assert 'declare_parameter<std::string>("input_topic")' in source
    assert 'declare_parameter<std::string>("input_topic", input_topic_)' not in source
    assert "std::string input_topic_{};" in source
    assert 'std::string input_topic_{"audio/frame"};' not in source
    assert '"input_topic is required"' in source


def test_recorder_drops_invalid_frames_without_format_conversion() -> None:
    source = (PACKAGE_ROOT / "src" / "fa_record_node.cpp").read_text(
        encoding="utf-8"
    )

    assert "isSupportedFrame" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert 'msg.encoding == "PCM16LE"' in source
    assert 'msg.encoding == "FLOAT32LE"' in source
    assert "Frame format changed during recording; dropping frame" in source
    assert "resample" not in source
    assert "normalize" not in source
    assert "gain" not in source.lower()


def test_file_writer_backend_is_ros_free() -> None:
    backend_header = (
        PACKAGE_ROOT / "include" / "fa_record" / "backends" / "file_writer_backend.hpp"
    )
    backend_source = PACKAGE_ROOT / "src" / "backends" / "file_writer_backend.cpp"
    forbidden_tokens = (
        "rclcpp",
        "fa_interfaces",
        "diagnostic_msgs",
        "std_msgs/msg",
    )

    assert backend_header.is_file()
    assert backend_source.is_file()
    for path in (backend_header, backend_source):
        source = path.read_text(encoding="utf-8")
        for token in forbidden_tokens:
            assert token not in source


def test_file_writer_backend_contract_is_explicit() -> None:
    header = (
        PACKAGE_ROOT / "include" / "fa_record" / "backends" / "file_writer_backend.hpp"
    ).read_text(encoding="utf-8")
    source = (
        PACKAGE_ROOT / "src" / "backends" / "file_writer_backend.cpp"
    ).read_text(encoding="utf-8")

    assert "class FileWriterBackend" in header
    assert "class WavFileWriterBackend final" in header
    assert "record parent directory does not exist" in source
    assert (
        "recording AudioFormat encoding must be PCM16LE/16-bit or FLOAT32LE/32-bit"
        in source
    )
    assert "recording format changed during active file" in source
    assert "recording data exceeds WAV uint32 data length" in source


def test_cmake_links_record_node_to_file_writer_backend() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")

    assert "add_library(fa_record_backends STATIC" in cmake_text
    assert "src/backends/file_writer_backend.cpp" in cmake_text
    assert "target_link_libraries(fa_record_node" in cmake_text
    assert "fa_record_backends" in cmake_text
    assert "install(DIRECTORY include/" in cmake_text


def test_colcon_runs_pytest_contracts() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
