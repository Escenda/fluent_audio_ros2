from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_defines_explicit_input_topic() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_record"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"


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
    assert "Frame format changed during recording; dropping frame" in source
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
