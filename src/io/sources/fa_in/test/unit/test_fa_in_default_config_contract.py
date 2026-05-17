from pathlib import Path

import yaml


def test_default_config_requires_explicit_source_identifier() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_in_node"]["ros__parameters"]
    selector = params["audio"]["device_selector"]

    assert params["backend"]["name"] == "alsa_capture"
    assert params["audio"]["encoding"] == "PCM16LE"
    assert params["audio"]["bit_depth"] == 16
    assert selector["mode"] == "name"
    assert selector["identifier"] == ""
    assert '"default"' not in config_text


def test_source_backend_has_no_struct_default() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_in" / "fa_in_node.hpp"

    assert "std::string backend_name{};" in header_path.read_text(encoding="utf-8")


def test_alsa_backend_filters_plugin_pcm_sources() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "isRawAlsaHardwareSource" in source
    assert 'rfind("hw:", 0)' in source
    assert "devices.emplace_back(source_id" in source


def test_alsa_backend_validates_format_contract_and_disables_resampling() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "alsaFormatForConfig" in source
    assert '"audio.encoding/audio.bit_depth must be one of PCM16LE/16, PCM32LE/32, FLOAT32LE/32"' in source
    assert "snd_pcm_hw_params_set_rate_resample(pcm_handle_, params, 0)" in source


def test_runtime_read_failure_fails_closed_without_prepare_retry() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_in_node.cpp"
    source = source_path.read_text(encoding="utf-8")
    capture_loop = source.split("void FaInNode::captureLoop()")[1].split(
        "void FaInNode::publishFrame"
    )[0]

    assert "failClosed(" in capture_loop
    assert "snd_pcm_prepare" not in capture_loop
    assert "std::this_thread::sleep_for" not in capture_loop
    assert "rclcpp::shutdown()" in source


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
