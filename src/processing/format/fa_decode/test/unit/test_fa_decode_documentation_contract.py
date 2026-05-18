from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_decode_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "external_codec_decoder.md").is_file()


def test_fa_decode_is_declared_as_ros_package_after_contract_completion() -> None:
    assert (PACKAGE_ROOT / "package.xml").is_file()
    assert (PACKAGE_ROOT / "CMakeLists.txt").is_file()


def test_fa_decode_default_config_requires_explicit_external_backend() -> None:
    config = yaml.safe_load((PACKAGE_ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    params = config["fa_decode"]["ros__parameters"]

    assert params["backend.name"] == "external_codec_decoder"
    assert params["backend.command.executable"] == ""
    assert params["backend.command.arguments"] == []
    assert params["backend.command.timeout_ms"] > 0
    assert params["backend.command.max_output_bytes"] > 0
    assert params["input_topic"] == "audio/encoded/mic"
    assert params["output_topic"] == "audio/pcm16/mic"
    assert params["input"]["codec"] == "opus"
    assert params["input"]["container"] == "ogg"
    assert params["input"]["payload_format"] == "ogg_page"
    assert params["output"]["encoding"] == "PCM16LE"
    assert params["output"]["bit_depth"] == 16
    assert params["output"]["layout"] == "interleaved"


def test_backend_is_ros_free_and_node_owns_message_conversion() -> None:
    backend_header = (
        PACKAGE_ROOT / "include" / "fa_decode" / "backends" / "external_codec_decoder.hpp"
    ).read_text(encoding="utf-8")
    backend_source = (
        PACKAGE_ROOT / "src" / "backends" / "external_codec_decoder.cpp"
    ).read_text(encoding="utf-8")
    node_source = (PACKAGE_ROOT / "src" / "fa_decode_node.cpp").read_text(encoding="utf-8")

    for token in ("rclcpp", "fa_interfaces", "AudioFrame", "EncodedAudioChunk"):
        assert token not in backend_header
        assert token not in backend_source

    assert "fa_interfaces::msg::EncodedAudioChunk" in node_source
    assert "fa_interfaces::msg::AudioFrame" in node_source
    assert "out.stream_id = config_.output_topic;" in node_source
    assert "out.encoding = result.encoding;" in node_source
    assert "out.epoch = in.epoch;" in node_source


def test_launch_and_node_require_explicit_runtime_config() -> None:
    launch_text = (PACKAGE_ROOT / "launch" / "fa_decode.launch.py").read_text(
        encoding="utf-8"
    )
    node_source = (PACKAGE_ROOT / "src" / "fa_decode_node.cpp").read_text(encoding="utf-8")
    load_parameters = node_source.split("void FaDecodeNode::loadParameters")[1].split(
        "void FaDecodeNode::setupBackend"
    )[0]

    assert "default_value" not in launch_text
    assert "FindPackageShare" not in launch_text
    assert "PathJoinSubstitution" not in launch_text
    assert 'description="設定ファイルへのパス。必ず明示する。"' in launch_text
    assert 'readRequiredString(*this, "backend.name")' in load_parameters
    assert 'readRequiredStringArray(*this, "backend.command.arguments")' in load_parameters
    assert 'readRequiredInt(*this, "output.sample_rate")' in load_parameters
    assert 'readRequiredBool(*this, "qos.reliable")' in load_parameters
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in load_parameters
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in load_parameters
    for line in load_parameters.splitlines():
        if "declare_parameter" in line:
            assert ", config_." not in line


def test_external_decoder_docs_do_not_claim_runtime_metadata_probe() -> None:
    algorithm_doc = (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    backend_doc = (
        PACKAGE_ROOT / "docs" / "backends" / "external_codec_decoder.md"
    ).read_text(encoding="utf-8")
    combined_doc = algorithm_doc + "\n" + backend_doc

    assert "actual format metadata" not in combined_doc
    assert "stdout は byte payload のみ" in combined_doc
    assert "stdout から推定しない" in combined_doc
    assert "output contract" in combined_doc


def test_colcon_runs_pytest_gtest_and_graph_contracts() -> None:
    cmake_text = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    package_xml = (PACKAGE_ROOT / "package.xml").read_text(encoding="utf-8")

    assert "find_package(ament_cmake_gtest REQUIRED)" in cmake_text
    assert "find_package(ament_cmake_pytest REQUIRED)" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_backend_test" in cmake_text
    assert "ament_add_gtest(${PROJECT_NAME}_graph_smoke_test" in cmake_text
    assert "ament_add_pytest_test(${PROJECT_NAME}_pytest test" in cmake_text
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in cmake_text
    assert "<test_depend>ament_cmake_gtest</test_depend>" in package_xml
    assert "<test_depend>ament_cmake_pytest</test_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-pytest</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
