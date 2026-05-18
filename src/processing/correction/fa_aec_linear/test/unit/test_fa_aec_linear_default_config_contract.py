from pathlib import Path

import yaml


def test_default_config_drops_reference_failures() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config_text = config_path.read_text(encoding="utf-8")

    params = config["fa_aec_linear"]["ros__parameters"]

    assert params["reference_failure_policy"] == "drop"
    assert params["expected_channels"] == 1
    assert params["expected"]["encoding"] == "PCM16LE"
    assert params["expected"]["bit_depth"] == 16
    assert "-1" not in config_text


def test_aec_linear_outputs_audio_frame_identity_without_analysis_metadata() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_aec_linear_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    assert "msg.source_id.empty()" in source
    assert "msg.stream_id != expected_stream_id" in source
    assert "fa_aec_linear received a null reference AudioFrame pointer" in source
    assert "fa_aec_linear received a null mic AudioFrame pointer" in source
    assert "fa_aec_linear publisher is not initialized" in source
    assert "validateFrame(*msg, config_.ref_topic)" in source
    assert "validateFrame(*msg, config_.mic_topic)" in source
    assert "msg.layout != kInterleavedLayout" in source
    assert "out_msg.source_id = msg->source_id;" in source
    assert "out_msg.stream_id = config_.output_topic;" in source
    assert "out_msg.layout = kInterleavedLayout;" in source
    assert "computeRmsPeak" not in source
    assert ".rms" not in source
    assert ".peak" not in source
    assert ".vad" not in source


def test_aec_linear_rejects_ambiguous_format_and_hidden_clamp() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_aec_linear_node.cpp").read_text(
        encoding="utf-8"
    )
    header = (
        package_root
        / "include"
        / "fa_aec_linear"
        / "fa_aec_linear_node.hpp"
    ).read_text(encoding="utf-8")
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package_root / "docs" / "テスト設計.md").read_text(
        encoding="utf-8"
    )

    assert "isSupportedAudioFormatPair" in source
    assert "PCM16LE" in source
    assert "FLOAT32LE" in source
    assert "PCM32LE" not in source
    assert "msg.encoding != config_.expected_encoding" in source
    assert "msg.bit_depth != static_cast<uint32_t>(config_.expected_bit_depth)" in source
    assert "ref->encoding != msg->encoding" in source
    assert "mic_f32.size() != ref_f32.size()" in source
    assert "expected encoding/bit_depth must be PCM16LE/16 or FLOAT32LE/32" in source
    assert "AEC linear output sample out of normalized range" in source
    assert "Add an explicit dynamics/limiter node if range control is required" in source
    assert "std::clamp" not in source
    assert "static bool encodeFromFloat" in header
    assert "std::string & error_message" in header
    assert "hidden range clamp" in spec
    assert "PCM32LE/32" in algorithm
    assert "clamp せず frame を drop" in algorithm
    assert "output range overflow は clamp せず drop" in test_plan


def test_aec_linear_rejects_channel_wildcards_and_documents_stream_binding() -> None:
    package_root = Path(__file__).parents[2]
    source = (package_root / "src" / "fa_aec_linear_node.cpp").read_text(
        encoding="utf-8"
    )
    spec = (package_root / "docs" / "仕様書.md").read_text(encoding="utf-8")
    algorithm = (package_root / "docs" / "アルゴリズム詳細説明書.md").read_text(
        encoding="utf-8"
    )
    test_plan = (package_root / "docs" / "テスト設計.md").read_text(
        encoding="utf-8"
    )

    assert "config_.expected_channels <= 0" in source
    assert "config_.expected_channels > 0" not in source
    assert "msg.channels != static_cast<uint32_t>(config_.expected_channels)" in source
    assert "channel 検査の無効化は禁止" in spec
    assert "mic frame の `stream_id` は `mic_topic`" in spec
    assert "reference frame の `stream_id` は `ref_topic`" in spec
    assert "channel 検査の wildcard はない" in algorithm
    assert "stream_id` が `mic_topic`" in test_plan
    assert "stream_id` が `ref_topic`" in test_plan


def test_main_shutdown_is_guarded_by_rclcpp_ok_and_manifest_declares_launch_deps() -> None:
    package_root = Path(__file__).parents[2]
    main_source = (package_root / "src" / "main.cpp").read_text(encoding="utf-8")
    package_xml = (package_root / "package.xml").read_text(encoding="utf-8")

    assert "if (rclcpp::ok()) {\n      rclcpp::shutdown();" in main_source
    assert "}\n  if (rclcpp::ok()) {\n    rclcpp::shutdown();" in main_source
    assert "<exec_depend>launch</exec_depend>" in package_xml
    assert "<exec_depend>launch_ros</exec_depend>" in package_xml
    assert "<test_depend>ament_lint_auto</test_depend>" in package_xml
    assert "<test_depend>python3-yaml</test_depend>" in package_xml
