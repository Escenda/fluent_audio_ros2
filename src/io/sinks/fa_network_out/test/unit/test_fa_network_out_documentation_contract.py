from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def test_fa_network_out_has_standard_design_documents() -> None:
    assert (PACKAGE_ROOT / "README.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "仕様書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "アルゴリズム詳細説明書.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "テスト設計.md").is_file()
    assert (PACKAGE_ROOT / "docs" / "backends" / "network_pcm_sender.md").is_file()


def test_fa_network_out_is_declared_as_ros_package_after_contract_completion() -> None:
    assert (PACKAGE_ROOT / "package.xml").is_file()
    assert (PACKAGE_ROOT / "CMakeLists.txt").is_file()
    assert (PACKAGE_ROOT / "launch" / "fa_network_out.launch.py").is_file()
    assert (PACKAGE_ROOT / "config" / "default.yaml").is_file()


def test_fa_network_out_docs_keep_network_sink_adapter_boundary() -> None:
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_doc = (
        PACKAGE_ROOT / "docs" / "backends" / "network_pcm_sender.md"
    ).read_text(encoding="utf-8")

    assert "jitter buffer" in spec
    assert "codec encode" in spec
    assert "endpoint 推測" in spec
    assert "does not resolve DNS names" in backend_doc


def test_fa_network_out_runtime_parameters_are_required_not_defaulted() -> None:
    source = (PACKAGE_ROOT / "src" / "fa_network_out_node.cpp").read_text(
        encoding="utf-8"
    )

    assert 'declare_parameter("input_topic", config_.input_topic)' not in source
    assert 'declare_parameter<int>("qos.depth", config_.qos_depth)' not in source
    assert 'declare_parameter<bool>("qos.reliable", config_.qos_reliable)' not in source
    assert "std::max<int>" not in source
    assert "SystemDefaultsQoS" not in source
    assert 'readRequiredString(*this, "input_topic")' in source
    assert 'readRequiredBool(*this, "qos.reliable")' in source
    assert 'readRequiredInt(*this, "diagnostics.qos.depth")' in source
    assert 'readRequiredBool(*this, "diagnostics.qos.reliable")' in source
