from pathlib import Path

import yaml


def test_default_config_requires_explicit_backend() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_aec_nn"]["ros__parameters"]

    assert params["enabled"] is True
    assert params["backend"] == ""


def test_disabled_branch_drops_before_publish() -> None:
    source_path = Path(__file__).parents[2] / "src" / "fa_aec_nn_node.cpp"
    source = source_path.read_text(encoding="utf-8")

    disabled_index = source.index("if (!config_.enabled)")
    publish_index = source.index("pub_->publish(*msg)")

    assert disabled_index < publish_index
    assert "Dropping frame because fa_aec_nn is disabled" in source
