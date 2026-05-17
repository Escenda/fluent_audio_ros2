from pathlib import Path

import yaml


def test_default_config_requires_explicit_source_identifier() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_in_node"]["ros__parameters"]
    selector = params["audio"]["device_selector"]

    assert params["backend"]["name"] == "alsa_capture"
    assert selector["mode"] == "name"
    assert selector["identifier"] == ""


def test_source_backend_has_no_struct_default() -> None:
    header_path = Path(__file__).parents[2] / "include" / "fa_in" / "fa_in_node.hpp"

    assert "std::string backend_name{};" in header_path.read_text(encoding="utf-8")
