from pathlib import Path

import yaml

from fa_tts_py.backends.factory import build_tts_backend
from fa_tts_py.backends.pyopenjtalk_backend import PyOpenJTalkBackend


def test_default_config_has_no_playback_or_gain_parameters() -> None:
    config_path = Path(__file__).parents[2] / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_tts"]["ros__parameters"]

    assert params["backend.name"] == "pyopenjtalk"
    assert params["backend.openjtalk_dict_dir"] == ""
    assert params["output_topic"] == "audio/tts/frame"
    assert params["output.source_id"] == "fa_tts"
    assert params["output.stream_id"] == "tts_synthesis"
    assert params["qos.depth"] == 10
    assert params["qos.reliable"] is True
    assert "playback_topic" not in params
    assert "use_playback_topic" not in params
    assert "stop_topic" not in params
    assert "default_volume_db" not in params
def test_tts_backend_factory_rejects_missing_or_unknown_backend() -> None:
    try:
        build_tts_backend("", openjtalk_dict_dir="")
    except RuntimeError as exc:
        assert str(exc) == "backend.name is required"
    else:
        raise AssertionError("missing backend.name was accepted")

    try:
        build_tts_backend("cloud_fallback", openjtalk_dict_dir="")
    except RuntimeError as exc:
        assert str(exc) == "unsupported fa_tts backend.name: cloud_fallback"
    else:
        raise AssertionError("unknown backend.name was accepted")


def test_pyopenjtalk_backend_does_not_import_runtime_until_selected() -> None:
    backend = PyOpenJTalkBackend.__new__(PyOpenJTalkBackend)

    assert backend.name == "pyopenjtalk"
