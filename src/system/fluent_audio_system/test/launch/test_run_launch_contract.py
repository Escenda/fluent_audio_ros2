from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[2]


def test_vlabor_include_action_matches_fluent_vision_profile_shape() -> None:
    fixture_path = PACKAGE_ROOT / "test" / "fixtures" / "vlabor_include_action.yaml"
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    action = fixture["action"]

    assert action["type"] == "include"
    assert action["package"] == "fluent_audio_system"
    assert action["launch"] == "run.py"
    assert action["enabled"] == "${fluent_audio_enabled}"
    assert set(action["args"]) == {
        "config",
        "fa_in_enabled",
        "fa_out_enabled",
        "fa_in_source_id",
        "fa_out_sink_id",
    }
    assert action["args"] == {
        "config": "${fluent_audio_config}",
        "fa_in_enabled": "${fluent_audio_fa_in_enabled}",
        "fa_out_enabled": "${fluent_audio_fa_out_enabled}",
        "fa_in_source_id": "${fluent_audio_fa_in_source_id}",
        "fa_out_sink_id": "${fluent_audio_fa_out_sink_id}",
    }


def test_vlabor_include_action_does_not_expose_backend_or_model_args() -> None:
    fixture_path = PACKAGE_ROOT / "test" / "fixtures" / "vlabor_include_action.yaml"
    fixture = yaml.safe_load(fixture_path.read_text(encoding="utf-8"))
    action = fixture["action"]
    serialized_args = "\n".join(
        [*action["args"].keys(), *[str(value) for value in action["args"].values()]]
    )

    for forbidden_token in (
        "backend",
        "model",
        "threshold",
        "prompt",
        "timeout",
        "endpoint",
        "api_key",
        "secret",
        "openai",
    ):
        assert forbidden_token not in serialized_args
