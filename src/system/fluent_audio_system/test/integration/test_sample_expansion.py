from pathlib import Path

import pytest
import yaml

from fluent_audio_system import config_schema
from fluent_audio_system.config_schema import load_required_packages, load_system_config


PACKAGE_ROOT = Path(__file__).parents[2]
FIXTURE_DIR = PACKAGE_ROOT / "test" / "fixtures"


def _patch_fluent_audio_system_share(monkeypatch: pytest.MonkeyPatch) -> None:
    def package_share(package_name: str) -> str:
        if package_name != "fluent_audio_system":
            raise RuntimeError(f"unexpected package share lookup: {package_name}")
        return str(PACKAGE_ROOT)

    monkeypatch.setattr(config_schema, "_get_package_share_directory", package_share)


def _patch_profile_package_shares(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    for package_name in (
        "fa_in",
        "fa_out",
        "fa_sample_format",
        "fa_resample",
        "fa_tts",
        "fa_mix",
    ):
        config_dir = tmp_path / package_name / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "default.yaml").write_text(
            f"{package_name}:\n  ros__parameters: {{}}\n",
            encoding="utf-8",
        )

    def package_share(package_name: str) -> str:
        if package_name == "fluent_audio_system":
            return str(PACKAGE_ROOT)
        if package_name in (
            "fa_in",
            "fa_out",
            "fa_sample_format",
            "fa_resample",
            "fa_tts",
            "fa_mix",
        ):
            return str(tmp_path / package_name)
        raise RuntimeError(f"unexpected package share lookup: {package_name}")

    monkeypatch.setattr(config_schema, "_get_package_share_directory", package_share)


def test_valid_fixture_expands_enabled_nodes_and_remappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    spec = load_system_config(
        "${share:fluent_audio_system}/test/fixtures/valid_io_system.yaml"
    )

    assert len(spec.groups) == 1
    assert spec.groups[0].id == "io"
    assert len(spec.groups[0].nodes) == 1
    node = spec.groups[0].nodes[0]
    assert node.id == "fa_in"
    assert node.package == "fa_in"
    assert node.executable == "fa_in_node"
    assert node.params_file == str(FIXTURE_DIR / "fa_in.params.yaml")
    assert node.launch_remappings() == [("audio/frame", "robot/audio/input")]

    params = _load_fixture_params("fa_in.params.yaml", "fa_in_node")
    assert params["audio.stream_id"] == "audio/frame"
    assert params["audio.layout"] == "interleaved"


def test_so101_profile_config_expands_default_site_bound_nodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)

    spec = load_system_config(
        "${share:fluent_audio_system}/config/profiles/so101.yaml"
    )

    enabled_nodes = [
        node
        for group in spec.groups
        for node in group.nodes
    ]

    assert [node.id for node in enabled_nodes] == ["fa_in", "fa_out"]
    assert [node.package for node in enabled_nodes] == ["fa_in", "fa_out"]
    assert enabled_nodes[0].params_file == str(
        tmp_path / "fa_in" / "config" / "default.yaml"
    )
    assert enabled_nodes[1].params_file == str(
        tmp_path / "fa_out" / "config" / "default.yaml"
    )


def test_so101_mic_frontend_profile_expands_explicit_format_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)

    spec = load_system_config(
        "${share:fluent_audio_system}/config/profiles/so101_mic_frontend.yaml"
    )

    enabled_nodes = [
        node
        for group in spec.groups
        for node in group.nodes
    ]

    assert [node.id for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
    ]
    assert [node.package for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
    ]

    sample_format = enabled_nodes[1]
    assert sample_format.params_file == str(
        tmp_path / "fa_sample_format" / "config" / "default.yaml"
    )
    assert sample_format.parameters == {
        "input_topic": "audio/frame",
        "output_topic": "audio/sample_format/mic",
        "expected.sample_rate": 48000,
    }

    resample = enabled_nodes[2]
    assert resample.params_file == str(
        tmp_path / "fa_resample" / "config" / "default.yaml"
    )
    assert resample.parameters == {
        "mic.input_topic": "audio/sample_format/mic",
        "mic.output_topic": "audio/resample16k/mic",
    }


@pytest.mark.parametrize(
    ("profile_path", "group_id"),
    (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
    ),
)
def test_vad_frontend_profiles_bind_consumers_to_vad_stream(
    profile_path: str,
    group_id: str,
) -> None:
    config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
    group = next(group for group in config["groups"] if group["id"] == group_id)
    params_by_id = {node["id"]: node.get("parameters", {}) for node in group["nodes"]}

    vad_stream_id = params_by_id["fa_vad"]["input_topic"]

    assert params_by_id["fa_kws"]["audio_topic"] == vad_stream_id
    assert params_by_id["fa_turn_detector"]["audio_topic"] == vad_stream_id
    assert params_by_id["fa_asr"]["audio_topic"] == vad_stream_id
    assert params_by_id["fa_asr"]["expected_stream_id"] == vad_stream_id


def test_required_packages_for_so101_mic_frontend_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/profiles/so101_mic_frontend.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_sample_format",
        "fa_resample",
    ]


def test_so101_tts_output_profile_expands_explicit_playback_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)
    dictionary_dir = tmp_path / "open_jtalk_dic"
    dictionary_dir.mkdir()
    monkeypatch.setenv("FLUENT_AUDIO_OPENJTALK_DICT_DIR", str(dictionary_dir))

    spec = load_system_config(
        "${share:fluent_audio_system}/config/profiles/so101_tts_output.yaml"
    )

    enabled_nodes = [
        node
        for group in spec.groups
        for node in group.nodes
    ]

    assert [node.id for node in enabled_nodes] == [
        "fa_out",
        "fa_tts",
        "fa_resample",
        "fa_sample_format",
        "fa_mix",
    ]
    assert [node.package for node in enabled_nodes] == [
        "fa_out",
        "fa_tts",
        "fa_resample",
        "fa_sample_format",
        "fa_mix",
    ]

    tts = enabled_nodes[1]
    assert tts.params_file == str(
        tmp_path / "fa_tts" / "config" / "default.yaml"
    )
    assert tts.parameters == {
        "backend.openjtalk_dict_dir": str(dictionary_dir),
        "output_topic": "audio/tts/frame",
    }

    resample = enabled_nodes[2]
    assert resample.params_file == str(
        tmp_path / "fa_resample" / "config" / "default.yaml"
    )
    assert resample.parameters == {
        "target_sample_rate": 48000,
        "mic.enabled": True,
        "mic.input_topic": "audio/tts/frame",
        "mic.output_topic": "audio/tts/48k_float32",
        "ref.enabled": False,
    }

    sample_format = enabled_nodes[3]
    assert sample_format.params_file == str(
        tmp_path / "fa_sample_format" / "config" / "default.yaml"
    )
    assert sample_format.parameters == {
        "input_topic": "audio/tts/48k_float32",
        "output_topic": "audio/tts/pcm16",
        "input.encoding": "FLOAT32LE",
        "input.bit_depth": 32,
        "output.encoding": "PCM16LE",
        "output.bit_depth": 16,
        "expected.sample_rate": 48000,
        "expected.channels": 1,
        "expected.layout": "interleaved",
    }

    mix = enabled_nodes[4]
    assert mix.params_file == str(
        tmp_path / "fa_mix" / "config" / "default.yaml"
    )
    assert mix.parameters == {
        "input_topics": ["audio/tts/pcm16"],
        "input_gains_db": [0.0],
        "master_index": 0,
        "output_topic": "audio/output/frame",
        "expected.sample_rate": 48000,
        "expected.channels": 1,
        "expected.bit_depth": 16,
        "expected.encoding": "PCM16LE",
    }


def test_so101_tts_output_profile_requires_openjtalk_dictionary_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)
    monkeypatch.delenv("FLUENT_AUDIO_OPENJTALK_DICT_DIR", raising=False)

    with pytest.raises(
        RuntimeError,
        match="environment variable FLUENT_AUDIO_OPENJTALK_DICT_DIR is required",
    ):
        load_system_config(
            "${share:fluent_audio_system}/config/profiles/so101_tts_output.yaml"
        )


def test_sample_config_documents_tts_playback_conversion_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)

    spec = load_system_config(
        "${share:fluent_audio_system}/config/fluent_audio_system.sample.yaml"
    )
    enabled_nodes = [
        node
        for group in spec.groups
        for node in group.nodes
    ]

    assert [node.id for node in enabled_nodes] == [
        "fa_in",
        "fa_out",
        "fa_sample_format",
        "fa_resample",
    ]
    assert enabled_nodes[2].parameters == {
        "input_topic": "audio/frame",
        "output_topic": "audio/sample_format/mic",
        "expected.sample_rate": 48000,
    }
    assert enabled_nodes[3].parameters == {
        "mic.input_topic": "audio/sample_format/mic",
        "mic.output_topic": "audio/resample16k/mic",
    }

    raw = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "fluent_audio_system.sample.yaml").read_text(
            encoding="utf-8"
        )
    )
    groups = {group["id"]: group for group in raw["groups"]}
    format_nodes = {node["id"]: node for node in groups["format"]["nodes"]}
    generation_nodes = {
        node["id"]: node for node in groups["generation_routing"]["nodes"]
    }

    assert format_nodes["fa_resample_tts"]["parameters"] == {
        "target_sample_rate": 48000,
        "mic.enabled": True,
        "mic.input_topic": "audio/tts/frame",
        "mic.output_topic": "audio/tts/48k_float32",
        "ref.enabled": False,
    }
    assert format_nodes["fa_sample_format_tts"]["parameters"] == {
        "input_topic": "audio/tts/48k_float32",
        "output_topic": "audio/tts/pcm16",
        "input.encoding": "FLOAT32LE",
        "input.bit_depth": 32,
        "output.encoding": "PCM16LE",
        "output.bit_depth": 16,
        "expected.sample_rate": 48000,
        "expected.channels": 1,
        "expected.layout": "interleaved",
    }
    assert generation_nodes["fa_tts"]["parameters"] == {
        "backend.openjtalk_dict_dir": "${env:FLUENT_AUDIO_OPENJTALK_DICT_DIR}",
        "output_topic": "audio/tts/frame",
    }
    assert generation_nodes["fa_mix"]["parameters"] == {
        "input_topics": ["audio/tts/pcm16"],
        "input_gains_db": [0.0],
        "master_index": 0,
        "output_topic": "audio/output/frame",
        "expected.sample_rate": 48000,
        "expected.channels": 1,
        "expected.bit_depth": 16,
        "expected.encoding": "PCM16LE",
    }


def test_sample_config_documents_disabled_analysis_feature_nodes() -> None:
    raw = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "fluent_audio_system.sample.yaml").read_text(
            encoding="utf-8"
        )
    )
    groups = {group["id"]: group for group in raw["groups"]}
    analysis_nodes = {node["id"]: node for node in groups["analysis"]["nodes"]}

    assert set(analysis_nodes) == {"fa_log_mel", "fa_loudness", "fa_stft"}
    assert analysis_nodes["fa_loudness"]["enable"] is False
    assert analysis_nodes["fa_loudness"]["package"] == "fa_loudness"
    assert analysis_nodes["fa_loudness"]["params_file"] == (
        "${share:fa_loudness}/config/default.yaml"
    )
    assert analysis_nodes["fa_loudness"]["parameters"] == {
        "input_topic": "audio/resample16k/mic",
    }


def test_missing_fixture_params_file_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    with pytest.raises(RuntimeError, match="params_file not found"):
        load_system_config(
            "${share:fluent_audio_system}/test/fixtures/missing_params_system.yaml"
        )


def test_invalid_schema_fixture_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    with pytest.raises(RuntimeError, match="system.inter_group_delay is required"):
        load_system_config(
            "${share:fluent_audio_system}/test/fixtures/invalid_schema_system.yaml"
        )


def test_sequence_remapping_fixture_expands_to_launch_pairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    spec = load_system_config(
        "${share:fluent_audio_system}/test/fixtures/remapping_system.yaml"
    )

    assert len(spec.groups) == 1
    assert len(spec.groups[0].nodes) == 1
    assert spec.groups[0].nodes[0].launch_remappings() == [
        ("audio/frame", "robot/audio/input"),
        ("vad/state", "robot/audio/vad/state"),
    ]


def _load_fixture_params(fixture_name: str, node_name: str) -> dict[str, str | int | bool]:
    raw = yaml.safe_load((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    return raw[node_name]["ros__parameters"]
