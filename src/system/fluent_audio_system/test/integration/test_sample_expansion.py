from pathlib import Path
from typing import TypeAlias

import pytest
import yaml

from fluent_audio_system import config_schema
from fluent_audio_system.config_schema import load_required_packages, load_system_config


PACKAGE_ROOT = Path(__file__).parents[2]
SRC_ROOT = PACKAGE_ROOT.parents[1]
FIXTURE_DIR = PACKAGE_ROOT / "test" / "fixtures"

FixtureParamScalar: TypeAlias = str | int | float | bool
FixtureParamMapping: TypeAlias = dict[str, "FixtureParamValue"]
FixtureParamValue: TypeAlias = FixtureParamScalar | FixtureParamMapping


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
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_audio_window",
        "fa_audio_mcp",
        "fa_dialogue",
        "fa_vad",
        "fa_kws",
        "fa_asr",
        "fa_turn_detector",
        "fa_tts",
        "fa_mix",
    ):
        config_dir = tmp_path / package_name / "config"
        config_dir.mkdir(parents=True)
        if package_name == "fa_in":
            config_text = "fa_in:\n  ros__parameters:\n    backend.name: alsa_capture\n"
        elif package_name == "fa_out":
            config_text = "fa_out:\n  ros__parameters:\n    backend.name: alsa_playback\n"
        elif package_name == "fa_audio_mcp":
            config_text = "fa_audio_mcp:\n  ros__parameters: {}\n"
        elif package_name == "fa_dialogue":
            config_text = "fa_dialogue:\n  ros__parameters: {}\n"
        elif package_name == "fa_vad":
            config_text = "fa_vad:\n  ros__parameters: {}\n"
        elif package_name == "fa_kws":
            config_text = "fa_kws:\n  ros__parameters: {}\n"
        elif package_name == "fa_asr":
            config_text = "fa_asr:\n  ros__parameters: {}\n"
        elif package_name == "fa_turn_detector":
            config_text = "fa_turn_detector:\n  ros__parameters: {}\n"
        elif package_name == "fa_tts":
            config_text = "fa_tts:\n  ros__parameters:\n    backend.name: pyopenjtalk\n"
        else:
            config_text = f"{package_name}:\n  ros__parameters: {{}}\n"
        (config_dir / "default.yaml").write_text(config_text, encoding="utf-8")

    def package_share(package_name: str) -> str:
        if package_name == "fluent_audio_system":
            return str(PACKAGE_ROOT)
        if package_name in (
            "fa_in",
            "fa_out",
            "fa_sample_format",
            "fa_resample",
            "fa_dc_offset_removal",
            "fa_high_pass",
            "fa_audio_window",
            "fa_audio_mcp",
            "fa_dialogue",
            "fa_vad",
            "fa_kws",
            "fa_asr",
            "fa_turn_detector",
            "fa_tts",
            "fa_mix",
        ):
            return str(tmp_path / package_name)
        raise RuntimeError(f"unexpected package share lookup: {package_name}")

    monkeypatch.setattr(config_schema, "_get_package_share_directory", package_share)


def _set_kws_frontend_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {
        "FLUENT_AUDIO_VAD_MODEL_DIR": str(tmp_path / "models" / "silero"),
        "FLUENT_AUDIO_VAD_PROVIDER": "cpu",
        "FLUENT_AUDIO_VAD_WORKER": str(tmp_path / "bin" / "silero_vad_worker"),
        "FLUENT_AUDIO_KWS_PROVIDER": "cpu",
        "FLUENT_AUDIO_KWS_WORKER": str(tmp_path / "bin" / "sherpa_onnx_kws_worker"),
        "FLUENT_AUDIO_KWS_ENCODER": str(tmp_path / "models" / "kws" / "encoder.onnx"),
        "FLUENT_AUDIO_KWS_DECODER": str(tmp_path / "models" / "kws" / "decoder.onnx"),
        "FLUENT_AUDIO_KWS_JOINER": str(tmp_path / "models" / "kws" / "joiner.onnx"),
        "FLUENT_AUDIO_KWS_TOKENS": str(tmp_path / "models" / "kws" / "tokens.txt"),
        "FLUENT_AUDIO_KWS_KEYWORDS": str(tmp_path / "models" / "kws" / "keywords.txt"),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)


def _set_voice_frontend_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _set_kws_frontend_env(monkeypatch, tmp_path)
    values = {
        "FLUENT_AUDIO_ASR_MODEL_PATH": str(
            tmp_path / "models" / "asr" / "ggml-large-v3.bin"
        ),
        "FLUENT_AUDIO_ASR_WORKER": str(tmp_path / "bin" / "whisper_cpp_worker"),
        "FLUENT_AUDIO_TURN_DETECTOR_MODEL": str(
            tmp_path / "models" / "turn_detector" / "smart_turn.onnx"
        ),
        "FLUENT_AUDIO_TURN_DETECTOR_PROVIDER": "cpu",
        "FLUENT_AUDIO_TURN_DETECTOR_WORKER": str(
            tmp_path / "bin" / "smart_turn_worker"
        ),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
def test_system_configs_keep_runtime_node_identity_package_aligned() -> None:
    violations: list[str] = []
    expected_node_names = {
        "fa_in": "fa_in",
        "fa_vad": "fa_vad",
    }
    for config_path in sorted((PACKAGE_ROOT / "config").rglob("*.yaml")):
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        for group in config["groups"]:
            for node in group["nodes"]:
                expected = expected_node_names.get(node["package"])
                if expected is None:
                    continue
                if node["node_name"] != expected:
                    relative_path = config_path.relative_to(PACKAGE_ROOT)
                    violations.append(
                        f"{relative_path}: {group['id']}/{node['id']} "
                        f"node_name={node['node_name']} expected={expected}"
                    )

    assert violations == []


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

    params = _load_fixture_params("fa_in.params.yaml", "fa_in")
    for required_param in (
        "backend.name",
        "output_topic",
        "audio.device_selector.mode",
        "audio.device_selector.identifier",
        "audio.device_selector.index",
        "audio.sample_rate",
        "audio.channels",
        "audio.bit_depth",
        "audio.chunk_ms",
        "audio.encoding",
        "audio.stream_id",
        "audio.layout",
        "audio.qos.depth",
        "audio.qos.reliable",
        "diagnostics.publish_period_ms",
        "diagnostics.qos.depth",
        "diagnostics.qos.reliable",
    ):
        assert required_param in params
    assert params["output_topic"] == "audio/frame"
    assert params["audio.device_selector.mode"] == "id"
    assert params["audio.device_selector.identifier"] == "hw:CARD=Loopback,DEV=0"
    assert params["audio.device_selector.index"] == -1
    assert params["audio.stream_id"] == "audio/raw/mic"
    assert params["audio.layout"] == "interleaved"
    assert params["audio.qos.depth"] == 10
    assert params["audio.qos.reliable"] is False
    assert params["diagnostics.qos.depth"] == 10
    assert params["diagnostics.qos.reliable"] is False


def test_required_packages_for_full_sample_config_validate_package_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/fluent_audio_system.sample.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_out",
        "fa_sample_format",
        "fa_resample",
    ]


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
    assert [node.node_name for node in enabled_nodes] == ["fa_in", "fa_out"]
    assert enabled_nodes[0].params_file == str(
        tmp_path / "fa_in" / "config" / "default.yaml"
    )
    assert enabled_nodes[1].params_file == str(
        tmp_path / "fa_out" / "config" / "default.yaml"
    )
    assert enabled_nodes[0].parameters == {
        "output_topic": "audio/frame",
        "audio.stream_id": "audio/raw/mic",
    }
    assert enabled_nodes[1].parameters == {
        "input_topic": "audio/output/frame",
        "input_stream_id": "audio/playback/main",
        "playback_done_topic": "audio/output/playback_done",
        "playback_control_service": "audio/output/playback_control",
    }


def test_required_packages_for_so101_profile_excludes_disabled_pipelines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/profiles/so101.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_out",
    ]
    assert "fa_vad" not in packages
    assert "fa_kws" not in packages
    assert "fa_asr" not in packages
    assert "fa_turn_detector" not in packages
    assert "fa_denoise" not in packages


def test_audio_embedding_profiles_keep_backend_and_model_out_of_profile_config() -> None:
    for profile_path, group_id in (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
    ):
        config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
        ai_group = next(group for group in config["groups"] if group["id"] == group_id)
        audio_embedding = next(
            node for node in ai_group["nodes"] if node["id"] == "fa_audio_embedding"
        )

        assert audio_embedding["enable"] is False
        assert audio_embedding["package"] == "fa_audio_embedding"
        assert audio_embedding["params_file"] == (
            "${share:fa_audio_embedding}/config/default.yaml"
        )
        expected_binding = (
            {
                "input_topic": "audio/high_pass/frame",
                "expected_stream_id": "audio/high_pass/mic",
            }
            if profile_path == "config/profiles/so101_mic_frontend.yaml"
            else {
                "input_topic": "audio/resample16k/mic",
                "expected_stream_id": "audio/preprocessed/mono16k",
            }
        )
        assert audio_embedding["parameters"] == expected_binding
        assert all(
            not key.startswith("backend.") and "model" not in key
            for key in audio_embedding["parameters"]
        )


def test_so101_mic_frontend_profile_expands_explicit_preprocess_pipeline(
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
        "fa_dc_offset_removal",
        "fa_high_pass",
    ]
    assert [node.package for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
    ]
    assert enabled_nodes[0].node_name == "fa_in"
    assert enabled_nodes[0].parameters == {
        "output_topic": "audio/frame",
        "audio.stream_id": "audio/raw/mic",
    }

    sample_format = enabled_nodes[1]
    assert sample_format.params_file == str(
        tmp_path / "fa_sample_format" / "config" / "default.yaml"
    )
    assert sample_format.parameters == {
        "input_topic": "audio/frame",
        "output_topic": "audio/sample_format/mic",
        "input_stream_id": "audio/raw/mic",
        "output.stream_id": "audio/float32/mic",
        "expected.sample_rate": 48000,
    }
    assert (
        sample_format.parameters["input_topic"]
        == enabled_nodes[0].parameters["output_topic"]
    )
    assert (
        sample_format.parameters["input_stream_id"]
        == enabled_nodes[0].parameters["audio.stream_id"]
    )

    resample = enabled_nodes[2]
    assert resample.params_file == str(
        tmp_path / "fa_resample" / "config" / "default.yaml"
    )
    assert resample.parameters == {
        "mic.input_topic": "audio/sample_format/mic",
        "mic.output_topic": "audio/resample16k/mic",
        "mic.input_stream_id": "audio/float32/mic",
        "mic.output.stream_id": "audio/preprocessed/mono16k",
    }

    dc_offset = enabled_nodes[3]
    assert dc_offset.params_file == str(
        tmp_path / "fa_dc_offset_removal" / "config" / "default.yaml"
    )
    assert dc_offset.parameters == {
        "input_topic": "audio/resample16k/mic",
        "output_topic": "audio/dc_offset_removed/frame",
        "input_stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/dc_offset_removed/mic",
        "expected.sample_rate": 16000,
    }
    assert dc_offset.parameters["input_topic"] == resample.parameters["mic.output_topic"]
    assert (
        dc_offset.parameters["input_stream_id"]
        == resample.parameters["mic.output.stream_id"]
    )

    high_pass = enabled_nodes[4]
    assert high_pass.params_file == str(
        tmp_path / "fa_high_pass" / "config" / "default.yaml"
    )
    assert high_pass.parameters == {
        "input_topic": "audio/dc_offset_removed/frame",
        "output_topic": "audio/high_pass/frame",
        "input_stream_id": "audio/dc_offset_removed/mic",
        "output.stream_id": "audio/high_pass/mic",
        "filter.cutoff_hz": 80.0,
        "expected.sample_rate": 16000,
    }
    assert high_pass.parameters["input_topic"] == dc_offset.parameters["output_topic"]
    assert high_pass.parameters["input_stream_id"] == dc_offset.parameters["output.stream_id"]


def test_so101_kws_frontend_profile_expands_vad_and_kws_worker_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)
    _set_kws_frontend_env(monkeypatch, tmp_path)

    spec = load_system_config(
        "${share:fluent_audio_system}/config/profiles/so101_kws_frontend.yaml"
    )

    enabled_nodes = [
        node
        for group in spec.groups
        for node in group.nodes
    ]
    params_by_id = {node.id: node.parameters for node in enabled_nodes}

    assert [node.id for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_vad",
        "fa_kws",
    ]
    assert [node.package for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_vad",
        "fa_kws",
    ]

    vad_params = params_by_id["fa_vad"]
    high_pass_params = params_by_id["fa_high_pass"]
    assert vad_params["input_topic"] == high_pass_params["output_topic"]
    assert vad_params["input_stream_id"] == high_pass_params["output.stream_id"]
    assert vad_params["backend.name"] == "silero"
    assert vad_params["backend.model_path"] == str(tmp_path / "models" / "silero")
    assert vad_params["backend.execution_provider"] == "cpu"
    assert vad_params["backend.command"] == str(tmp_path / "bin" / "silero_vad_worker")

    kws_params = params_by_id["fa_kws"]
    assert kws_params["audio_topic"] == vad_params["input_topic"]
    assert kws_params["expected_stream_id"] == vad_params["input_stream_id"]
    assert kws_params["vad_topic"] == "voice/vad_state"
    assert kws_params["output_topic"] == "voice/wake_word"
    assert kws_params["backend.name"] == "sherpa_onnx_kws"
    assert kws_params["backend.execution_provider"] == "cpu"
    assert kws_params["backend.command"] == str(
        tmp_path / "bin" / "sherpa_onnx_kws_worker"
    )
    assert kws_params["model.encoder"] == str(tmp_path / "models" / "kws" / "encoder.onnx")
    assert kws_params["model.decoder"] == str(tmp_path / "models" / "kws" / "decoder.onnx")
    assert kws_params["model.joiner"] == str(tmp_path / "models" / "kws" / "joiner.onnx")
    assert kws_params["model.tokens"] == str(tmp_path / "models" / "kws" / "tokens.txt")
    assert kws_params["kws.keywords_file"] == str(
        tmp_path / "models" / "kws" / "keywords.txt"
    )
    assert kws_params["backend.args"][:3] == ["detect", "--audio", "{audio}"]
    assert "{audio}" not in kws_params["backend.health_args"]
    assert kws_params["backend.timeout_sec"] == 5.0
    assert kws_params["backend.workspace_dir"] == "/tmp/fluent_audio/fa_kws/so101"
    assert kws_params["backend.cleanup_audio_files"] is True
    assert kws_params["output.qos.depth"] == 10
    assert kws_params["output.qos.reliable"] is False
    assert "qos.depth" not in kws_params
    assert "qos.reliable" not in kws_params
    for placeholder in (
        "{encoder}",
        "{decoder}",
        "{joiner}",
        "{tokens}",
        "{keywords}",
        "{provider}",
        "{sample_rate}",
        "{num_threads}",
        "{max_active_paths}",
        "{num_trailing_blanks}",
        "{keywords_score}",
        "{keywords_threshold}",
    ):
        assert placeholder in kws_params["backend.args"]
        assert placeholder in kws_params["backend.health_args"]


def test_so101_kws_frontend_profile_requires_worker_and_model_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)

    with pytest.raises(
        RuntimeError,
        match="environment variable FLUENT_AUDIO_VAD_MODEL_DIR is required",
    ):
        load_system_config(
            "${share:fluent_audio_system}/config/profiles/so101_kws_frontend.yaml"
        )


def test_so101_voice_frontend_profile_expands_full_voice_backend_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)
    _set_voice_frontend_env(monkeypatch, tmp_path)

    spec = load_system_config(
        "${share:fluent_audio_system}/config/profiles/so101_voice_frontend.yaml"
    )

    enabled_nodes = [node for group in spec.groups for node in group.nodes]
    params_by_id = {node.id: node.parameters for node in enabled_nodes}

    assert [node.id for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_archive_sample_format",
        "fa_audio_window",
        "fa_vad",
        "fa_kws",
        "fa_asr",
        "fa_turn_detector",
        "fa_dialogue",
    ]
    assert [node.package for node in enabled_nodes] == [
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_sample_format",
        "fa_audio_window",
        "fa_vad",
        "fa_kws",
        "fa_asr",
        "fa_turn_detector",
        "fa_dialogue",
    ]

    vad_params = params_by_id["fa_vad"]
    high_pass_params = params_by_id["fa_high_pass"]
    archive_format_params = params_by_id["fa_archive_sample_format"]
    audio_window_params = params_by_id["fa_audio_window"]
    asr_params = params_by_id["fa_asr"]
    turn_detector_params = params_by_id["fa_turn_detector"]
    dialogue_params = params_by_id["fa_dialogue"]

    assert vad_params["input_topic"] == high_pass_params["output_topic"]
    assert vad_params["input_stream_id"] == high_pass_params["output.stream_id"]
    assert archive_format_params["input_topic"] == high_pass_params["output_topic"]
    assert archive_format_params["input_stream_id"] == high_pass_params["output.stream_id"]
    assert archive_format_params["output_topic"] == "audio/archive_pcm16/frame"
    assert archive_format_params["output.stream_id"] == "audio/archive_pcm16/mic"
    assert archive_format_params["input.encoding"] == "FLOAT32LE"
    assert archive_format_params["input.bit_depth"] == 32
    assert archive_format_params["output.encoding"] == "PCM16LE"
    assert archive_format_params["output.bit_depth"] == 16
    assert audio_window_params["input_topic"] == archive_format_params["output_topic"]
    assert audio_window_params["input.stream_id"] == archive_format_params["output.stream_id"]
    assert audio_window_params["input.source_id"] == "mic"
    assert audio_window_params["service_name"] == "export_audio_window"
    assert audio_window_params["archive_service_name"] == "archive_audio_window"
    assert audio_window_params["expected.encoding"] == "PCM16LE"
    assert audio_window_params["expected.sample_rate"] == 16000
    assert audio_window_params["expected.channels"] == 1
    assert audio_window_params["expected.bit_depth"] == 16
    assert audio_window_params["expected.layout"] == "interleaved"
    assert audio_window_params["window.retention_seconds"] == 1800
    assert audio_window_params["window.id"] == "so101_voice_frontend_mic_archive"
    assert audio_window_params["window.epoch"] == 1
    assert audio_window_params["audio.default_scope"] == "mic"
    assert audio_window_params["audio.supported_scopes"] == ["mic"]
    assert (
        audio_window_params["export.output_directory"]
        == "/tmp/fluent_audio/fa_audio_window/so101_voice_frontend"
    )
    assert asr_params["audio_topic"] == vad_params["input_topic"]
    assert asr_params["expected_stream_id"] == vad_params["input_stream_id"]
    assert asr_params["vad_topic"] == "voice/vad_state"
    assert asr_params["turn_context_topic"] == "conversation/turn_context"
    assert asr_params["asr_result_topic"] == "voice/asr/result"
    assert asr_params["backend.name"] == "whisper.cpp"
    assert asr_params["backend.model_path"] == str(
        tmp_path / "models" / "asr" / "ggml-large-v3.bin"
    )
    assert asr_params["backend.command"] == str(
        tmp_path / "bin" / "whisper_cpp_worker"
    )
    assert asr_params["workspace_dir"] == "/tmp/fluent_audio/fa_asr/so101_voice_frontend"
    assert "{audio}" in asr_params["backend.args"]
    assert "{model}" in asr_params["backend.args"]
    assert "{sample_rate}" in asr_params["backend.args"]
    assert "{audio}" not in asr_params["backend.health_args"]
    assert "{model}" in asr_params["backend.health_args"]

    assert turn_detector_params["audio_topic"] == vad_params["input_topic"]
    assert turn_detector_params["expected_stream_id"] == vad_params["input_stream_id"]
    assert turn_detector_params["vad_topic"] == "voice/vad_state"
    assert turn_detector_params["turn_context_topic"] == "conversation/turn_context"
    assert turn_detector_params["output_topic"] == "voice/turn_end"
    assert turn_detector_params["backend.name"] == "smart_turn_onnx"
    assert turn_detector_params["backend.model_path"] == str(
        tmp_path / "models" / "turn_detector" / "smart_turn.onnx"
    )
    assert turn_detector_params["backend.execution_provider"] == "cpu"
    assert turn_detector_params["backend.command"] == str(
        tmp_path / "bin" / "smart_turn_worker"
    )
    assert (
        turn_detector_params["backend.workspace_dir"]
        == "/tmp/fluent_audio/fa_turn_detector/so101_voice_frontend"
    )
    assert "{audio}" in turn_detector_params["backend.args"]
    assert "{audio}" not in turn_detector_params["backend.health_args"]
    assert "{model}" in turn_detector_params["backend.health_args"]
    assert "{provider}" in turn_detector_params["backend.health_args"]

    assert dialogue_params["wake_word_topic"] == "voice/wake_word"
    assert dialogue_params["asr_result_topic"] == asr_params["asr_result_topic"]
    assert dialogue_params["turn_end_topic"] == turn_detector_params["output_topic"]
    assert dialogue_params["turn_context_topic"] == asr_params["turn_context_topic"]
    assert dialogue_params["turn_context_topic"] == turn_detector_params["turn_context_topic"]
    assert dialogue_params["session_prefix"] == "so101_voice-"
    assert dialogue_params["wake.max_age_ms"] == 1500
    assert dialogue_params["wake.allow_zero_stamp"] is False


def test_so101_voice_frontend_profile_requires_asr_and_turn_detector_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)
    _set_kws_frontend_env(monkeypatch, tmp_path)

    with pytest.raises(
        RuntimeError,
        match="environment variable FLUENT_AUDIO_ASR_MODEL_PATH is required",
    ):
        load_system_config(
            "${share:fluent_audio_system}/config/profiles/so101_voice_frontend.yaml"
        )


@pytest.mark.parametrize(
    ("profile_path", "group_id"),
    (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
    ),
)
def test_voice_frontend_profiles_bind_ai_consumers_to_resampled_mic_stream(
    profile_path: str,
    group_id: str,
) -> None:
    config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
    group = next(group for group in config["groups"] if group["id"] == group_id)
    params_by_id = {node["id"]: node.get("parameters", {}) for node in group["nodes"]}

    vad_transport_topic = params_by_id["fa_vad"]["input_topic"]
    vad_stream_id = params_by_id["fa_vad"]["input_stream_id"]

    assert params_by_id["fa_kws"]["audio_topic"] == vad_transport_topic
    assert params_by_id["fa_kws"]["expected_stream_id"] == vad_stream_id
    assert params_by_id["fa_turn_detector"]["audio_topic"] == vad_transport_topic
    assert params_by_id["fa_turn_detector"]["expected_stream_id"] == vad_stream_id
    assert params_by_id["fa_asr"]["audio_topic"] == vad_transport_topic
    assert params_by_id["fa_asr"]["expected_stream_id"] == vad_stream_id
    assert params_by_id["fa_audio_embedding"]["input_topic"] == vad_transport_topic
    assert params_by_id["fa_audio_embedding"]["expected_stream_id"] == vad_stream_id


@pytest.mark.parametrize(
    ("profile_path", "group_id"),
    (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
        ("config/profiles/so101_kws_frontend.yaml", "voice_frontend"),
        ("config/profiles/so101_voice_frontend.yaml", "voice_frontend"),
    ),
)
def test_vad_profiles_carry_external_worker_contract_in_system_config(
    profile_path: str,
    group_id: str,
) -> None:
    config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
    group = next(group for group in config["groups"] if group["id"] == group_id)
    vad = next(node for node in group["nodes"] if node["id"] == "fa_vad")
    params = vad["parameters"]

    assert params["backend.name"] == "silero"
    assert params["backend.model_path"] == "${env:FLUENT_AUDIO_VAD_MODEL_DIR}"
    assert params["backend.execution_provider"] == "${env:FLUENT_AUDIO_VAD_PROVIDER}"
    assert params["backend.command"] == "${env:FLUENT_AUDIO_VAD_WORKER}"
    expected_topic, expected_stream_id = (
        ("audio/high_pass/frame", "audio/high_pass/mic")
        if profile_path.startswith("config/profiles/so101_")
        else ("audio/resample16k/mic", "audio/preprocessed/mono16k")
    )
    assert params["input_topic"] == expected_topic
    assert params["input_stream_id"] == expected_stream_id


@pytest.mark.parametrize(
    ("profile_path", "group_id"),
    (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
        ("config/profiles/so101_kws_frontend.yaml", "voice_frontend"),
        ("config/profiles/so101_voice_frontend.yaml", "voice_frontend"),
    ),
)
def test_kws_profiles_carry_external_worker_contract_in_system_config(
    profile_path: str,
    group_id: str,
) -> None:
    config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
    group = next(group for group in config["groups"] if group["id"] == group_id)
    kws = next(node for node in group["nodes"] if node["id"] == "fa_kws")
    params = kws["parameters"]

    assert params["backend.name"] == "sherpa_onnx_kws"
    assert params["backend.command"] == "${env:FLUENT_AUDIO_KWS_WORKER}"
    assert params["backend.execution_provider"] == "${env:FLUENT_AUDIO_KWS_PROVIDER}"
    assert params["model.encoder"] == "${env:FLUENT_AUDIO_KWS_ENCODER}"
    assert params["model.decoder"] == "${env:FLUENT_AUDIO_KWS_DECODER}"
    assert params["model.joiner"] == "${env:FLUENT_AUDIO_KWS_JOINER}"
    assert params["model.tokens"] == "${env:FLUENT_AUDIO_KWS_TOKENS}"
    assert params["kws.keywords_file"] == "${env:FLUENT_AUDIO_KWS_KEYWORDS}"
    assert params["backend.timeout_sec"] == 5.0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["output.qos.depth"] == 10
    assert params["output.qos.reliable"] is False
    assert "{audio}" in params["backend.args"]
    assert "{audio}" not in params["backend.health_args"]
    assert "qos.depth" not in params
    assert "qos.reliable" not in params


@pytest.mark.parametrize(
    ("profile_path", "group_id"),
    (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
        ("config/profiles/so101_voice_frontend.yaml", "voice_frontend"),
    ),
)
def test_asr_profiles_carry_whisper_worker_contract_in_system_config(
    profile_path: str,
    group_id: str,
) -> None:
    config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
    group = next(group for group in config["groups"] if group["id"] == group_id)
    asr = next(node for node in group["nodes"] if node["id"] == "fa_asr")
    params = asr["parameters"]

    assert params["backend.name"] == "whisper.cpp"
    assert params["backend.command"] == "${env:FLUENT_AUDIO_ASR_WORKER}"
    assert params["backend.model_path"] == "${env:FLUENT_AUDIO_ASR_MODEL_PATH}"
    assert params["vad_topic"] == "voice/vad_state"
    assert params["turn_context_topic"] == "conversation/turn_context"
    assert params["asr_result_topic"] == "voice/asr/result"
    assert params["backend.timeout_sec"] == 120.0
    assert params["workspace_dir"]
    assert params["cleanup_audio_files"] is True
    assert params["result.qos.depth"] == 10
    assert params["result.qos.reliable"] is True
    assert "{audio}" in params["backend.args"]
    assert "{model}" in params["backend.args"]
    assert "{sample_rate}" in params["backend.args"]
    assert "{model}" in params["backend.health_args"]
    assert "{audio}" not in params["backend.health_args"]


@pytest.mark.parametrize(
    ("profile_path", "group_id"),
    (
        ("config/fluent_audio_system.sample.yaml", "ai"),
        ("config/profiles/so101.yaml", "ai"),
        ("config/profiles/so101_mic_frontend.yaml", "voice_frontend"),
        ("config/profiles/so101_voice_frontend.yaml", "voice_frontend"),
    ),
)
def test_turn_detector_profiles_carry_external_worker_contract_in_system_config(
    profile_path: str,
    group_id: str,
) -> None:
    config = yaml.safe_load((PACKAGE_ROOT / profile_path).read_text(encoding="utf-8"))
    group = next(group for group in config["groups"] if group["id"] == group_id)
    turn_detector = next(node for node in group["nodes"] if node["id"] == "fa_turn_detector")
    params = turn_detector["parameters"]

    assert params["backend.name"] == "smart_turn_onnx"
    assert params["backend.command"] == "${env:FLUENT_AUDIO_TURN_DETECTOR_WORKER}"
    assert params["backend.model_path"] == "${env:FLUENT_AUDIO_TURN_DETECTOR_MODEL}"
    assert (
        params["backend.execution_provider"]
        == "${env:FLUENT_AUDIO_TURN_DETECTOR_PROVIDER}"
    )
    assert params["vad_topic"] == "voice/vad_state"
    assert params["turn_context_topic"] == "conversation/turn_context"
    assert params["output_topic"] == "voice/turn_end"
    assert params["backend.timeout_sec"] == 5.0
    assert params["backend.workspace_dir"]
    assert params["backend.cleanup_audio_files"] is True
    assert params["backend.threshold"] == 0.5
    assert params["output.qos.depth"] == 10
    assert params["output.qos.reliable"] is True
    assert "{audio}" in params["backend.args"]
    assert "{model}" in params["backend.args"]
    assert "{provider}" in params["backend.args"]
    assert "{model}" in params["backend.health_args"]
    assert "{provider}" in params["backend.health_args"]
    assert "{audio}" not in params["backend.health_args"]


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
        "fa_dc_offset_removal",
        "fa_high_pass",
    ]


def test_required_packages_for_so101_kws_frontend_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/profiles/so101_kws_frontend.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_vad",
        "fa_kws",
    ]


def test_required_packages_for_so101_voice_frontend_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/profiles/so101_voice_frontend.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_in",
        "fa_sample_format",
        "fa_resample",
        "fa_dc_offset_removal",
        "fa_high_pass",
        "fa_audio_window",
        "fa_vad",
        "fa_kws",
        "fa_asr",
        "fa_turn_detector",
        "fa_dialogue",
    ]


def test_so101_agent_audio_tools_profile_expands_mcp_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_profile_package_shares(monkeypatch, tmp_path)

    spec = load_system_config(
        "${share:fluent_audio_system}/config/profiles/so101_agent_audio_tools.yaml"
    )

    enabled_nodes = [node for group in spec.groups for node in group.nodes]

    assert [node.id for node in enabled_nodes] == ["fa_audio_mcp"]
    node = enabled_nodes[0]
    assert node.package == "fa_audio_mcp"
    assert node.executable == "fa_audio_mcp_server"
    assert node.node_name == "fa_audio_mcp_server"
    assert node.env == {
        "FLUENT_AUDIO_MCP_TRANSPORT": "streamable-http",
        "FLUENT_AUDIO_MCP_HOST": "127.0.0.1",
        "FLUENT_AUDIO_MCP_PORT": "9110",
        "FLUENT_AUDIO_MCP_SERVICE_TIMEOUT_SEC": "10.0",
        "FLUENT_AUDIO_ARCHIVE_AUDIO_WINDOW_SERVICE": "archive_audio_window",
        "FLUENT_AUDIO_TRANSCRIBE_AUDIO_SERVICE": "transcribe_audio",
        "FLUENT_AUDIO_ARCHIVE_SCOPE_MIC": "mic",
        "FLUENT_AUDIO_TRANSCRIBE_SCOPE_MIC": "audio/high_pass/mic",
    }


def test_required_packages_for_so101_agent_audio_tools_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/profiles/so101_agent_audio_tools.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_audio_mcp",
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
        "mic.input_stream_id": "tts_synthesis",
        "mic.output.stream_id": "tts_synthesis_48k",
        "ref.enabled": False,
    }

    sample_format = enabled_nodes[3]
    assert sample_format.params_file == str(
        tmp_path / "fa_sample_format" / "config" / "default.yaml"
    )
    assert sample_format.parameters == {
        "input_topic": "audio/tts/48k_float32",
        "output_topic": "audio/tts/pcm16",
        "input_stream_id": "tts_synthesis_48k",
        "output.stream_id": "tts_playback_pcm16",
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
        "input_stream_ids": ["tts_playback_pcm16"],
        "input_gains_db": [0.0],
        "master_index": 0,
        "output_topic": "audio/output/frame",
        "output.stream_id": "audio/playback/main",
        "expected.sample_rate": 48000,
        "expected.channels": 1,
        "expected.bit_depth": 16,
        "expected.encoding": "PCM16LE",
    }


def test_required_packages_for_so101_tts_output_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_fluent_audio_system_share(monkeypatch)

    packages = load_required_packages(
        "${share:fluent_audio_system}/config/profiles/so101_tts_output.yaml"
    )

    assert packages == [
        "fa_interfaces",
        "fluent_audio_system",
        "fa_out",
        "fa_tts",
        "fa_resample",
        "fa_sample_format",
        "fa_mix",
    ]
    assert "fa_in" not in packages
    assert "fa_vad" not in packages
    assert "fa_asr" not in packages


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
    assert enabled_nodes[0].parameters == {
        "output_topic": "audio/frame",
        "audio.stream_id": "audio/raw/mic",
    }
    assert enabled_nodes[1].parameters == {
        "input_topic": "audio/output/frame",
        "input_stream_id": "audio/playback/main",
        "playback_done_topic": "audio/output/playback_done",
        "playback_control_service": "audio/output/playback_control",
    }
    assert enabled_nodes[2].parameters == {
        "input_topic": "audio/frame",
        "output_topic": "audio/sample_format/mic",
        "input_stream_id": "audio/raw/mic",
        "output.stream_id": "audio/float32/mic",
        "expected.sample_rate": 48000,
    }
    assert enabled_nodes[3].parameters == {
        "mic.input_topic": "audio/sample_format/mic",
        "mic.output_topic": "audio/resample16k/mic",
        "mic.input_stream_id": "audio/float32/mic",
        "mic.output.stream_id": "audio/preprocessed/mono16k",
    }
    assert (
        enabled_nodes[2].parameters["input_topic"]
        == enabled_nodes[0].parameters["output_topic"]
    )
    assert (
        enabled_nodes[2].parameters["input_stream_id"]
        == enabled_nodes[0].parameters["audio.stream_id"]
    )
    assert (
        enabled_nodes[3].parameters["mic.input_topic"]
        == enabled_nodes[2].parameters["output_topic"]
    )
    assert (
        enabled_nodes[3].parameters["mic.input_stream_id"]
        == enabled_nodes[2].parameters["output.stream_id"]
    )

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
        "mic.input_stream_id": "tts_synthesis",
        "mic.output.stream_id": "tts_synthesis_48k",
        "ref.enabled": False,
    }
    assert format_nodes["fa_encode"]["enable"] is False
    assert format_nodes["fa_encode"]["package"] == "fa_encode"
    assert format_nodes["fa_encode"]["params_file"] == (
        "${share:fa_encode}/config/default.yaml"
    )
    assert format_nodes["fa_encode"]["parameters"] == {
        "backend.command.executable": "${env:FLUENT_AUDIO_CODEC_ENCODER}",
        "input_topic": "audio/resample16k/mic",
        "output_topic": "audio/encoded/mic",
        "input_stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/encoded/mic/opus",
    }
    assert format_nodes["fa_decode"]["enable"] is False
    assert format_nodes["fa_decode"]["package"] == "fa_decode"
    assert format_nodes["fa_decode"]["params_file"] == (
        "${share:fa_decode}/config/default.yaml"
    )
    assert format_nodes["fa_decode"]["parameters"] == {
        "backend.command.executable": "${env:FLUENT_AUDIO_CODEC_DECODER}",
        "input_topic": "audio/encoded/mic",
        "output_topic": "audio/decoded/mic",
        "input_stream_id": "audio/encoded/mic/opus",
        "output.stream_id": "audio/decoded/mic/pcm16",
    }
    assert format_nodes["fa_sample_format_tts"]["parameters"] == {
        "input_topic": "audio/tts/48k_float32",
        "output_topic": "audio/tts/pcm16",
        "input_stream_id": "tts_synthesis_48k",
        "output.stream_id": "tts_playback_pcm16",
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
        "input_stream_ids": ["tts_playback_pcm16"],
        "input_gains_db": [0.0],
        "master_index": 0,
        "output_topic": "audio/output/frame",
        "output.stream_id": "audio/playback/main",
        "expected.sample_rate": 48000,
        "expected.channels": 1,
        "expected.bit_depth": 16,
        "expected.encoding": "PCM16LE",
    }
    assert (
        generation_nodes["fa_mix"]["parameters"]["output_topic"]
        == enabled_nodes[1].parameters["input_topic"]
    )
    assert (
        generation_nodes["fa_mix"]["parameters"]["output.stream_id"]
        == enabled_nodes[1].parameters["input_stream_id"]
    )


def test_sample_config_documents_disabled_analysis_feature_nodes() -> None:
    raw = yaml.safe_load(
        (PACKAGE_ROOT / "config" / "fluent_audio_system.sample.yaml").read_text(
            encoding="utf-8"
        )
    )
    groups = {group["id"]: group for group in raw["groups"]}
    analysis_nodes = {node["id"]: node for node in groups["analysis"]["nodes"]}

    assert set(analysis_nodes) == {
        "fa_cqt",
        "fa_log_mel",
        "fa_loudness",
        "fa_mfcc",
        "fa_onset",
        "fa_pitch",
        "fa_stft",
        "fa_tempo",
    }
    for node_id, output_stream_id in (
        ("fa_cqt", "audio/features/cqt/frames"),
        ("fa_log_mel", "audio/features/log_mel/frames"),
        ("fa_loudness", "audio/features/loudness/frames"),
        ("fa_mfcc", "audio/features/mfcc/frames"),
        ("fa_onset", "audio/features/onset/frames"),
        ("fa_pitch", "audio/features/pitch/frames"),
        ("fa_stft", "audio/features/stft/frames"),
        ("fa_tempo", "audio/features/tempo/frames"),
    ):
        params = analysis_nodes[node_id]["parameters"]
        assert params["expected.stream_id"] == "audio/preprocessed/mono16k"
        assert params["output.stream_id"] == output_stream_id
        assert params["expected.stream_id"] != params["input_topic"]
        assert params["output.stream_id"] != params["input_topic"]

    assert analysis_nodes["fa_cqt"]["enable"] is False
    assert analysis_nodes["fa_cqt"]["package"] == "fa_cqt"
    assert analysis_nodes["fa_cqt"]["params_file"] == (
        "${share:fa_cqt}/config/default.yaml"
    )
    assert analysis_nodes["fa_cqt"]["parameters"] == {
        "input_topic": "audio/frame_buffer/cqt",
        "expected.stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/features/cqt/frames",
        "feature.frame_length": 4096,
        "feature.hop_length": 512,
    }
    assert analysis_nodes["fa_loudness"]["enable"] is False
    assert analysis_nodes["fa_loudness"]["package"] == "fa_loudness"
    assert analysis_nodes["fa_loudness"]["params_file"] == (
        "${share:fa_loudness}/config/default.yaml"
    )
    assert analysis_nodes["fa_loudness"]["parameters"] == {
        "input_topic": "audio/resample16k/mic",
        "expected.stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/features/loudness/frames",
    }
    assert analysis_nodes["fa_mfcc"]["enable"] is False
    assert analysis_nodes["fa_mfcc"]["package"] == "fa_mfcc"
    assert analysis_nodes["fa_mfcc"]["params_file"] == (
        "${share:fa_mfcc}/config/default.yaml"
    )
    assert analysis_nodes["fa_mfcc"]["parameters"] == {
        "input_topic": "audio/resample16k/mic",
        "expected.stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/features/mfcc/frames",
        "feature.n_fft": 320,
        "feature.hop_length": 160,
        "feature.n_mfcc": 13,
    }
    assert analysis_nodes["fa_onset"]["enable"] is False
    assert analysis_nodes["fa_onset"]["package"] == "fa_onset"
    assert analysis_nodes["fa_onset"]["params_file"] == (
        "${share:fa_onset}/config/default.yaml"
    )
    assert analysis_nodes["fa_onset"]["parameters"] == {
        "input_topic": "audio/resample16k/mic",
        "expected.stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/features/onset/frames",
        "feature.n_fft": 320,
        "feature.hop_length": 160,
        "detector.threshold": 0.1,
    }
    assert analysis_nodes["fa_pitch"]["enable"] is False
    assert analysis_nodes["fa_pitch"]["package"] == "fa_pitch"
    assert analysis_nodes["fa_pitch"]["params_file"] == (
        "${share:fa_pitch}/config/default.yaml"
    )
    assert analysis_nodes["fa_pitch"]["parameters"] == {
        "input_topic": "audio/resample16k/mic",
        "expected.stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/features/pitch/frames",
        "feature.n_fft": 320,
        "feature.hop_length": 160,
        "feature.f_min_hz": 80.0,
    }
    assert analysis_nodes["fa_tempo"]["enable"] is False
    assert analysis_nodes["fa_tempo"]["package"] == "fa_tempo"
    assert analysis_nodes["fa_tempo"]["params_file"] == (
        "${share:fa_tempo}/config/default.yaml"
    )
    assert analysis_nodes["fa_tempo"]["parameters"] == {
        "input_topic": "audio/resample16k/mic",
        "expected.stream_id": "audio/preprocessed/mono16k",
        "output.stream_id": "audio/features/tempo/frames",
        "feature.n_fft": 320,
        "feature.hop_length": 160,
        "tempo.bpm_min": 60.0,
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


def _flatten_fixture_params(
    params: FixtureParamMapping,
    prefix: str = "",
) -> dict[str, FixtureParamScalar]:
    flattened: dict[str, FixtureParamScalar] = {}
    for key, value in params.items():
        param_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_fixture_params(value, param_key))
            continue
        if isinstance(value, (str, int, float, bool)):
            flattened[param_key] = value
            continue
        raise AssertionError(f"unsupported fixture parameter type for {param_key}")
    return flattened


def _load_fixture_params(
    fixture_name: str,
    node_name: str,
) -> dict[str, FixtureParamScalar]:
    raw = yaml.safe_load((FIXTURE_DIR / fixture_name).read_text(encoding="utf-8"))
    return _flatten_fixture_params(raw[node_name]["ros__parameters"])
