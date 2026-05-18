from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_tempo_py.backends.onset_autocorrelation import (
    InternalOnsetAutocorrelationBackend,
    TempoConfig,
)


PACKAGE_ROOT = Path(__file__).parents[2]


def _config() -> TempoConfig:
    return TempoConfig(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        method="onset_autocorrelation",
        bpm_min=60.0,
        bpm_max=180.0,
        confidence_threshold=0.2,
    )


def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_tempo"]["ros__parameters"]

    assert params["backend.name"] == "internal_onset_autocorrelation"
    assert params["expected.sample_rate"] == 16000
    assert params["expected.channels"] == 1
    assert params["expected.encoding"] == "FLOAT32LE"
    assert params["expected.bit_depth"] == 32
    assert params["expected.layout"] == "interleaved"
    assert params["feature.n_fft"] == 400
    assert params["feature.hop_length"] == 160
    assert params["feature.method"] == "onset_autocorrelation"
    assert params["tempo.bpm_min"] == 60.0
    assert params["tempo.bpm_max"] == 180.0
    assert params["tempo.confidence_threshold"] == 0.2


def test_backend_estimates_tempo_from_periodic_impulses() -> None:
    backend = InternalOnsetAutocorrelationBackend(_config())
    samples = np.zeros(16400, dtype=np.float32)
    for start in range(400, 16400, 8000):
        samples[start : start + 80] = np.hanning(80).astype(np.float32)

    result = backend.compute(samples)

    assert result.frame_count == 101
    assert result.frame_times_sec.shape == (101,)
    assert result.onset_envelope.shape == (101,)
    assert result.beats.shape == (101,)
    assert result.frame_times_sec.dtype == np.float32
    assert result.onset_envelope.dtype == np.float32
    assert result.beats.dtype == np.bool_
    assert np.all(np.isfinite(result.onset_envelope))
    assert result.tempo_detected is True
    assert result.beat_period_frames == 50
    assert result.tempo_bpm == pytest.approx(120.0, abs=1.0)
    assert result.confidence >= 0.2
    assert np.count_nonzero(result.beats) >= 2


def test_backend_silence_has_no_tempo_detection() -> None:
    backend = InternalOnsetAutocorrelationBackend(_config())
    result = backend.compute(np.zeros(400, dtype=np.float32))

    assert result.frame_count == 1
    assert np.allclose(result.frame_times_sec, [0.0125], rtol=0.0, atol=1.0e-7)
    assert np.allclose(result.onset_envelope, 0.0, rtol=0.0, atol=0.0)
    assert result.beats.tolist() == [False]
    assert result.tempo_detected is False
    assert result.tempo_bpm == 0.0
    assert result.confidence == 0.0
    assert result.beat_period_frames == 0


def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalOnsetAutocorrelationBackend(_config())

    with pytest.raises(ValueError, match="float32"):
        backend.compute(np.zeros(400, dtype=np.float64))
    with pytest.raises(ValueError, match="one-dimensional"):
        backend.compute(np.zeros((1, 400), dtype=np.float32))
    with pytest.raises(ValueError, match="normalized"):
        backend.compute(np.full(400, 2.0, dtype=np.float32))
    with pytest.raises(ValueError, match="feature.n_fft"):
        backend.compute(np.zeros(399, dtype=np.float32))
    with pytest.raises(ValueError, match="align"):
        backend.compute(np.zeros(401, dtype=np.float32))


def test_backend_config_rejects_invalid_feature_contract() -> None:
    with pytest.raises(RuntimeError, match="feature.method must be onset_autocorrelation"):
        InternalOnsetAutocorrelationBackend(
            TempoConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="beat_tracker",
                bpm_min=60.0,
                bpm_max=180.0,
                confidence_threshold=0.2,
            )
        )
    with pytest.raises(RuntimeError, match="tempo.bpm_max must be finite"):
        InternalOnsetAutocorrelationBackend(
            TempoConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="onset_autocorrelation",
                bpm_min=180.0,
                bpm_max=60.0,
                confidence_threshold=0.2,
            )
        )
    with pytest.raises(RuntimeError, match="tempo.bpm_max is too high"):
        InternalOnsetAutocorrelationBackend(
            TempoConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="onset_autocorrelation",
                bpm_min=60.0,
                bpm_max=7000.0,
                confidence_threshold=0.2,
            )
        )
    with pytest.raises(RuntimeError, match="tempo.confidence_threshold"):
        InternalOnsetAutocorrelationBackend(
            TempoConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="onset_autocorrelation",
                bpm_min=60.0,
                bpm_max=180.0,
                confidence_threshold=1.1,
            )
        )


def test_backend_is_ros_free_and_not_ai_runtime() -> None:
    backend_path = PACKAGE_ROOT / "fa_tempo_py" / "backends" / "onset_autocorrelation.py"
    node_path = PACKAGE_ROOT / "fa_tempo_py" / "tempo_node.py"
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_text = backend_path.read_text(encoding="utf-8")
    node_text = node_path.read_text(encoding="utf-8")

    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "TempoFrame" not in backend_text
    assert "InternalOnsetAutocorrelationBackend" in node_text
    assert "VAD / KWS / ASR / Turn Detector" in spec
    assert "非 AI analysis node" in spec
    assert "model runtime は持たず" in readme
    assert "resample" not in backend_text


def test_node_requires_explicit_ros_parameters_and_binds_stream_identity() -> None:
    node_path = PACKAGE_ROOT / "fa_tempo_py" / "tempo_node.py"
    node_text = node_path.read_text(encoding="utf-8")

    assert "ParameterUninitializedException" in node_text
    assert "Parameter.Type.STRING" in node_text
    assert 'declare_parameter("input_topic", "audio/features/input")' not in node_text
    assert 'declare_parameter("feature.n_fft", 400)' not in node_text
    assert "msg.stream_id != self.input_topic" in node_text
    assert "except ValueError as exc:" in node_text
    assert "except RuntimeError as exc:" in node_text
    assert "raise" in node_text


def test_system_sample_aligns_tempo_window_with_default_mic_chunk() -> None:
    sample_path = (
        PACKAGE_ROOT.parents[2]
        / "system"
        / "fluent_audio_system"
        / "config"
        / "fluent_audio_system.sample.yaml"
    )
    config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    analysis_group = next(group for group in config["groups"] if group["id"] == "analysis")
    tempo_node = next(node for node in analysis_group["nodes"] if node["id"] == "fa_tempo")

    assert tempo_node["parameters"]["input_topic"] == "audio/resample16k/mic"
    assert tempo_node["parameters"]["feature.n_fft"] == 320
    assert tempo_node["parameters"]["feature.hop_length"] == 160
    assert tempo_node["parameters"]["tempo.bpm_min"] == 60.0
