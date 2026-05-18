from pathlib import Path

import numpy as np
import pytest
import yaml

from fa_pitch_py.backends.autocorrelation import InternalAutocorrelationBackend, PitchConfig


PACKAGE_ROOT = Path(__file__).parents[2]


def _config() -> PitchConfig:
    return PitchConfig(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        method="autocorrelation",
        f_min_hz=80.0,
        f_max_hz=800.0,
        confidence_threshold=0.3,
    )


def test_default_config_uses_internal_backend_and_canonical_audio() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_pitch"]["ros__parameters"]

    assert params["backend.name"] == "internal_autocorrelation"
    assert params["expected.sample_rate"] == 16000
    assert params["expected.channels"] == 1
    assert params["expected.encoding"] == "FLOAT32LE"
    assert params["expected.bit_depth"] == 32
    assert params["expected.layout"] == "interleaved"
    assert params["feature.n_fft"] == 400
    assert params["feature.hop_length"] == 160
    assert params["feature.method"] == "autocorrelation"
    assert params["feature.f_min_hz"] == 80.0
    assert params["feature.f_max_hz"] == 800.0
    assert params["detector.confidence_threshold"] == 0.3


def test_backend_estimates_pitch_for_sine_wave() -> None:
    backend = InternalAutocorrelationBackend(_config())
    time = np.arange(1520, dtype=np.float32) / np.float32(16000.0)
    samples = (0.4 * np.sin(np.float32(2.0 * np.pi * 200.0) * time)).astype(np.float32)

    result = backend.compute(samples)

    assert result.frame_count == 8
    assert result.frame_times_sec.shape == (8,)
    assert result.frequencies_hz.shape == (8,)
    assert result.confidence.shape == (8,)
    assert result.voiced.shape == (8,)
    assert result.frequencies_hz.dtype == np.float32
    assert result.confidence.dtype == np.float32
    assert result.voiced.dtype == np.bool_
    assert np.all(np.isfinite(result.frequencies_hz))
    assert np.all(np.isfinite(result.confidence))
    assert np.all(result.voiced)
    assert np.allclose(result.frequencies_hz, 200.0, rtol=0.0, atol=5.0)


def test_backend_silence_is_unvoiced_zero_pitch() -> None:
    backend = InternalAutocorrelationBackend(_config())
    result = backend.compute(np.zeros(400, dtype=np.float32))

    assert result.frame_count == 1
    assert np.allclose(result.frequencies_hz, 0.0, rtol=0.0, atol=0.0)
    assert np.allclose(result.confidence, 0.0, rtol=0.0, atol=0.0)
    assert result.voiced.tolist() == [False]
    assert np.allclose(result.frame_times_sec, [0.0125], rtol=0.0, atol=1.0e-7)


def test_backend_rejects_non_canonical_audio_without_hidden_conversion() -> None:
    backend = InternalAutocorrelationBackend(_config())

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
    with pytest.raises(RuntimeError, match="feature.method must be autocorrelation"):
        InternalAutocorrelationBackend(
            PitchConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="yin",
                f_min_hz=80.0,
                f_max_hz=800.0,
                confidence_threshold=0.3,
            )
        )
    with pytest.raises(RuntimeError, match="feature.f_max_hz must be <= Nyquist"):
        InternalAutocorrelationBackend(
            PitchConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="autocorrelation",
                f_min_hz=80.0,
                f_max_hz=9000.0,
                confidence_threshold=0.3,
            )
        )
    with pytest.raises(RuntimeError, match="feature.f_min_hz requires feature.n_fft"):
        InternalAutocorrelationBackend(
            PitchConfig(
                sample_rate=16000,
                n_fft=120,
                hop_length=60,
                method="autocorrelation",
                f_min_hz=80.0,
                f_max_hz=800.0,
                confidence_threshold=0.3,
            )
        )
    with pytest.raises(RuntimeError, match="detector.confidence_threshold"):
        InternalAutocorrelationBackend(
            PitchConfig(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                method="autocorrelation",
                f_min_hz=80.0,
                f_max_hz=800.0,
                confidence_threshold=1.1,
            )
        )


def test_backend_is_ros_free_and_not_ai_runtime() -> None:
    backend_path = PACKAGE_ROOT / "fa_pitch_py" / "backends" / "autocorrelation.py"
    node_path = PACKAGE_ROOT / "fa_pitch_py" / "pitch_node.py"
    readme = (PACKAGE_ROOT / "README.md").read_text(encoding="utf-8")
    spec = (PACKAGE_ROOT / "docs" / "仕様書.md").read_text(encoding="utf-8")
    backend_text = backend_path.read_text(encoding="utf-8")
    node_text = node_path.read_text(encoding="utf-8")

    assert "import rclpy" not in backend_text
    assert "fa_interfaces" not in backend_text
    assert "PitchFrame" not in backend_text
    assert "InternalAutocorrelationBackend" in node_text
    assert "VAD / KWS / ASR / Turn Detector" in spec
    assert "非 AI analysis node" in spec
    assert "model runtime は持たず" in readme
    assert "resample" not in backend_text


def test_node_requires_explicit_ros_parameters_and_binds_stream_identity() -> None:
    node_path = PACKAGE_ROOT / "fa_pitch_py" / "pitch_node.py"
    node_text = node_path.read_text(encoding="utf-8")

    assert "ParameterUninitializedException" in node_text
    assert "Parameter.Type.STRING" in node_text
    assert 'declare_parameter("input_topic", "audio/features/input")' not in node_text
    assert 'declare_parameter("feature.n_fft", 400)' not in node_text
    assert "msg.stream_id != self.input_topic" in node_text
    assert "except ValueError as exc:" in node_text
    assert "except RuntimeError as exc:" in node_text
    assert "raise" in node_text


def test_system_sample_aligns_pitch_window_with_default_mic_chunk() -> None:
    sample_path = (
        PACKAGE_ROOT.parents[2]
        / "system"
        / "fluent_audio_system"
        / "config"
        / "fluent_audio_system.sample.yaml"
    )
    config = yaml.safe_load(sample_path.read_text(encoding="utf-8"))
    analysis_group = next(group for group in config["groups"] if group["id"] == "analysis")
    pitch_node = next(node for node in analysis_group["nodes"] if node["id"] == "fa_pitch")

    assert pitch_node["parameters"]["input_topic"] == "audio/resample16k/mic"
    assert pitch_node["parameters"]["feature.n_fft"] == 320
    assert pitch_node["parameters"]["feature.hop_length"] == 160
    assert pitch_node["parameters"]["feature.f_min_hz"] == 80.0
