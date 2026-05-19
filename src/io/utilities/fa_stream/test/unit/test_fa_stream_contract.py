from pathlib import Path

import pytest
import yaml

from fa_stream_py.backends.network_streamer import (
    AudioStreamFormat,
    _validate_audio_format,
)


PACKAGE_ROOT = Path(__file__).parents[2]


def test_default_config_defines_required_stream_parameters() -> None:
    config_path = PACKAGE_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    params = config["fa_stream"]["ros__parameters"]

    assert params["input_topic"] == "audio/frame"
    assert params["ffmpeg_path"] == "ffmpeg"
    assert params["output_url"] == ""
    assert params["audio_codec"] == "libmp3lame"
    assert params["bitrate"] == "128k"
    assert params["container_format"] == "mp3"
    assert params["content_type"] == "audio/mpeg"
    assert params["loglevel"] == "warning"
def test_network_streamer_backend_rejects_non_pcm16le_encoding() -> None:
    with pytest.raises(
        RuntimeError,
        match="Only PCM16LE audio stream encoding is supported: PCM16BE",
    ):
        _validate_audio_format(
            AudioStreamFormat(
                sample_rate=48000,
                channels=1,
                encoding="PCM16BE",
                bit_depth=16,
                layout="interleaved",
            )
        )


def test_network_streamer_backend_accepts_pcm16le_s16le_contract() -> None:
    _validate_audio_format(
        AudioStreamFormat(
            sample_rate=48000,
            channels=1,
            encoding="PCM16LE",
            bit_depth=16,
            layout="interleaved",
        )
    )
