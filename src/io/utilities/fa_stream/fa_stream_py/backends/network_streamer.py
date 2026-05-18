from __future__ import annotations

from dataclasses import dataclass
import shutil
import signal
import subprocess


@dataclass(frozen=True)
class NetworkStreamerConfig:
    ffmpeg_path: str
    output_url: str
    audio_codec: str
    bitrate: str
    container_format: str
    content_type: str
    loglevel: str


@dataclass(frozen=True)
class AudioStreamFormat:
    sample_rate: int
    channels: int
    encoding: str
    bit_depth: int
    layout: str


class NetworkStreamerBackend:
    def __init__(self, config: NetworkStreamerConfig) -> None:
        self._config = _validated_config(config)
        self._process: subprocess.Popen[bytes] | None = None
        self._format: AudioStreamFormat | None = None

    def ensure_started(self, audio_format: AudioStreamFormat) -> None:
        _validate_audio_format(audio_format)
        if self._process is None:
            self._start(audio_format)
            return
        if self._format != audio_format:
            current = self._format
            raise RuntimeError(
                "Audio stream format changed during stream: "
                f"{current} -> {audio_format}"
            )

    def write(self, data: bytes) -> None:
        if not data:
            raise RuntimeError("audio chunk data is required")
        if self._process is None or self._process.stdin is None:
            raise RuntimeError("network streamer backend is not started")
        try:
            self._process.stdin.write(data)
        except BrokenPipeError as exc:
            self.close()
            raise RuntimeError("ffmpeg pipe closed unexpectedly") from exc

    def close(self) -> None:
        if self._process is None:
            return

        process = self._process
        self._process = None
        self._format = None

        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass

        try:
            process.send_signal(signal.SIGINT)
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait(timeout=2.0)

    def _start(self, audio_format: AudioStreamFormat) -> None:
        command = _build_ffmpeg_command(self._config, audio_format)
        try:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as exc:
            self._process = None
            raise RuntimeError(f"failed to start ffmpeg: {exc}") from exc
        self._format = audio_format


def _validated_config(config: NetworkStreamerConfig) -> NetworkStreamerConfig:
    values = (
        ("ffmpeg_path", config.ffmpeg_path),
        ("output_url", config.output_url),
        ("audio_codec", config.audio_codec),
        ("bitrate", config.bitrate),
        ("container_format", config.container_format),
        ("content_type", config.content_type),
        ("loglevel", config.loglevel),
    )
    for name, value in values:
        if not value.strip():
            raise RuntimeError(f"{name} is required")
    if shutil.which(config.ffmpeg_path) is None:
        raise RuntimeError(f"ffmpeg_path is not executable: {config.ffmpeg_path}")
    return NetworkStreamerConfig(
        ffmpeg_path=config.ffmpeg_path.strip(),
        output_url=config.output_url.strip(),
        audio_codec=config.audio_codec.strip(),
        bitrate=config.bitrate.strip(),
        container_format=config.container_format.strip(),
        content_type=config.content_type.strip(),
        loglevel=config.loglevel.strip(),
    )


def _validate_audio_format(audio_format: AudioStreamFormat) -> None:
    if audio_format.layout != "interleaved":
        raise RuntimeError(
            f"Only interleaved AudioFrame layout is supported: {audio_format.layout}"
        )
    if audio_format.encoding != "PCM16LE":
        raise RuntimeError(
            f"Only PCM16LE AudioFrame encoding is supported: {audio_format.encoding}"
        )
    if audio_format.bit_depth != 16:
        raise RuntimeError(f"Only 16-bit PCM is supported: {audio_format.bit_depth}")
    if audio_format.sample_rate <= 0:
        raise RuntimeError("sample_rate must be > 0")
    if audio_format.channels <= 0:
        raise RuntimeError("channels must be > 0")


def _build_ffmpeg_command(
    config: NetworkStreamerConfig,
    audio_format: AudioStreamFormat,
) -> list[str]:
    return [
        config.ffmpeg_path,
        "-loglevel",
        config.loglevel,
        "-f",
        "s16le",
        "-ar",
        str(audio_format.sample_rate),
        "-ac",
        str(audio_format.channels),
        "-i",
        "pipe:0",
        "-c:a",
        config.audio_codec,
        "-b:a",
        config.bitrate,
        "-content_type",
        config.content_type,
        "-f",
        config.container_format,
        config.output_url,
    ]
