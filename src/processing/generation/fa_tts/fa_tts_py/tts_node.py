#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import logging
from array import array
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from fa_interfaces.msg import AudioFrame
from fa_interfaces.srv import Speak
from fa_tts_py.backends.base import SynthesizedAudio
from fa_tts_py.backends.factory import build_tts_backend


@dataclass(frozen=True)
class CacheMetadata:
    encoding: str
    sample_rate: int
    channels: int
    bit_depth: int


class FaTtsNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_tts")

        self.declare_parameter("backend.name", "")
        self.declare_parameter("backend.openjtalk_dict_dir", "")
        self.declare_parameter("default_voice", "")
        self.declare_parameter("output_topic", "audio/tts/frame")
        self.declare_parameter("cache_dir", "")

        self.backend_name = self.get_parameter("backend.name").get_parameter_value().string_value
        self.openjtalk_dict_dir = (
            self.get_parameter("backend.openjtalk_dict_dir").get_parameter_value().string_value
        )
        self.default_voice = self.get_parameter("default_voice").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        if not self.output_topic:
            raise RuntimeError("output_topic is required")
        self.backend = build_tts_backend(
            self.backend_name,
            openjtalk_dict_dir=self.openjtalk_dict_dir,
        )

        cache_dir_param = self.get_parameter("cache_dir").get_parameter_value().string_value
        self.cache_dir = Path(cache_dir_param).expanduser() if cache_dir_param else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # RELIABLE QoS でメッセージドロップを防ぐ
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.durability = DurabilityPolicy.VOLATILE
        qos.history = HistoryPolicy.KEEP_LAST

        self.tts_pub = self.create_publisher(AudioFrame, self.output_topic, qos)
        self.srv = self.create_service(Speak, "speak", self.handle_speak)

        self.cache: dict[str, SynthesizedAudio] = {}
        self.get_logger().info("Starting FA TTS node with backend.name=%s", self.backend.name)

    def handle_speak(self, request: Speak.Request, response: Speak.Response) -> Speak.Response:
        text = request.text.strip()
        self.get_logger().info(f"TTS request: text={text[:50]}...")

        if not text:
            response.success = False
            response.message = "text is empty"
            self.get_logger().warn("TTS rejected: empty text")
            return response
        if request.play:
            response.success = False
            response.message = "request.play is not supported by fa_tts; route audio/tts/frame through fa_mix/fa_out"
            self.get_logger().warn("TTS rejected: playback routing requested")
            return response
        if request.volume_db != 0.0:
            response.success = False
            response.message = "request.volume_db is not supported by fa_tts; use fa_mix or a dynamics node"
            self.get_logger().warn("TTS rejected: playback gain requested")
            return response

        request_stamp = self.get_clock().now().to_msg()

        voice_id = request.voice_id or self.default_voice
        cache_key = request.cache_key or self.make_cache_key(text, voice_id)
        try:
            self.validate_cache_key(cache_key)
        except ValueError as exc:
            response.success = False
            response.message = str(exc)
            self.get_logger().warn(f"TTS rejected: {exc}")
            return response

        cached = self.cache.get(cache_key)
        if cached is None and self.cache_dir:
            cached = self.load_cache_from_disk(cache_key)

        if cached is None:
            self.get_logger().info("TTS cache miss, synthesizing...")
            try:
                cached = self.backend.synthesize(text, voice_id)
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f"TTS failed: {exc}")
                response.success = False
                response.message = f"TTS failed: {exc}"
                return response
            if self.cache_dir:
                try:
                    self.write_cache_to_disk(cache_key, cached)
                except RuntimeError as exc:
                    response.success = False
                    response.message = str(exc)
                    self.get_logger().error(str(exc))
                    return response
            self.cache[cache_key] = cached
            self.get_logger().info(f"TTS synthesized: {len(cached.audio_bytes)} bytes, {cached.sample_rate}Hz")
        else:
            self.get_logger().info("TTS cache hit")

        frame = self.build_frame(cached, stamp=request_stamp)
        response.frame = frame
        response.success = True
        response.message = "ok"

        self.tts_pub.publish(frame)
        response.played = False

        return response

    def build_frame(
        self,
        cached: SynthesizedAudio,
        *,
        stamp=None,
        epoch: int | None = None,
    ) -> AudioFrame:
        frame = AudioFrame()
        frame.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        frame.source_id = "fa_tts"
        frame.stream_id = self.output_topic
        frame.encoding = cached.encoding
        frame.sample_rate = cached.sample_rate
        frame.channels = cached.channels
        frame.bit_depth = cached.bit_depth
        frame.layout = "interleaved"
        frame.data = array('B', cached.audio_bytes)
        frame.epoch = int(epoch) if epoch is not None else 0
        return frame

    def make_cache_key(self, text: str, voice_id: str) -> str:
        digest = hashlib.sha1()  # nosec B303
        digest.update(text.encode("utf-8"))
        digest.update(voice_id.encode("utf-8"))
        return digest.hexdigest()

    @staticmethod
    def validate_cache_key(cache_key: str) -> None:
        if len(cache_key) != 40:
            raise ValueError("cache_key must be 40 lowercase hex characters")
        for character in cache_key:
            if character not in "0123456789abcdef":
                raise ValueError("cache_key must be 40 lowercase hex characters")

    def cache_file_path(self, cache_key: str) -> Path | None:
        self.validate_cache_key(cache_key)
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{cache_key}.pcm"

    def load_cache_from_disk(self, cache_key: str) -> SynthesizedAudio | None:
        path = self.cache_file_path(cache_key)
        if not path or not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            self.get_logger().warning("Failed to read cache file %s: %s", path, exc)
            return None
        meta_path = path.with_suffix(".meta")
        if not meta_path.exists():
            self.get_logger().warning("Ignoring TTS cache without metadata: %s", path)
            return None
        try:
            metadata = self.parse_cache_metadata(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            self.get_logger().warning("Ignoring invalid TTS cache metadata %s: %s", meta_path, exc)
            return None
        return SynthesizedAudio(
            data,
            metadata.encoding,
            metadata.sample_rate,
            metadata.channels,
            metadata.bit_depth,
        )

    @staticmethod
    def parse_cache_metadata(text: str) -> CacheMetadata:
        numeric_values: dict[str, int] = {}
        encoding_value: str | None = None
        for line in text.strip().split("\n"):
            if not line:
                continue
            key, _, value = line.partition(":")
            if not value:
                raise ValueError("cache metadata line must be key:value")
            key = key.strip()
            if key == "encoding":
                encoding_value = value.strip()
                continue
            if key not in ("sample_rate", "channels", "bit_depth"):
                raise ValueError(f"unsupported cache metadata key: {key}")
            numeric_values[key] = int(value.strip())
        missing = {"sample_rate", "channels", "bit_depth"}.difference(numeric_values)
        if missing:
            raise ValueError(f"cache metadata missing keys: {sorted(missing)}")
        if not encoding_value:
            raise ValueError("cache metadata missing encoding")
        if encoding_value != "FLOAT32LE":
            raise ValueError(f"unsupported cache metadata encoding: {encoding_value}")
        if (
            numeric_values["sample_rate"] <= 0
            or numeric_values["channels"] <= 0
            or numeric_values["bit_depth"] <= 0
        ):
            raise ValueError("cache metadata numeric values must be > 0")
        if numeric_values["bit_depth"] != 32:
            raise ValueError("cache metadata bit_depth must be 32 for FLOAT32LE")
        return CacheMetadata(
            encoding_value,
            numeric_values["sample_rate"],
            numeric_values["channels"],
            numeric_values["bit_depth"],
        )

    def write_cache_to_disk(self, cache_key: str, cached: SynthesizedAudio) -> None:
        path = self.cache_file_path(cache_key)
        if not path:
            return
        try:
            path.write_bytes(cached.audio_bytes)
            meta = "\n".join([
                f"encoding:{cached.encoding}",
                f"sample_rate:{cached.sample_rate}",
                f"channels:{cached.channels}",
                f"bit_depth:{cached.bit_depth}",
            ])
            path.with_suffix(".meta").write_text(meta)
        except OSError as exc:
            raise RuntimeError(f"failed to write TTS cache file {path}: {exc}") from exc


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FaTtsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
