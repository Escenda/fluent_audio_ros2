#!/usr/bin/env python3
import hashlib
import logging
import os
import sys
from array import array
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from std_msgs.msg import Empty

from fa_interfaces.msg import AudioFrame
from fa_interfaces.srv import Speak

# Set pyopenjtalk dictionary directory to user's home directory
os.environ.setdefault("OPEN_JTALK_DICT_DIR", str(Path.home() / ".pyopenjtalk"))

# Disable tqdm progress bar for pyopenjtalk downloads
os.environ["TQDM_DISABLE"] = "1"

try:
    # Suppress download progress output during pyopenjtalk import
    _original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import pyopenjtalk
    sys.stderr.close()
    sys.stderr = _original_stderr
except ImportError as exc:  # pragma: no cover - import guard
    sys.stderr = _original_stderr
    pyopenjtalk = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class CachedAudio:
    __slots__ = ("audio_bytes", "sample_rate", "channels", "bit_depth", "rms", "peak")

    def __init__(self, audio_bytes: bytes, sample_rate: int, channels: int, bit_depth: int,
                 rms: float, peak: float) -> None:
        self.audio_bytes = audio_bytes
        self.sample_rate = sample_rate
        self.channels = channels
        self.bit_depth = bit_depth
        self.rms = rms
        self.peak = peak


class FaTtsNode(Node):
    def __init__(self) -> None:
        super().__init__("fa_tts")
        self.get_logger().info("Starting FA TTS node (pyopenjtalk backend)")
        if pyopenjtalk is None:
            raise RuntimeError(f"pyopenjtalk is not available: {_IMPORT_ERROR}") from _IMPORT_ERROR

        self.declare_parameter("default_voice", "")
        self.declare_parameter("output_topic", "audio/tts/frame")
        self.declare_parameter("playback_topic", "audio/output/frame")
        self.declare_parameter("use_playback_topic", True)
        self.declare_parameter("stop_topic", "audio/output/stop")
        self.declare_parameter("cache_dir", "")
        self.declare_parameter("default_volume_db", 0.0)

        self.default_voice = self.get_parameter("default_voice").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value
        self.playback_topic = self.get_parameter("playback_topic").get_parameter_value().string_value
        self.publish_playback = self.get_parameter("use_playback_topic").get_parameter_value().bool_value
        self.stop_topic = self.get_parameter("stop_topic").get_parameter_value().string_value
        self.default_volume_db = self.get_parameter("default_volume_db").get_parameter_value().double_value

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
        self.play_pub = self.create_publisher(AudioFrame, self.playback_topic, qos)
        self.srv = self.create_service(Speak, "speak", self.handle_speak)

        # Barge-in/stop 後に古い音声が鳴らないよう、再生世代(epoch)を管理する。
        # epoch は AudioFrame.epoch に埋め込み、audio_output 側で stop 時に世代を進めて破棄する。
        self.playback_epoch = 1
        self.stop_sub = self.create_subscription(Empty, self.stop_topic, self.on_stop, qos)

        self.cache: Dict[str, CachedAudio] = {}

    def on_stop(self, _msg: Empty) -> None:
        """音声停止を受けたら再生世代を進める。"""
        self.playback_epoch += 1

    def handle_speak(self, request: Speak.Request, response: Speak.Response) -> Speak.Response:
        text = request.text.strip()
        self.get_logger().info(f"TTS request: text={text[:50]}... play={request.play}")

        if not text:
            response.success = False
            response.message = "text is empty"
            self.get_logger().warn("TTS rejected: empty text")
            return response

        # stop 後に遅れて publish される旧音声を確実に drop できるよう、
        # stamp/epoch はリクエスト受領時点で確定させて全フレームに適用する。
        request_stamp = self.get_clock().now().to_msg()
        request_epoch = self.playback_epoch

        voice_id = request.voice_id or self.default_voice
        volume_db = request.volume_db if request.volume_db != 0.0 else self.default_volume_db
        cache_key = request.cache_key or self.make_cache_key(text, voice_id, volume_db)

        cached = self.cache.get(cache_key)
        if cached is None and self.cache_dir:
            cached = self.load_cache_from_disk(cache_key)

        if cached is None:
            self.get_logger().info("TTS cache miss, synthesizing...")
            try:
                cached = self.synthesize(text, voice_id, volume_db)
            except Exception as exc:  # pylint: disable=broad-except
                self.get_logger().error(f"TTS failed: {exc}")
                response.success = False
                response.message = f"TTS failed: {exc}"
                return response
            self.cache[cache_key] = cached
            if self.cache_dir:
                self.write_cache_to_disk(cache_key, cached)
            self.get_logger().info(f"TTS synthesized: {len(cached.audio_bytes)} bytes, {cached.sample_rate}Hz")
        else:
            self.get_logger().info("TTS cache hit")

        frame = self.build_frame(cached, stamp=request_stamp, epoch=request_epoch)
        response.frame = frame
        response.success = True
        response.message = "ok"

        self.tts_pub.publish(frame)
        response.played = False
        if request.play and self.publish_playback:
            self.play_pub.publish(frame)
            response.played = True
            audio_duration = len(cached.audio_bytes) / 2 / cached.sample_rate  # 16bit mono
            self.get_logger().info(f"TTS published to playback: {audio_duration:.2f}s")
        else:
            self.get_logger().warn(f"TTS NOT published to playback: play={request.play}, publish_playback={self.publish_playback}")

        return response

    def synthesize(self, text: str, voice_id: str, volume_db: float) -> CachedAudio:
        options = {}
        if voice_id:
            options["voice"] = voice_id
        wav, sample_rate = pyopenjtalk.tts(text, **options)
        waveform = np.asarray(wav, dtype=np.float32)

        # pyopenjtalk が返す波形は float64 だが、環境によっては
        # -32768〜32767 相当のスケールで返ってくることがある。
        # そのまま[-1,1]にクリップすると全区間が潰れてノイズ化するため、
        # 振幅が1.0を超える場合は16bitスケールとして正規化する。
        if waveform.size:
            abs_max = float(np.max(np.abs(waveform)))
            if abs_max > 1.0:
                waveform /= 32768.0
        else:
            abs_max = 1.0
        if volume_db != 0.0:
            gain = float(10.0 ** (volume_db / 20.0))
            waveform *= gain
        waveform = np.clip(waveform, -1.0, 1.0)
        pcm = (waveform * 32767.0).astype(np.int16)
        audio_bytes = pcm.tobytes()
        float_samples = pcm.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(np.square(float_samples)))) if float_samples.size else 0.0
        peak = float(np.max(np.abs(float_samples))) if float_samples.size else 0.0
        return CachedAudio(audio_bytes, int(sample_rate), 1, 16, rms, peak)

    def build_frame(self, cached: CachedAudio, *, stamp=None, epoch: int | None = None) -> AudioFrame:
        frame = AudioFrame()
        frame.header.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        frame.encoding = "PCM16LE"
        frame.sample_rate = cached.sample_rate
        frame.channels = cached.channels
        frame.bit_depth = cached.bit_depth
        frame.rms = cached.rms
        frame.peak = cached.peak
        frame.vad = False
        frame.data = array('B', cached.audio_bytes)
        frame.epoch = int(epoch) if epoch is not None else 0
        return frame

    def make_cache_key(self, text: str, voice_id: str, volume_db: float) -> str:
        digest = hashlib.sha1()  # nosec B303
        digest.update(text.encode("utf-8"))
        digest.update(voice_id.encode("utf-8"))
        digest.update(str(volume_db).encode("utf-8"))
        return digest.hexdigest()

    def cache_file_path(self, cache_key: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{cache_key}.pcm"

    def load_cache_from_disk(self, cache_key: str) -> Optional[CachedAudio]:
        path = self.cache_file_path(cache_key)
        if not path or not path.exists():
            return None
        try:
            data = path.read_bytes()
        except OSError as exc:
            self.get_logger().warning("Failed to read cache file %s: %s", path, exc)
            return None
        meta_path = path.with_suffix(".meta")
        sample_rate = 48000
        channels = 1
        bit_depth = 16
        rms = 0.0
        peak = 0.0
        if meta_path.exists():
            try:
                content = meta_path.read_text().strip().split("\n")
                for line in content:
                    if not line:
                        continue
                    key, _, value = line.partition(":")
                    key = key.strip()
                    value = value.strip()
                    if key == "sample_rate":
                        sample_rate = int(value)
                    elif key == "channels":
                        channels = int(value)
                    elif key == "bit_depth":
                        bit_depth = int(value)
                    elif key == "rms":
                        rms = float(value)
                    elif key == "peak":
                        peak = float(value)
            except OSError as exc:
                self.get_logger().warning("Failed to parse cache metadata %s: %s", meta_path, exc)
        return CachedAudio(data, sample_rate, channels, bit_depth, rms, peak)

    def write_cache_to_disk(self, cache_key: str, cached: CachedAudio) -> None:
        path = self.cache_file_path(cache_key)
        if not path:
            return
        try:
            path.write_bytes(cached.audio_bytes)
            meta = "\n".join([
                f"sample_rate:{cached.sample_rate}",
                f"channels:{cached.channels}",
                f"bit_depth:{cached.bit_depth}",
                f"rms:{cached.rms}",
                f"peak:{cached.peak}",
            ])
            path.with_suffix(".meta").write_text(meta)
        except OSError as exc:
            self.get_logger().warning("Failed to write cache file %s: %s", path, exc)


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
