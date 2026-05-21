from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from fa_asr_py.backends.base import AsrAudioPayload


ERROR_TIME_RANGE_UNRESOLVED = "time_range_unresolved"
ERROR_WINDOW_NOT_FOUND = "window_not_found"
ERROR_RANGE_OUTSIDE_WINDOW = "range_outside_window"
ERROR_RANGE_NOT_CONTINUOUS = "range_not_continuous"

_NSEC_PER_SEC = 1_000_000_000
_NSEC_PER_MSEC = 1_000_000
_NUMERIC_RANGE_PATTERN = re.compile(r"^([0-9]+)\.\.([0-9]+)$")


@dataclass(frozen=True)
class NumericTimeRange:
    start_unix_ns: int
    end_unix_ns: int


@dataclass(frozen=True)
class TimelineSlice:
    payload: AsrAudioPayload
    time_range: NumericTimeRange

    @property
    def samples(self) -> np.ndarray:
        return self.payload.float32_samples()

    @property
    def sample_rate(self) -> int:
        return self.payload.sample_rate_hz

    @property
    def sample_count(self) -> int:
        return self.payload.sample_count


@dataclass(frozen=True)
class TimelineFrame:
    start_unix_ns: int
    floor_end_unix_ns: int
    end_unix_ns: int
    payload: AsrAudioPayload


class TimelineRangeError(ValueError):
    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code


def parse_numeric_time_range(time_range_spec: str) -> NumericTimeRange:
    match = _NUMERIC_RANGE_PATTERN.fullmatch(time_range_spec)
    if match is None:
        raise TimelineRangeError(
            ERROR_TIME_RANGE_UNRESOLVED,
            "time_range_spec must be <start_unix_ns>..<end_unix_ns>",
        )
    start_unix_ns = int(match.group(1))
    end_unix_ns = int(match.group(2))
    if end_unix_ns <= start_unix_ns:
        raise TimelineRangeError(
            ERROR_TIME_RANGE_UNRESOLVED,
            "time_range_spec end_unix_ns must be greater than start_unix_ns",
        )
    return NumericTimeRange(start_unix_ns=start_unix_ns, end_unix_ns=end_unix_ns)


class RollingAsrTimeline:
    def __init__(
        self,
        *,
        sample_rate: int,
        retention_sec: float,
        timestamp_alignment_tolerance_ms: float,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if not np.isfinite(retention_sec) or retention_sec <= 0.0:
            raise ValueError("retention_sec must be finite and greater than zero")
        if (
            not np.isfinite(timestamp_alignment_tolerance_ms)
            or timestamp_alignment_tolerance_ms < 0.0
        ):
            raise ValueError(
                "timestamp_alignment_tolerance_ms must be finite and greater "
                "than or equal to zero"
            )
        retention_ns = int(retention_sec * _NSEC_PER_SEC)
        if retention_ns <= 0:
            raise ValueError("retention_sec is too small to represent in nanoseconds")

        self._sample_rate = int(sample_rate)
        self._retention_ns = retention_ns
        self._timestamp_alignment_tolerance_ns = int(
            round(timestamp_alignment_tolerance_ms * _NSEC_PER_MSEC)
        )
        self._frames: list[TimelineFrame] = []
        self._latest_end_unix_ns = 0
        self._retained_start_unix_ns = 0

    @property
    def retained_start_unix_ns(self) -> int:
        return self._retained_start_unix_ns

    @property
    def latest_end_unix_ns(self) -> int:
        return self._latest_end_unix_ns

    def append(
        self,
        *,
        start_unix_ns: int,
        payload: AsrAudioPayload | None = None,
        samples: np.ndarray | None = None,
    ) -> None:
        if start_unix_ns <= 0:
            raise TimelineRangeError(
                ERROR_TIME_RANGE_UNRESOLVED,
                "AudioFrame header.stamp must resolve to a positive unix nanosecond timestamp",
            )
        audio_payload = self._resolve_append_payload(payload=payload, samples=samples)
        if audio_payload.sample_rate_hz != self._sample_rate:
            raise ValueError("payload sample_rate_hz must match timeline sample_rate")

        start_unix_ns = self._align_adjacent_frame_start(start_unix_ns)
        floor_duration_ns = self._sample_index_to_floor_duration_ns(audio_payload.sample_count)
        ceil_duration_ns = self._sample_index_to_ceil_duration_ns(audio_payload.sample_count)
        floor_end_unix_ns = start_unix_ns + floor_duration_ns
        end_unix_ns = start_unix_ns + ceil_duration_ns

        self._frames.append(
            TimelineFrame(
                start_unix_ns=start_unix_ns,
                floor_end_unix_ns=floor_end_unix_ns,
                end_unix_ns=end_unix_ns,
                payload=audio_payload,
            )
        )
        self._latest_end_unix_ns = max(self._latest_end_unix_ns, end_unix_ns)
        self._retained_start_unix_ns = max(
            0,
            self._latest_end_unix_ns - self._retention_ns,
        )
        self._drop_expired_frames()

    def slice(self, time_range: NumericTimeRange) -> TimelineSlice:
        if not self._frames:
            raise TimelineRangeError(
                ERROR_WINDOW_NOT_FOUND,
                "ASR timeline has no retained audio",
            )
        if time_range.start_unix_ns < self._retained_start_unix_ns:
            raise TimelineRangeError(
                ERROR_RANGE_OUTSIDE_WINDOW,
                "requested start is older than the retained ASR timeline window",
            )
        if time_range.end_unix_ns > self._latest_end_unix_ns:
            raise TimelineRangeError(
                ERROR_RANGE_OUTSIDE_WINDOW,
                "requested end is newer than the retained ASR timeline window",
            )

        pieces: list[bytes] = []
        encoding = self._frames[0].payload.encoding
        channels = self._frames[0].payload.channels
        actual_start_unix_ns: int | None = None
        actual_end_unix_ns: int | None = None
        cursor_unix_ns = time_range.start_unix_ns
        for frame in self._frames:
            if frame.end_unix_ns <= cursor_unix_ns:
                continue
            if frame.start_unix_ns > cursor_unix_ns:
                raise TimelineRangeError(
                    ERROR_RANGE_NOT_CONTINUOUS,
                    "requested ASR timeline range crosses an audio gap",
                )
            if frame.payload.encoding != encoding or frame.payload.channels != channels:
                raise TimelineRangeError(
                    ERROR_RANGE_NOT_CONTINUOUS,
                    "requested ASR timeline range crosses mixed audio payload formats",
                )
            piece_end_unix_ns = min(time_range.end_unix_ns, frame.end_unix_ns)
            frame_start_index = self._floor_sample_index(cursor_unix_ns - frame.start_unix_ns)
            frame_end_index = self._ceil_sample_index(piece_end_unix_ns - frame.start_unix_ns)
            frame_start_index = min(frame_start_index, frame.payload.sample_count)
            frame_end_index = min(frame_end_index, frame.payload.sample_count)

            if frame_start_index < frame_end_index:
                pieces.append(
                    self._slice_payload_bytes(
                        frame.payload,
                        start_sample=frame_start_index,
                        end_sample=frame_end_index,
                    )
                )
                piece_actual_start_unix_ns = (
                    frame.start_unix_ns
                    + self._sample_index_to_floor_duration_ns(frame_start_index)
                )
                piece_actual_end_unix_ns = (
                    frame.start_unix_ns
                    + self._sample_index_to_ceil_duration_ns(frame_end_index)
                )
                if actual_start_unix_ns is None:
                    actual_start_unix_ns = piece_actual_start_unix_ns
                actual_end_unix_ns = piece_actual_end_unix_ns

            cursor_unix_ns = piece_end_unix_ns
            if cursor_unix_ns == time_range.end_unix_ns:
                if actual_start_unix_ns is None or actual_end_unix_ns is None:
                    raise TimelineRangeError(
                        ERROR_RANGE_NOT_CONTINUOUS,
                        "requested ASR timeline range contains no selectable samples",
                    )
                if actual_start_unix_ns < self._retained_start_unix_ns:
                    raise TimelineRangeError(
                        ERROR_RANGE_OUTSIDE_WINDOW,
                        "requested start quantizes before the retained ASR timeline window",
                    )
                payload = AsrAudioPayload(
                    encoding=encoding,
                    sample_rate_hz=self._sample_rate,
                    channels=channels,
                    data=b"".join(pieces),
                    sample_count=sum(
                        self._payload_sample_count(
                            payload_bytes=piece,
                            encoding=encoding,
                            channels=channels,
                        )
                        for piece in pieces
                    ),
                )
                return TimelineSlice(
                    payload=payload,
                    time_range=NumericTimeRange(
                        start_unix_ns=actual_start_unix_ns,
                        end_unix_ns=actual_end_unix_ns,
                    ),
                )

        raise TimelineRangeError(
            ERROR_RANGE_NOT_CONTINUOUS,
            "requested ASR timeline range is not covered by retained audio",
        )

    def _drop_expired_frames(self) -> None:
        retained_frames: list[TimelineFrame] = []
        for frame in self._frames:
            if frame.end_unix_ns > self._retained_start_unix_ns:
                retained_frames.append(frame)
        self._frames = retained_frames

    def _resolve_append_payload(
        self,
        *,
        payload: AsrAudioPayload | None,
        samples: np.ndarray | None,
    ) -> AsrAudioPayload:
        if payload is not None and samples is not None:
            raise ValueError("append accepts either payload or samples, not both")
        if payload is not None:
            return payload
        if samples is None:
            raise ValueError("append requires payload or samples")
        return AsrAudioPayload.from_float32_samples(
            samples,
            sample_rate_hz=self._sample_rate,
            channels=1,
        )

    @staticmethod
    def _slice_payload_bytes(
        payload: AsrAudioPayload,
        *,
        start_sample: int,
        end_sample: int,
    ) -> bytes:
        bytes_per_sample = RollingAsrTimeline._payload_bytes_per_sample(
            encoding=payload.encoding
        )
        stride = bytes_per_sample * payload.channels
        return payload.data[start_sample * stride : end_sample * stride]

    @staticmethod
    def _payload_sample_count(
        *,
        payload_bytes: bytes,
        encoding: str,
        channels: int,
    ) -> int:
        stride = RollingAsrTimeline._payload_bytes_per_sample(encoding=encoding) * channels
        return len(payload_bytes) // stride

    @staticmethod
    def _payload_bytes_per_sample(*, encoding: str) -> int:
        if encoding == "FLOAT32LE":
            return np.dtype("<f4").itemsize
        if encoding == "PCM16LE":
            return np.dtype("<i2").itemsize
        raise ValueError(f"unsupported ASR timeline payload encoding: {encoding}")

    def _align_adjacent_frame_start(self, start_unix_ns: int) -> int:
        if not self._frames:
            return start_unix_ns

        previous = self._frames[-1]
        tolerance_ns = self._timestamp_alignment_tolerance_ns
        if start_unix_ns < previous.floor_end_unix_ns:
            overlap_ns = previous.floor_end_unix_ns - start_unix_ns
            if overlap_ns <= tolerance_ns:
                return previous.floor_end_unix_ns
            raise TimelineRangeError(
                ERROR_WINDOW_NOT_FOUND,
                "overlapping AudioFrame rejected from ASR timeline",
            )

        if start_unix_ns > previous.end_unix_ns:
            gap_ns = start_unix_ns - previous.end_unix_ns
            if gap_ns <= tolerance_ns:
                return previous.end_unix_ns

        return start_unix_ns

    def _sample_index_to_floor_duration_ns(self, sample_index: int) -> int:
        numerator = sample_index * _NSEC_PER_SEC
        duration_ns, _remainder = divmod(numerator, self._sample_rate)
        return duration_ns

    def _sample_index_to_ceil_duration_ns(self, sample_index: int) -> int:
        numerator = sample_index * _NSEC_PER_SEC
        duration_ns, remainder = divmod(numerator, self._sample_rate)
        if remainder != 0:
            duration_ns += 1
        return duration_ns

    def _floor_sample_index(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise TimelineRangeError(
                ERROR_RANGE_NOT_CONTINUOUS,
                "requested time is before ASR frame coverage",
            )
        numerator = delta_ns * self._sample_rate
        sample_index, _remainder = divmod(numerator, _NSEC_PER_SEC)
        return sample_index

    def _ceil_sample_index(self, delta_ns: int) -> int:
        if delta_ns < 0:
            raise TimelineRangeError(
                ERROR_RANGE_NOT_CONTINUOUS,
                "requested time is before ASR frame coverage",
            )
        numerator = delta_ns * self._sample_rate
        sample_index, remainder = divmod(numerator, _NSEC_PER_SEC)
        if remainder != 0:
            sample_index += 1
        return sample_index
