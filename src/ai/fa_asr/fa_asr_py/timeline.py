from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


ERROR_TIME_RANGE_UNRESOLVED = "time_range_unresolved"
ERROR_WINDOW_NOT_FOUND = "window_not_found"
ERROR_RANGE_OUTSIDE_WINDOW = "range_outside_window"
ERROR_RANGE_NOT_CONTINUOUS = "range_not_continuous"

_NSEC_PER_SEC = 1_000_000_000
_NUMERIC_RANGE_PATTERN = re.compile(r"^([0-9]+)\.\.([0-9]+)$")


@dataclass(frozen=True)
class NumericTimeRange:
    start_unix_ns: int
    end_unix_ns: int


@dataclass(frozen=True)
class TimelineSlice:
    samples: np.ndarray
    sample_rate: int
    time_range: NumericTimeRange


@dataclass(frozen=True)
class TimelineFrame:
    start_unix_ns: int
    floor_end_unix_ns: int
    end_unix_ns: int
    samples: np.ndarray


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
    def __init__(self, *, sample_rate: int, retention_sec: float) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be greater than zero")
        if not np.isfinite(retention_sec) or retention_sec <= 0.0:
            raise ValueError("retention_sec must be finite and greater than zero")
        retention_ns = int(retention_sec * _NSEC_PER_SEC)
        if retention_ns <= 0:
            raise ValueError("retention_sec is too small to represent in nanoseconds")

        self._sample_rate = int(sample_rate)
        self._retention_ns = retention_ns
        self._frames: list[TimelineFrame] = []
        self._latest_end_unix_ns = 0
        self._retained_start_unix_ns = 0

    @property
    def retained_start_unix_ns(self) -> int:
        return self._retained_start_unix_ns

    @property
    def latest_end_unix_ns(self) -> int:
        return self._latest_end_unix_ns

    def append(self, *, start_unix_ns: int, samples: np.ndarray) -> None:
        if start_unix_ns <= 0:
            raise TimelineRangeError(
                ERROR_TIME_RANGE_UNRESOLVED,
                "AudioFrame header.stamp must resolve to a positive unix nanosecond timestamp",
            )
        if samples.ndim != 1:
            raise ValueError("samples must be a 1D array")
        if samples.dtype != np.float32:
            raise ValueError("samples dtype must be np.float32")
        if samples.size == 0:
            raise ValueError("samples must not be empty")

        floor_duration_ns = self._sample_index_to_floor_duration_ns(samples.size)
        ceil_duration_ns = self._sample_index_to_ceil_duration_ns(samples.size)
        floor_end_unix_ns = start_unix_ns + floor_duration_ns
        end_unix_ns = start_unix_ns + ceil_duration_ns
        if self._frames and start_unix_ns < self._frames[-1].floor_end_unix_ns:
            raise TimelineRangeError(
                ERROR_WINDOW_NOT_FOUND,
                "overlapping AudioFrame rejected from ASR timeline",
            )

        self._frames.append(
            TimelineFrame(
                start_unix_ns=start_unix_ns,
                floor_end_unix_ns=floor_end_unix_ns,
                end_unix_ns=end_unix_ns,
                samples=samples.copy(),
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

        pieces: list[np.ndarray] = []
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
            piece_end_unix_ns = min(time_range.end_unix_ns, frame.end_unix_ns)
            frame_start_index = self._floor_sample_index(cursor_unix_ns - frame.start_unix_ns)
            frame_end_index = self._ceil_sample_index(piece_end_unix_ns - frame.start_unix_ns)
            frame_start_index = min(frame_start_index, frame.samples.size)
            frame_end_index = min(frame_end_index, frame.samples.size)

            if frame_start_index < frame_end_index:
                pieces.append(frame.samples[frame_start_index:frame_end_index])
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
                return TimelineSlice(
                    samples=np.concatenate(pieces).astype(np.float32, copy=False),
                    sample_rate=self._sample_rate,
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
