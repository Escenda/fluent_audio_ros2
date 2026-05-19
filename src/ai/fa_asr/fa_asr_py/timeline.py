from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


ERROR_TIME_RANGE_UNRESOLVED = "time_range_unresolved"
ERROR_WINDOW_NOT_FOUND = "window_not_found"
ERROR_RANGE_OUTSIDE_WINDOW = "range_outside_window"

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

        duration_ns = self._sample_count_to_duration_ns(samples.size)
        end_unix_ns = start_unix_ns + duration_ns
        if self._frames and start_unix_ns < self._frames[-1].end_unix_ns:
            raise TimelineRangeError(
                ERROR_WINDOW_NOT_FOUND,
                "overlapping AudioFrame rejected from ASR timeline",
            )

        self._frames.append(
            TimelineFrame(
                start_unix_ns=start_unix_ns,
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
        cursor_unix_ns = time_range.start_unix_ns
        for frame in self._frames:
            if frame.end_unix_ns <= cursor_unix_ns:
                continue
            if frame.start_unix_ns > cursor_unix_ns:
                raise TimelineRangeError(
                    ERROR_WINDOW_NOT_FOUND,
                    "requested ASR timeline range crosses an audio gap",
                )
            piece_end_unix_ns = min(time_range.end_unix_ns, frame.end_unix_ns)
            pieces.append(
                self._slice_frame(
                    frame,
                    start_unix_ns=cursor_unix_ns,
                    end_unix_ns=piece_end_unix_ns,
                )
            )
            cursor_unix_ns = piece_end_unix_ns
            if cursor_unix_ns == time_range.end_unix_ns:
                return TimelineSlice(
                    samples=np.concatenate(pieces).astype(np.float32, copy=False),
                    sample_rate=self._sample_rate,
                    time_range=time_range,
                )

        raise TimelineRangeError(
            ERROR_WINDOW_NOT_FOUND,
            "requested ASR timeline range is not covered by retained audio",
        )

    def _drop_expired_frames(self) -> None:
        retained_frames: list[TimelineFrame] = []
        for frame in self._frames:
            if frame.end_unix_ns > self._retained_start_unix_ns:
                retained_frames.append(frame)
        self._frames = retained_frames

    def _sample_count_to_duration_ns(self, sample_count: int) -> int:
        numerator = sample_count * _NSEC_PER_SEC
        duration_ns, remainder = divmod(numerator, self._sample_rate)
        if remainder != 0:
            raise TimelineRangeError(
                ERROR_TIME_RANGE_UNRESOLVED,
                "sample count cannot be represented exactly in unix nanoseconds",
            )
        return duration_ns

    def _slice_frame(
        self,
        frame: TimelineFrame,
        *,
        start_unix_ns: int,
        end_unix_ns: int,
    ) -> np.ndarray:
        start_index = self._exact_sample_index(
            start_unix_ns - frame.start_unix_ns,
            "requested start does not map to an exact ASR sample boundary",
        )
        end_index = self._exact_sample_index(
            end_unix_ns - frame.start_unix_ns,
            "requested end does not map to an exact ASR sample boundary",
        )
        return frame.samples[start_index:end_index]

    def _exact_sample_index(self, delta_ns: int, message: str) -> int:
        if delta_ns < 0:
            raise TimelineRangeError(ERROR_WINDOW_NOT_FOUND, message)
        numerator = delta_ns * self._sample_rate
        sample_index, remainder = divmod(numerator, _NSEC_PER_SEC)
        if remainder != 0:
            raise TimelineRangeError(ERROR_TIME_RANGE_UNRESOLVED, message)
        return sample_index
