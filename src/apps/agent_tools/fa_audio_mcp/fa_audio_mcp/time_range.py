from __future__ import annotations

from dataclasses import dataclass
import re

from fa_audio_mcp.errors import AudioToolError


_NUMERIC_RANGE_RE = re.compile(r"^(0|[1-9][0-9]*)\.\.(0|[1-9][0-9]*)$")
_NOW_ENDPOINT_RE = re.compile(r"^now(?:-(0|[1-9][0-9]*)(ms|s|m))?$")
_NSEC_PER_MSEC = 1_000_000
_NSEC_PER_SEC = 1_000_000_000
_NSEC_PER_MIN = 60 * _NSEC_PER_SEC


@dataclass(frozen=True)
class NumericTimeRange:
    start_unix_ns: int
    end_unix_ns: int
    requested_spec: str = ""

    @property
    def spec(self) -> str:
        return f"{self.start_unix_ns}..{self.end_unix_ns}"


def parse_numeric_time_range(time_range: str) -> NumericTimeRange:
    stripped_time_range = time_range.strip()
    match = _NUMERIC_RANGE_RE.fullmatch(stripped_time_range)
    if match is None:
        raise AudioToolError(
            "invalid_time_range",
            "time_range must use numeric '<start_unix_ns>..<end_unix_ns>' form",
        )

    start_unix_ns = int(match.group(1))
    end_unix_ns = int(match.group(2))
    if start_unix_ns >= end_unix_ns:
        raise AudioToolError(
            "invalid_time_range",
            "time_range must satisfy 0 <= start_unix_ns < end_unix_ns",
        )

    return NumericTimeRange(
        start_unix_ns=start_unix_ns,
        end_unix_ns=end_unix_ns,
        requested_spec=stripped_time_range,
    )


def resolve_time_range(time_range: str, *, now_unix_ns: int | None = None) -> NumericTimeRange:
    stripped_time_range = time_range.strip()
    if _NUMERIC_RANGE_RE.fullmatch(stripped_time_range) is not None:
        return parse_numeric_time_range(stripped_time_range)

    if "now" not in stripped_time_range:
        raise AudioToolError(
            "invalid_time_range",
            "time_range must be numeric or use 'now[-duration]..now[-duration]' form",
        )
    if now_unix_ns is None:
        raise AudioToolError(
            "invalid_time_range",
            "relative 'now' time_range requires an explicit node clock timestamp",
        )
    if now_unix_ns < 0:
        raise AudioToolError(
            "invalid_time_range",
            "relative 'now' timestamp must be non-negative",
        )

    parts = stripped_time_range.split("..")
    if len(parts) != 2:
        raise AudioToolError(
            "invalid_time_range",
            "relative time_range must contain exactly one '..' separator",
        )
    start_unix_ns = _resolve_now_endpoint(parts[0], now_unix_ns)
    end_unix_ns = _resolve_now_endpoint(parts[1], now_unix_ns)
    if start_unix_ns >= end_unix_ns:
        raise AudioToolError(
            "invalid_time_range",
            "time_range must satisfy 0 <= start_unix_ns < end_unix_ns",
        )
    return NumericTimeRange(
        start_unix_ns=start_unix_ns,
        end_unix_ns=end_unix_ns,
        requested_spec=stripped_time_range,
    )


def requested_time_range_spec(time_range: NumericTimeRange) -> str:
    if time_range.requested_spec:
        return time_range.requested_spec
    return time_range.spec


def _resolve_now_endpoint(endpoint: str, now_unix_ns: int) -> int:
    match = _NOW_ENDPOINT_RE.fullmatch(endpoint.strip())
    if match is None:
        raise AudioToolError(
            "invalid_time_range",
            "relative endpoint must be 'now', 'now-<N>ms', 'now-<N>s', or 'now-<N>m'",
        )

    amount_text = match.group(1)
    unit = match.group(2)
    if amount_text is None or unit is None:
        return now_unix_ns

    offset_ns = _duration_to_nanoseconds(int(amount_text), unit)
    if offset_ns > now_unix_ns:
        raise AudioToolError(
            "invalid_time_range",
            "relative 'now' endpoint resolved before timestamp zero",
        )
    return now_unix_ns - offset_ns


def _duration_to_nanoseconds(amount: int, unit: str) -> int:
    if unit == "ms":
        return amount * _NSEC_PER_MSEC
    if unit == "s":
        return amount * _NSEC_PER_SEC
    if unit == "m":
        return amount * _NSEC_PER_MIN
    raise AudioToolError("invalid_time_range", "unsupported relative time unit")
