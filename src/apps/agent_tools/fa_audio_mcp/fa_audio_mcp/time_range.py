from __future__ import annotations

from dataclasses import dataclass
import re

from fa_audio_mcp.errors import AudioToolError


_NUMERIC_RANGE_RE = re.compile(r"^(0|[1-9][0-9]*)\.\.(0|[1-9][0-9]*)$")


@dataclass(frozen=True)
class NumericTimeRange:
    start_unix_ns: int
    end_unix_ns: int

    @property
    def spec(self) -> str:
        return f"{self.start_unix_ns}..{self.end_unix_ns}"


def parse_numeric_time_range(time_range: str) -> NumericTimeRange:
    match = _NUMERIC_RANGE_RE.fullmatch(time_range)
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
    )
