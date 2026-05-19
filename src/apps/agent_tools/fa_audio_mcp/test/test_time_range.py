import pytest

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.time_range import parse_numeric_time_range


def test_numeric_time_range_accepts_valid_range() -> None:
    time_range = parse_numeric_time_range("1000000000..2000000000")

    assert time_range.start_unix_ns == 1000000000
    assert time_range.end_unix_ns == 2000000000
    assert time_range.spec == "1000000000..2000000000"


@pytest.mark.parametrize(
    "value",
    [
        "now-10s..now",
        "200..100",
        "-1..100",
        "100",
        "100..",
        "..200",
        "100...200",
        "marker_a..marker_b",
    ],
)
def test_numeric_time_range_rejects_invalid_ranges(value: str) -> None:
    with pytest.raises(AudioToolError):
        parse_numeric_time_range(value)
