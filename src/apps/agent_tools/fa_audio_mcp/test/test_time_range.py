import pytest

from fa_audio_mcp.errors import AudioToolError
from fa_audio_mcp.time_range import TimeMarkerResolver, parse_numeric_time_range, resolve_time_range


def test_numeric_time_range_accepts_valid_range() -> None:
    time_range = parse_numeric_time_range(" 1000000000..2000000000 ")

    assert time_range.start_unix_ns == 1000000000
    assert time_range.end_unix_ns == 2000000000
    assert time_range.spec == "1000000000..2000000000"
    assert time_range.requested_spec == "1000000000..2000000000"


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


@pytest.mark.parametrize(
    ("value", "start_unix_ns", "end_unix_ns"),
    [
        ("now-10s..now", 1700000000000000000, 1700000010000000000),
        ("now-1500ms..now-500ms", 1700000008500000000, 1700000009500000000),
        ("now-2m..now-1m", 1699999890000000000, 1699999950000000000),
    ],
)
def test_resolve_time_range_resolves_now_relative_specs(
    value: str,
    start_unix_ns: int,
    end_unix_ns: int,
) -> None:
    time_range = resolve_time_range(value, now_unix_ns=1700000010000000000)

    assert time_range.start_unix_ns == start_unix_ns
    assert time_range.end_unix_ns == end_unix_ns
    assert time_range.spec == f"{start_unix_ns}..{end_unix_ns}"
    assert time_range.requested_spec == value


def test_resolve_time_range_requires_clock_for_now_relative_specs() -> None:
    with pytest.raises(AudioToolError) as exc_info:
        resolve_time_range("now-10s..now")

    assert exc_info.value.error_code == "invalid_time_range"


@pytest.mark.parametrize(
    "value",
    [
        "now..now",
        "now+1s..now",
        "now-10h..now",
        "now-10s",
        "now-10s..now-20s",
        "now-200s..now",
    ],
)
def test_resolve_time_range_rejects_invalid_now_relative_specs(value: str) -> None:
    with pytest.raises(AudioToolError) as exc_info:
        resolve_time_range(value, now_unix_ns=100000000000)

    assert exc_info.value.error_code == "invalid_time_range"


def test_resolve_time_range_resolves_marker_range_with_offset() -> None:
    resolver = TimeMarkerResolver(
        {
            "action_12.start": 1_700_000_000_000_000_000,
            "action_12.end": 1_700_000_005_000_000_000,
        }
    )

    time_range = resolve_time_range(
        "action_12.start..action_12.end+2s",
        marker_resolver=resolver,
    )

    assert time_range.start_unix_ns == 1_700_000_000_000_000_000
    assert time_range.end_unix_ns == 1_700_000_007_000_000_000
    assert time_range.spec == "1700000000000000000..1700000007000000000"
    assert time_range.requested_spec == "action_12.start..action_12.end+2s"


def test_resolve_time_range_resolves_hyphen_marker_and_ms_offset() -> None:
    resolver = TimeMarkerResolver(
        {
            "action-12.start": 2_000_000_000,
            "action-12.end": 5_000_000_000,
        }
    )

    time_range = resolve_time_range(
        "action-12.start+250ms..action-12.end-500ms",
        marker_resolver=resolver,
    )

    assert time_range.start_unix_ns == 2_250_000_000
    assert time_range.end_unix_ns == 4_500_000_000
    assert time_range.requested_spec == "action-12.start+250ms..action-12.end-500ms"


@pytest.mark.parametrize(
    "value",
    [
        "action_12.start..action_12.end",
        "action_12.start..missing.end",
        "action_12.middle..action_12.end",
        "action_12.start-2ms..action_12.end",
        "action_12.end..action_12.start",
    ],
)
def test_resolve_time_range_rejects_invalid_marker_specs(value: str) -> None:
    resolver = TimeMarkerResolver(
        {
            "action_12.start": 1_000_000,
            "action_12.end": 2_000_000,
        }
    )
    marker_resolver = None if value == "action_12.start..action_12.end" else resolver

    with pytest.raises(AudioToolError) as exc_info:
        resolve_time_range(value, marker_resolver=marker_resolver)

    assert exc_info.value.error_code == "invalid_time_range"
