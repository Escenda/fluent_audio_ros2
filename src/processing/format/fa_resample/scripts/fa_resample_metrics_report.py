#!/usr/bin/env python3
"""Render fa_resample GTest XML metric properties as a text report."""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


QUALITY_METRICS = (
    "rms_error",
    "peak_error",
    "snr_db",
    "compared_samples",
)

BACKEND_METRICS = (
    "algorithmic_delay_input_samples",
    "algorithmic_delay_output_samples",
    "algorithmic_delay_ms",
    "processing_time_mean_ms",
    "processing_time_max_ms",
    "input_frames_total",
    "output_frames_total",
    "frame_count_error_samples",
)

METRIC_SUFFIXES = QUALITY_METRICS + BACKEND_METRICS


@dataclass(frozen=True)
class MetricValue:
    source: Path
    testcase: str
    prefix: str
    metric: str
    value: str


@dataclass(frozen=True)
class MetricGroup:
    prefix: str
    values: tuple[MetricValue, ...]


class MetricsReportError(RuntimeError):
    """Raised when fa_resample metric XML cannot produce a report."""


def discover_xml_files(input_path: Path) -> tuple[Path, ...]:
    if not input_path.exists():
        raise MetricsReportError(f"input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix != ".xml":
            raise MetricsReportError(f"input file is not XML: {input_path}")
        return (input_path,)

    if not input_path.is_dir():
        raise MetricsReportError(f"input path is neither file nor directory: {input_path}")

    xml_files = tuple(sorted(input_path.rglob("fa_resample*.gtest.xml")))
    if not xml_files:
        raise MetricsReportError(f"no fa_resample GTest XML files found under: {input_path}")
    return xml_files


def parse_metric_values(xml_file: Path) -> tuple[MetricValue, ...]:
    try:
        root = ET.parse(xml_file).getroot()
    except ET.ParseError as error:
        raise MetricsReportError(f"invalid XML in {xml_file}: {error}") from error

    values: list[MetricValue] = []
    for testcase in root.iter("testcase"):
        testcase_name = testcase.get("name")
        if not testcase_name:
            raise MetricsReportError(f"testcase without name in {xml_file}")

        for prop in testcase.findall("./properties/property"):
            name = prop.get("name")
            value = prop.get("value")
            if name is None or value is None:
                raise MetricsReportError(
                    f"property without name or value in testcase {testcase_name}: {xml_file}"
                )
            metric = metric_suffix(name)
            if metric is None:
                continue
            prefix = name[: -(len(metric) + 1)]
            values.append(
                MetricValue(
                    source=xml_file,
                    testcase=testcase_name,
                    prefix=prefix,
                    metric=metric,
                    value=value,
                )
            )
    return tuple(values)


def metric_suffix(property_name: str) -> str | None:
    for metric in sorted(METRIC_SUFFIXES, key=len, reverse=True):
        suffix = f"_{metric}"
        if property_name.endswith(suffix) and len(property_name) > len(suffix):
            return metric
    return None


def load_metric_groups(input_path: Path) -> tuple[MetricGroup, ...]:
    values: list[MetricValue] = []
    for xml_file in discover_xml_files(input_path):
        values.extend(parse_metric_values(xml_file))

    if not values:
        raise MetricsReportError(f"no fa_resample metric properties found in: {input_path}")

    groups: dict[str, list[MetricValue]] = {}
    for value in values:
        groups.setdefault(value.prefix, []).append(value)

    metric_groups = tuple(
        MetricGroup(prefix=prefix, values=tuple(sorted(items, key=lambda item: item.metric)))
        for prefix, items in sorted(groups.items())
    )
    validate_metric_groups(metric_groups)
    return metric_groups


def validate_metric_groups(groups: tuple[MetricGroup, ...]) -> None:
    incomplete: list[str] = []
    for group in groups:
        required_metrics = set(BACKEND_METRICS)
        if "reference" not in group.prefix:
            required_metrics.update(QUALITY_METRICS)

        present_metrics = {value.metric for value in group.values}
        missing_metrics = tuple(sorted(required_metrics - present_metrics))
        if missing_metrics:
            incomplete.append(f"{group.prefix}: missing {', '.join(missing_metrics)}")

    if incomplete:
        raise MetricsReportError("incomplete fa_resample metric group(s): " + "; ".join(incomplete))


def format_report(groups: tuple[MetricGroup, ...]) -> str:
    lines = ["fa_resample metrics report", ""]
    for group in groups:
        lines.append(group.prefix)
        lines.append("-" * len(group.prefix))
        metric_width = max(len("metric"), *(len(value.metric) for value in group.values))
        value_width = max(len("value"), *(len(value.value) for value in group.values))
        lines.append(f"{'metric':<{metric_width}}  {'value':>{value_width}}")
        lines.append(f"{'-' * metric_width}  {'-' * value_width}")
        for metric in METRIC_SUFFIXES:
            matching_values = tuple(value for value in group.values if value.metric == metric)
            for value in matching_values:
                lines.append(f"{value.metric:<{metric_width}}  {value.value:>{value_width}}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert fa_resample_backend_test.gtest.xml quality/backend properties "
            "into a readable metrics report."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to fa_resample_backend_test.gtest.xml or a directory containing fa_resample XML.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        groups = load_metric_groups(args.input_path)
    except MetricsReportError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    print(format_report(groups), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
