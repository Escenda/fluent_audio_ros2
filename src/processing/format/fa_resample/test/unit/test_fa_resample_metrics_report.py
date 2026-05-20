import subprocess
import sys
from pathlib import Path


def package_root() -> Path:
    return Path(__file__).parents[2]


def script_path() -> Path:
    return package_root() / "scripts" / "fa_resample_metrics_report.py"


def write_xml(path: Path, properties: str) -> None:
    path.write_text(
        f"""<?xml version="1.0" encoding="UTF-8"?>
<testsuites>
  <testsuite name="fa_resample_backend_test">
    <testcase name="ComputesPassbandMetricsAgainstRequiredSoxrVhqReference">
      <properties>
{properties}
      </properties>
    </testcase>
  </testsuite>
</testsuites>
""",
        encoding="utf-8",
    )


def run_report(input_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script_path()), str(input_path)],
        check=False,
        text=True,
        capture_output=True,
    )


def test_metrics_report_groups_quality_and_backend_properties(tmp_path: Path) -> None:
    xml_file = tmp_path / "fa_resample_backend_test.gtest.xml"
    write_xml(
        xml_file,
        """        <property name="passband_soxr_mq_rms_error" value="1.250000000000000e-03"/>
        <property name="passband_soxr_mq_peak_error" value="3.000000000000000e-02"/>
        <property name="passband_soxr_mq_snr_db" value="4.210000000000000e+01"/>
        <property name="passband_soxr_mq_compared_samples" value="15980"/>
        <property name="passband_soxr_mq_algorithmic_delay_input_samples" value="5.400000000000000e+01"/>
        <property name="passband_soxr_mq_algorithmic_delay_output_samples" value="1.800000000000000e+01"/>
        <property name="passband_soxr_mq_algorithmic_delay_ms" value="1.200000000000000e+00"/>
        <property name="passband_soxr_mq_processing_time_mean_ms" value="2.300000000000000e-02"/>
        <property name="passband_soxr_mq_processing_time_max_ms" value="4.500000000000000e-02"/>
        <property name="passband_soxr_mq_input_frames_total" value="50"/>
        <property name="passband_soxr_mq_output_frames_total" value="50"/>
        <property name="passband_soxr_mq_frame_count_error_samples" value="0"/>""",
    )

    result = run_report(xml_file)

    assert result.returncode == 0
    assert "fa_resample metrics report" in result.stdout
    assert "passband_soxr_mq" in result.stdout
    assert "rms_error" in result.stdout
    assert "algorithmic_delay_input_samples" in result.stdout
    assert "algorithmic_delay_output_samples" in result.stdout
    assert "5.400000000000000e+01" in result.stdout
    assert "algorithmic_delay_ms" in result.stdout
    assert "frame_count_error_samples" in result.stdout
    assert result.stderr == ""


def test_metrics_report_accepts_directory_containing_fa_resample_xml(tmp_path: Path) -> None:
    result_dir = tmp_path / "test_results" / "fa_resample"
    result_dir.mkdir(parents=True)
    xml_file = result_dir / "fa_resample_backend_test.gtest.xml"
    write_xml(
        xml_file,
        """        <property name="alias_soxr_hq_rms_error" value="1.100000000000000e-03"/>
        <property name="alias_soxr_hq_peak_error" value="2.400000000000000e-02"/>
        <property name="alias_soxr_hq_snr_db" value="3.950000000000000e+01"/>
        <property name="alias_soxr_hq_compared_samples" value="15976"/>
        <property name="alias_soxr_hq_algorithmic_delay_input_samples" value="5.400000000000000e+01"/>
        <property name="alias_soxr_hq_algorithmic_delay_output_samples" value="1.800000000000000e+01"/>
        <property name="alias_soxr_hq_algorithmic_delay_ms" value="1.125000000000000e+00"/>
        <property name="alias_soxr_hq_processing_time_mean_ms" value="1.900000000000000e-02"/>
        <property name="alias_soxr_hq_processing_time_max_ms" value="3.600000000000000e-02"/>
        <property name="alias_soxr_hq_input_frames_total" value="50"/>
        <property name="alias_soxr_hq_output_frames_total" value="50"/>
        <property name="alias_soxr_hq_frame_count_error_samples" value="0"/>""",
    )

    result = run_report(tmp_path)

    assert result.returncode == 0
    assert "alias_soxr_hq" in result.stdout
    assert "snr_db" in result.stdout
    assert "3.950000000000000e+01" in result.stdout
    assert "compared_samples" in result.stdout
    assert result.stderr == ""


def test_metrics_report_fails_when_directory_has_no_fa_resample_xml(tmp_path: Path) -> None:
    result = run_report(tmp_path)

    assert result.returncode == 1
    assert "no fa_resample GTest XML files found" in result.stderr
    assert result.stdout == ""


def test_metrics_report_fails_when_xml_has_no_metric_properties(tmp_path: Path) -> None:
    xml_file = tmp_path / "fa_resample_backend_test.gtest.xml"
    write_xml(xml_file, '        <property name="unrelated_property" value="ignored"/>')

    result = run_report(xml_file)

    assert result.returncode == 1
    assert "no fa_resample metric properties found" in result.stderr
    assert result.stdout == ""


def test_metrics_report_fails_when_metric_group_is_incomplete(tmp_path: Path) -> None:
    xml_file = tmp_path / "fa_resample_backend_test.gtest.xml"
    write_xml(
        xml_file,
        """        <property name="alias_soxr_hq_snr_db" value="3.950000000000000e+01"/>
        <property name="alias_soxr_hq_compared_samples" value="15976"/>
        <property name="alias_soxr_hq_algorithmic_delay_ms" value="1.125000000000000e+00"/>""",
    )

    result = run_report(xml_file)

    assert result.returncode == 1
    assert "incomplete fa_resample metric group(s)" in result.stderr
    assert "alias_soxr_hq" in result.stderr
    assert "algorithmic_delay_input_samples" in result.stderr
    assert "rms_error" in result.stderr
    assert result.stdout == ""
