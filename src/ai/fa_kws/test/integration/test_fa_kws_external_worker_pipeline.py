import os
import shutil
import subprocess
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[2]


def _write_probe_source(path: Path) -> None:
    path.write_text(
        r'''
#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

int main(int argc, char **argv)
{
  if (argc != 3) {
    std::cerr << "usage: probe <worker> <tmp_dir>" << std::endl;
    return 2;
  }
  const std::filesystem::path worker = argv[1];
  const std::filesystem::path tmp_dir = argv[2];
  try {
    fa_kws::SherpaOnnxKwsBackendConfig cfg;
    cfg.target_sample_rate = 16000;
    cfg.model_num_threads = 1;
    cfg.execution_provider = "cpu";
    cfg.encoder_path = (tmp_dir / "encoder.onnx").string();
    cfg.decoder_path = (tmp_dir / "decoder.onnx").string();
    cfg.joiner_path = (tmp_dir / "joiner.onnx").string();
    cfg.tokens_path = (tmp_dir / "tokens.txt").string();
    cfg.keywords_path = (tmp_dir / "keywords.txt").string();
    cfg.max_active_paths = 4;
    cfg.num_trailing_blanks = 1;
    cfg.keywords_score = 1.0f;
    cfg.keywords_threshold = 0.25f;
    cfg.vad_threshold = 0.5f;
    cfg.cooldown = std::chrono::milliseconds{0};
    cfg.command = worker.string();
    cfg.args = {
      "detect",
      "--audio",
      "{audio}",
      "--encoder",
      "{encoder}",
      "--decoder",
      "{decoder}",
      "--joiner",
      "{joiner}",
      "--tokens",
      "{tokens}",
      "--keywords",
      "{keywords}",
      "--provider",
      "{provider}",
      "--sample-rate",
      "{sample_rate}",
      "--num-threads",
      "{num_threads}",
      "--max-active-paths",
      "{max_active_paths}",
      "--num-trailing-blanks",
      "{num_trailing_blanks}",
      "--keywords-score",
      "{keywords_score}",
      "--keywords-threshold",
      "{keywords_threshold}",
    };
    cfg.health_args = {
      "health",
      "--encoder",
      "{encoder}",
      "--decoder",
      "{decoder}",
      "--joiner",
      "{joiner}",
      "--tokens",
      "{tokens}",
      "--keywords",
      "{keywords}",
      "--provider",
      "{provider}",
      "--sample-rate",
      "{sample_rate}",
      "--num-threads",
      "{num_threads}",
      "--max-active-paths",
      "{max_active_paths}",
      "--num-trailing-blanks",
      "{num_trailing_blanks}",
      "--keywords-score",
      "{keywords_score}",
      "--keywords-threshold",
      "{keywords_threshold}",
    };
    cfg.timeout_sec = 2.0;
    cfg.workspace_dir = (tmp_dir / "workspace").string();
    cfg.cleanup_audio_files = true;

    fa_kws::SherpaOnnxKwsBackend backend(cfg);
    const std::vector<float> samples(160, 0.1f);
    const auto result = backend.process(
      samples,
      16000,
      1.0f,
      std::chrono::steady_clock::now());
    if (!result) {
      std::cout << "NO_DETECTION" << std::endl;
      return 0;
    }
    std::cout << result->keyword << "\t" << result->score << "\t"
              << result->start_time_sec << std::endl;
    return 0;
  } catch (const std::exception &exc) {
    std::cerr << exc.what() << std::endl;
    return 3;
  }
}
''',
        encoding="utf-8",
    )


def _prepare_model_files(tmp_path: Path) -> None:
    for filename in (
        "encoder.onnx",
        "decoder.onnx",
        "joiner.onnx",
        "tokens.txt",
        "keywords.txt",
    ):
        (tmp_path / filename).write_text(filename + "\n", encoding="utf-8")


def _prepare_worker(tmp_path: Path) -> Path:
    source = PACKAGE_ROOT / "test" / "fixtures" / "fake_kws_worker.py"
    worker = tmp_path / "fake_kws_worker.py"
    shutil.copy2(source, worker)
    worker.chmod(worker.stat().st_mode | 0o111)
    return worker


def _build_probe(tmp_path: Path) -> Path:
    source_path = tmp_path / "kws_external_probe.cpp"
    binary_path = tmp_path / "kws_external_probe"
    _write_probe_source(source_path)
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-I",
            str(PACKAGE_ROOT / "include"),
            str(PACKAGE_ROOT / "src" / "backends" / "sherpa_onnx_kws_backend.cpp"),
            str(source_path),
            "-o",
            str(binary_path),
        ],
        check=True,
    )
    return binary_path


def test_external_kws_worker_detects_keyword(tmp_path: Path) -> None:
    _prepare_model_files(tmp_path)
    worker = _prepare_worker(tmp_path)
    binary = _build_probe(tmp_path)

    result = subprocess.run(
        [str(binary), str(worker), str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "hello_fluent\t0.875\t0.25"
    assert not any((tmp_path / "workspace").glob("*.f32"))


def test_external_kws_worker_no_detection_is_not_error(tmp_path: Path) -> None:
    _prepare_model_files(tmp_path)
    worker = _prepare_worker(tmp_path)
    binary = _build_probe(tmp_path)
    env = os.environ.copy()
    env["FA_KWS_FAKE_MODE"] = "none"

    result = subprocess.run(
        [str(binary), str(worker), str(tmp_path)],
        check=True,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.stdout.strip() == "NO_DETECTION"


def test_external_kws_worker_invalid_stdout_fails_closed(tmp_path: Path) -> None:
    _prepare_model_files(tmp_path)
    worker = _prepare_worker(tmp_path)
    binary = _build_probe(tmp_path)
    env = os.environ.copy()
    env["FA_KWS_FAKE_MODE"] = "bad"

    result = subprocess.run(
        [str(binary), str(worker), str(tmp_path)],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 3
    assert "KWS backend output must be NO_DETECTION" in result.stderr
