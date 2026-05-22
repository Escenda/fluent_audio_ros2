#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"

#include <fcntl.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace
{

struct CommandResult
{
  int exit_code;
  std::string stdout_text;
  std::string stderr_text;
};

std::string trim(const std::string &value)
{
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

std::vector<std::string> splitLine(const std::string &line, char delimiter)
{
  std::vector<std::string> fields;
  std::string field;
  std::istringstream stream(line);
  while (std::getline(stream, field, delimiter)) {
    fields.push_back(field);
  }
  if (!line.empty() && line.back() == delimiter) {
    fields.emplace_back();
  }
  return fields;
}

float parseFiniteFloat(const std::string &value, const char *label)
{
  std::size_t parsed = 0;
  const float parsed_value = std::stof(value, &parsed);
  if (parsed != value.size() || !std::isfinite(parsed_value)) {
    throw std::runtime_error(std::string(label) + " must be finite");
  }
  return parsed_value;
}

double parseFiniteDouble(const std::string &value, const char *label)
{
  std::size_t parsed = 0;
  const double parsed_value = std::stod(value, &parsed);
  if (parsed != value.size() || !std::isfinite(parsed_value)) {
    throw std::runtime_error(std::string(label) + " must be finite");
  }
  return parsed_value;
}

std::string numberToString(float value)
{
  std::ostringstream stream;
  stream << std::setprecision(9) << value;
  return stream.str();
}

std::array<char, 4> encodeFloat32Le(float value)
{
  static_assert(sizeof(float) == 4, "KWS backend payload requires 32-bit float");
  static_assert(
    std::numeric_limits<float>::is_iec559,
    "KWS backend payload requires IEEE-754 float");

  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return {
    static_cast<char>(bits & 0xFFu),
    static_cast<char>((bits >> 8u) & 0xFFu),
    static_cast<char>((bits >> 16u) & 0xFFu),
    static_cast<char>((bits >> 24u) & 0xFFu),
  };
}

void writeFloat32LeRaw(std::ofstream &audio_file, const std::vector<float> &samples)
{
  for (const float sample : samples) {
    if (!std::isfinite(sample)) {
      throw std::invalid_argument("KWS backend samples must be finite");
    }
    if (sample < -1.0f || sample > 1.0f) {
      throw std::invalid_argument("KWS backend samples must be normalized to [-1.0, 1.0]");
    }
    const std::array<char, 4> bytes = encodeFloat32Le(sample);
    audio_file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!audio_file.good()) {
      throw std::runtime_error("failed to write KWS backend float32le payload");
    }
  }
}

void requireReadableRegularFile(const char *config_name, const std::string &path_value)
{
  if (path_value.empty()) {
    throw std::invalid_argument(std::string(config_name) + " is required");
  }

  const std::filesystem::path path(path_value);
  std::error_code ec;
  if (!std::filesystem::is_regular_file(path, ec)) {
    throw std::invalid_argument(std::string(config_name) + " must be a regular readable file: " +
                                path_value);
  }

  std::ifstream probe(path, std::ios::binary);
  if (!probe.good()) {
    throw std::invalid_argument(std::string(config_name) + " is not readable: " + path_value);
  }
}

void requireExecutableCommand(const std::string &command)
{
  if (command.empty()) {
    throw std::invalid_argument("backend.command is required");
  }
  if (command.find('/') != std::string::npos) {
    if (access(command.c_str(), X_OK) != 0) {
      throw std::invalid_argument("backend.command is not executable: " + command);
    }
    return;
  }

  const char *path_env = std::getenv("PATH");
  if (path_env == nullptr || path_env[0] == '\0') {
    throw std::invalid_argument("backend.command not found on PATH: " + command);
  }

  std::istringstream paths(path_env);
  std::string directory;
  while (std::getline(paths, directory, ':')) {
    const std::filesystem::path candidate =
      directory.empty() ? std::filesystem::path(command) : std::filesystem::path(directory) / command;
    if (access(candidate.c_str(), X_OK) == 0) {
      return;
    }
  }
  throw std::invalid_argument("backend.command not found on PATH: " + command);
}

std::string lastNonEmptyLine(const std::string &text)
{
  std::istringstream stream(text);
  std::string line;
  std::string last;
  while (std::getline(stream, line)) {
    const std::string candidate = trim(line);
    if (!candidate.empty()) {
      last = candidate;
    }
  }
  return last;
}

void setNonBlocking(int fd)
{
  const int flags = fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    throw std::runtime_error("fcntl(F_GETFL) failed");
  }
  if (fcntl(fd, F_SETFL, flags | O_NONBLOCK) < 0) {
    throw std::runtime_error("fcntl(F_SETFL) failed");
  }
}

void appendAvailable(int fd, std::string &target)
{
  char buffer[4096];
  while (true) {
    const ssize_t count = read(fd, buffer, sizeof(buffer));
    if (count > 0) {
      target.append(buffer, static_cast<std::size_t>(count));
      continue;
    }
    if (count == 0) {
      return;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return;
    }
    if (errno == EINTR) {
      continue;
    }
    throw std::runtime_error("read from worker pipe failed");
  }
}

CommandResult runCommand(const std::string &command,
                         const std::vector<std::string> &args,
                         double timeout_sec,
                         const char *operation)
{
  if (!std::isfinite(timeout_sec) || timeout_sec <= 0.0) {
    throw std::invalid_argument("backend.timeout_sec must be finite and greater than zero");
  }

  int stdout_pipe[2];
  int stderr_pipe[2];
  if (pipe(stdout_pipe) != 0) {
    throw std::runtime_error("failed to create stdout pipe");
  }
  if (pipe(stderr_pipe) != 0) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    throw std::runtime_error("failed to create stderr pipe");
  }

  const pid_t pid = fork();
  if (pid < 0) {
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);
    throw std::runtime_error("failed to fork KWS backend worker");
  }

  if (pid == 0) {
    dup2(stdout_pipe[1], STDOUT_FILENO);
    dup2(stderr_pipe[1], STDERR_FILENO);
    close(stdout_pipe[0]);
    close(stdout_pipe[1]);
    close(stderr_pipe[0]);
    close(stderr_pipe[1]);

    std::vector<std::string> argv_storage;
    argv_storage.reserve(args.size() + 1);
    argv_storage.push_back(command);
    argv_storage.insert(argv_storage.end(), args.begin(), args.end());

    std::vector<char *> argv;
    argv.reserve(argv_storage.size() + 1);
    for (std::string &item : argv_storage) {
      argv.push_back(item.data());
    }
    argv.push_back(nullptr);
    execvp(command.c_str(), argv.data());
    _exit(127);
  }

  close(stdout_pipe[1]);
  close(stderr_pipe[1]);
  setNonBlocking(stdout_pipe[0]);
  setNonBlocking(stderr_pipe[0]);

  const auto start = std::chrono::steady_clock::now();
  std::string stdout_text;
  std::string stderr_text;
  int status = 0;
  bool exited = false;

  while (!exited) {
    appendAvailable(stdout_pipe[0], stdout_text);
    appendAvailable(stderr_pipe[0], stderr_text);

    const pid_t wait_result = waitpid(pid, &status, WNOHANG);
    if (wait_result == pid) {
      exited = true;
      break;
    }
    if (wait_result < 0 && errno != EINTR) {
      close(stdout_pipe[0]);
      close(stderr_pipe[0]);
      throw std::runtime_error("waitpid failed for KWS backend worker");
    }

    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    if (elapsed.count() > timeout_sec) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      close(stdout_pipe[0]);
      close(stderr_pipe[0]);
      throw std::runtime_error(std::string("KWS backend ") + operation + " timed out");
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }

  appendAvailable(stdout_pipe[0], stdout_text);
  appendAvailable(stderr_pipe[0], stderr_text);
  close(stdout_pipe[0]);
  close(stderr_pipe[0]);

  int exit_code = -1;
  if (WIFEXITED(status)) {
    exit_code = WEXITSTATUS(status);
  } else if (WIFSIGNALED(status)) {
    exit_code = 128 + WTERMSIG(status);
  }

  return CommandResult{exit_code, stdout_text, stderr_text};
}

void requireSuccessfulCommand(const CommandResult &result, const char *operation)
{
  if (result.exit_code == 0) {
    return;
  }
  throw std::runtime_error(
    std::string("KWS backend ") + operation + " failed: code=" +
    std::to_string(result.exit_code) + " stderr=" + trim(result.stderr_text));
}

}  // namespace

namespace fa_kws
{

bool isSupportedSherpaOnnxExecutionProvider(const std::string &execution_provider)
{
  return execution_provider == "cpu" ||
         execution_provider == "cuda" ||
         execution_provider == "coreml";
}

std::string supportedSherpaOnnxExecutionProvidersForMessage()
{
  return "cpu, cuda, coreml";
}

SherpaOnnxKwsBackend::SherpaOnnxKwsBackend(const SherpaOnnxKwsBackendConfig &config)
: config_(config),
  last_detect_time_(std::chrono::steady_clock::now()),
  has_detect_time_(false)
{
  validateConfig();
  const CommandResult health = runCommand(
    config_.command,
    formatArgs(config_.health_args, "", false),
    config_.timeout_sec,
    "health check");
  requireSuccessfulCommand(health, "health check");
}

SherpaOnnxKwsBackend::~SherpaOnnxKwsBackend() = default;

std::optional<KwsDetection> SherpaOnnxKwsBackend::process(
  const std::vector<float> &samples,
  std::int32_t sample_rate,
  std::chrono::steady_clock::time_point now)
{
  if (samples.empty()) {
    throw std::invalid_argument("KWS backend samples are required");
  }
  if (sample_rate != config_.target_sample_rate) {
    throw std::invalid_argument(
      "KWS backend sample_rate must match configured target_sample_rate " +
      std::to_string(config_.target_sample_rate) + ", got " + std::to_string(sample_rate));
  }

  std::filesystem::create_directories(config_.workspace_dir);
  const auto stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                       now.time_since_epoch())
                       .count();
  const std::filesystem::path audio_path =
    std::filesystem::path(config_.workspace_dir) /
    ("fa_kws_" + std::to_string(getpid()) + "_" + std::to_string(stamp) + ".f32");
  {
    std::ofstream audio_file(audio_path, std::ios::binary);
    if (!audio_file.good()) {
      throw std::runtime_error("failed to create KWS backend audio file: " + audio_path.string());
    }
    writeFloat32LeRaw(audio_file, samples);
  }

  try {
    const CommandResult result = runCommand(
      config_.command,
      formatArgs(config_.args, audio_path.string(), true),
      config_.timeout_sec,
      "inference");
    requireSuccessfulCommand(result, "inference");
    if (config_.cleanup_audio_files) {
      std::filesystem::remove(audio_path);
    }

    const std::string line = lastNonEmptyLine(result.stdout_text);
    if (line == "NO_DETECTION") {
      return std::nullopt;
    }
    const std::vector<std::string> fields = splitLine(line, '\t');
    if (fields.size() != 4 || fields[0] != "DETECTED") {
      throw std::runtime_error("KWS backend output must be NO_DETECTION or DETECTED<TAB>keyword<TAB>score<TAB>start_time_sec");
    }
    if (fields[1].empty()) {
      throw std::runtime_error("KWS backend detected keyword must be non-empty");
    }
    const float score = parseFiniteFloat(fields[2], "KWS backend score");
    if (score < 0.0f || score > 1.0f) {
      throw std::runtime_error("KWS backend score must be in [0.0, 1.0]");
    }
    const double start_time_sec = parseFiniteDouble(fields[3], "KWS backend start_time_sec");
    if (start_time_sec < 0.0) {
      throw std::runtime_error("KWS backend start_time_sec must be >= 0");
    }

    const auto elapsed_ms =
      has_detect_time_
        ? std::chrono::duration_cast<std::chrono::milliseconds>(now - last_detect_time_).count()
        : std::numeric_limits<long long>::max();
    if (elapsed_ms < config_.cooldown.count()) {
      return std::nullopt;
    }
    last_detect_time_ = now;
    has_detect_time_ = true;

    KwsDetection det;
    det.keyword = fields[1];
    det.score = score;
    det.start_time_sec = start_time_sec;
    return det;
  } catch (...) {
    if (config_.cleanup_audio_files) {
      std::filesystem::remove(audio_path);
    }
    throw;
  }
}

void SherpaOnnxKwsBackend::reset()
{
  has_detect_time_ = false;
}

void SherpaOnnxKwsBackend::resetHard()
{
  has_detect_time_ = false;
}

void SherpaOnnxKwsBackend::validateConfig() const
{
  if (config_.target_sample_rate <= 0) {
    throw std::invalid_argument("backend.target_sample_rate must be > 0");
  }
  if (config_.model_num_threads <= 0) {
    throw std::invalid_argument("backend.model_num_threads must be > 0");
  }
  if (config_.execution_provider.empty()) {
    throw std::invalid_argument("backend.execution_provider is required");
  }
  if (!isSupportedSherpaOnnxExecutionProvider(config_.execution_provider)) {
    throw std::invalid_argument(
      "unsupported backend.execution_provider for sherpa_onnx_kws: " +
      config_.execution_provider +
      "; supported providers: " +
      supportedSherpaOnnxExecutionProvidersForMessage());
  }
  requireReadableRegularFile("backend.encoder", config_.encoder_path);
  requireReadableRegularFile("backend.decoder", config_.decoder_path);
  requireReadableRegularFile("backend.joiner", config_.joiner_path);
  requireReadableRegularFile("backend.tokens", config_.tokens_path);
  requireReadableRegularFile("backend.keywords", config_.keywords_path);
  if (config_.max_active_paths <= 0) {
    throw std::invalid_argument("backend.max_active_paths must be > 0");
  }
  if (config_.num_trailing_blanks < 0) {
    throw std::invalid_argument("backend.num_trailing_blanks must be >= 0");
  }
  if (!std::isfinite(config_.keywords_score) || config_.keywords_score <= 0.0f) {
    throw std::invalid_argument("backend.keywords_score must be finite and > 0");
  }
  if (!std::isfinite(config_.keywords_threshold) || config_.keywords_threshold <= 0.0f) {
    throw std::invalid_argument("backend.keywords_threshold must be finite and > 0");
  }
  if (config_.cooldown.count() < 0) {
    throw std::invalid_argument("backend.cooldown must be >= 0 ms");
  }
  requireExecutableCommand(config_.command);
  if (!std::isfinite(config_.timeout_sec) || config_.timeout_sec <= 0.0) {
    throw std::invalid_argument("backend.timeout_sec must be finite and greater than zero");
  }
  if (config_.workspace_dir.empty()) {
    throw std::invalid_argument("backend.workspace_dir is required");
  }
  if (config_.args.empty()) {
    throw std::invalid_argument("backend.args must not be empty");
  }
  if (config_.health_args.empty()) {
    throw std::invalid_argument("backend.health_args must not be empty");
  }
  formatArgs(config_.args, "/tmp/fa_kws_contract_audio.f32", true);
  formatArgs(config_.health_args, "", false);
}

std::vector<std::string> SherpaOnnxKwsBackend::formatArgs(
  const std::vector<std::string> &template_args,
  const std::string &audio_path,
  bool allow_audio_placeholder) const
{
  const std::set<std::string> allowed_fields{
    "audio",
    "encoder",
    "decoder",
    "joiner",
    "tokens",
    "keywords",
    "provider",
    "sample_rate",
    "num_threads",
    "max_active_paths",
    "num_trailing_blanks",
    "keywords_score",
    "keywords_threshold",
  };
  std::set<std::string> seen_fields;
  std::vector<std::string> formatted;
  formatted.reserve(template_args.size());

  for (const std::string &arg : template_args) {
    if (arg.empty()) {
      throw std::invalid_argument("backend args must be non-empty strings");
    }
    std::string out;
    std::size_t cursor = 0;
    while (cursor < arg.size()) {
      const std::size_t open = arg.find('{', cursor);
      if (open == std::string::npos) {
        out += arg.substr(cursor);
        break;
      }
      out += arg.substr(cursor, open - cursor);
      const std::size_t close = arg.find('}', open + 1);
      if (close == std::string::npos) {
        throw std::invalid_argument("backend args contains malformed placeholder");
      }
      const std::string field = arg.substr(open + 1, close - open - 1);
      if (field.empty() || allowed_fields.count(field) == 0) {
        throw std::invalid_argument("unsupported backend args placeholder: " + field);
      }
      if (field == "audio" && !allow_audio_placeholder) {
        throw std::invalid_argument("backend.health_args must not include {audio}");
      }
      seen_fields.insert(field);
      if (field == "audio") {
        out += audio_path;
      } else if (field == "encoder") {
        out += config_.encoder_path;
      } else if (field == "decoder") {
        out += config_.decoder_path;
      } else if (field == "joiner") {
        out += config_.joiner_path;
      } else if (field == "tokens") {
        out += config_.tokens_path;
      } else if (field == "keywords") {
        out += config_.keywords_path;
      } else if (field == "provider") {
        out += config_.execution_provider;
      } else if (field == "sample_rate") {
        out += std::to_string(config_.target_sample_rate);
      } else if (field == "num_threads") {
        out += std::to_string(config_.model_num_threads);
      } else if (field == "max_active_paths") {
        out += std::to_string(config_.max_active_paths);
      } else if (field == "num_trailing_blanks") {
        out += std::to_string(config_.num_trailing_blanks);
      } else if (field == "keywords_score") {
        out += numberToString(config_.keywords_score);
      } else if (field == "keywords_threshold") {
        out += numberToString(config_.keywords_threshold);
      }
      cursor = close + 1;
    }
    formatted.push_back(out);
  }

  const std::set<std::string> required_fields =
    allow_audio_placeholder
      ? std::set<std::string>{"audio", "encoder", "decoder", "joiner", "tokens", "keywords", "provider", "sample_rate", "num_threads", "max_active_paths", "num_trailing_blanks", "keywords_score", "keywords_threshold"}
      : std::set<std::string>{"encoder", "decoder", "joiner", "tokens", "keywords", "provider", "sample_rate", "num_threads", "max_active_paths", "num_trailing_blanks", "keywords_score", "keywords_threshold"};
  for (const std::string &required : required_fields) {
    if (seen_fields.count(required) == 0) {
      throw std::invalid_argument("backend args missing required placeholder: " + required);
    }
  }

  return formatted;
}

}  // namespace fa_kws
