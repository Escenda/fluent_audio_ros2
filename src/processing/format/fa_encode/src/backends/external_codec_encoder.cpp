#include "fa_encode/backends/external_codec_encoder.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdexcept>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <string>

namespace fa_encode::backends
{

namespace
{
struct CommandResult
{
  EncodeStatus status{EncodeStatus::kOk};
  int exit_code{-1};
  std::vector<uint8_t> output;
};

void closeFd(int & fd)
{
  if (fd >= 0) {
    ::close(fd);
    fd = -1;
  }
}

bool setNonBlocking(const int fd)
{
  const int flags = ::fcntl(fd, F_GETFL, 0);
  if (flags < 0) {
    return false;
  }
  return ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == 0;
}

void terminateChild(const pid_t pid)
{
  if (pid > 0) {
    ::kill(pid, SIGKILL);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
  }
}

bool writeToFd(const int fd, const std::vector<uint8_t> & input, size_t & offset)
{
  while (offset < input.size()) {
    const ssize_t written = ::write(fd, input.data() + offset, input.size() - offset);
    if (written > 0) {
      offset += static_cast<size_t>(written);
      continue;
    }
    if (written < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR)) {
      return true;
    }
    return false;
  }
  return true;
}

EncodeStatus readFromFd(
  const int fd,
  std::vector<uint8_t> & output,
  const size_t max_output_bytes,
  bool & open)
{
  std::array<uint8_t, 4096> buffer{};
  while (true) {
    const ssize_t read_count = ::read(fd, buffer.data(), buffer.size());
    if (read_count > 0) {
      output.insert(output.end(), buffer.begin(), buffer.begin() + read_count);
      if (output.size() > max_output_bytes) {
        return EncodeStatus::kOutputTooLarge;
      }
      continue;
    }
    if (read_count == 0) {
      open = false;
      return EncodeStatus::kOk;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
      return EncodeStatus::kOk;
    }
    return EncodeStatus::kCommandReadFailed;
  }
}

CommandResult commandStartFailed()
{
  return CommandResult{EncodeStatus::kCommandStartFailed, -1, {}};
}

CommandResult runExternalCommand(
  const ExternalCodecEncoderConfig & config,
  const std::vector<uint8_t> & input)
{
  int stdin_pipe[2] = {-1, -1};
  int stdout_pipe[2] = {-1, -1};
  if (::pipe(stdin_pipe) != 0 || ::pipe(stdout_pipe) != 0) {
    closeFd(stdin_pipe[0]);
    closeFd(stdin_pipe[1]);
    closeFd(stdout_pipe[0]);
    closeFd(stdout_pipe[1]);
    return commandStartFailed();
  }

  const pid_t pid = ::fork();
  if (pid < 0) {
    closeFd(stdin_pipe[0]);
    closeFd(stdin_pipe[1]);
    closeFd(stdout_pipe[0]);
    closeFd(stdout_pipe[1]);
    return commandStartFailed();
  }

  if (pid == 0) {
    ::dup2(stdin_pipe[0], STDIN_FILENO);
    ::dup2(stdout_pipe[1], STDOUT_FILENO);
    closeFd(stdin_pipe[0]);
    closeFd(stdin_pipe[1]);
    closeFd(stdout_pipe[0]);
    closeFd(stdout_pipe[1]);

    std::vector<char *> argv;
    argv.reserve(config.arguments.size() + 2U);
    argv.push_back(const_cast<char *>(config.executable.c_str()));
    for (const std::string & argument : config.arguments) {
      argv.push_back(const_cast<char *>(argument.c_str()));
    }
    argv.push_back(nullptr);

    ::execvp(config.executable.c_str(), argv.data());
    _exit(127);
  }

  closeFd(stdin_pipe[0]);
  closeFd(stdout_pipe[1]);
  if (!setNonBlocking(stdin_pipe[1]) || !setNonBlocking(stdout_pipe[0])) {
    closeFd(stdin_pipe[1]);
    closeFd(stdout_pipe[0]);
    terminateChild(pid);
    return commandStartFailed();
  }

  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::milliseconds(config.timeout_ms);
  const size_t max_output_bytes = static_cast<size_t>(config.max_output_bytes);

  std::vector<uint8_t> output;
  output.reserve(input.size());
  size_t input_offset = 0;
  bool stdin_open = true;
  bool stdout_open = true;
  bool child_exited = false;
  int child_status = 0;

  while (stdin_open || stdout_open || !child_exited) {
    if (std::chrono::steady_clock::now() >= deadline) {
      closeFd(stdin_pipe[1]);
      closeFd(stdout_pipe[0]);
      terminateChild(pid);
      return CommandResult{EncodeStatus::kCommandTimeout, -1, {}};
    }

    if (stdin_open && input_offset == input.size()) {
      closeFd(stdin_pipe[1]);
      stdin_open = false;
    }

    std::array<pollfd, 2> fds{};
    nfds_t fd_count = 0;
    if (stdin_open) {
      fds[fd_count].fd = stdin_pipe[1];
      fds[fd_count].events = POLLOUT;
      ++fd_count;
    }
    if (stdout_open) {
      fds[fd_count].fd = stdout_pipe[0];
      fds[fd_count].events = POLLIN;
      ++fd_count;
    }

    if (fd_count > 0) {
      const int poll_result = ::poll(fds.data(), fd_count, 10);
      if (poll_result < 0 && errno != EINTR) {
        closeFd(stdin_pipe[1]);
        closeFd(stdout_pipe[0]);
        terminateChild(pid);
        return CommandResult{EncodeStatus::kCommandReadFailed, -1, {}};
      }
      for (nfds_t index = 0; index < fd_count; ++index) {
        if (fds[index].fd == stdin_pipe[1] && (fds[index].revents & POLLOUT) != 0) {
          if (!writeToFd(stdin_pipe[1], input, input_offset)) {
            closeFd(stdin_pipe[1]);
            closeFd(stdout_pipe[0]);
            terminateChild(pid);
            return CommandResult{EncodeStatus::kCommandWriteFailed, -1, {}};
          }
        }
        if (fds[index].fd == stdout_pipe[0] &&
            (fds[index].revents & (POLLIN | POLLHUP)) != 0)
        {
          const EncodeStatus status =
            readFromFd(stdout_pipe[0], output, max_output_bytes, stdout_open);
          if (status != EncodeStatus::kOk) {
            closeFd(stdin_pipe[1]);
            closeFd(stdout_pipe[0]);
            terminateChild(pid);
            return CommandResult{status, -1, {}};
          }
          if (!stdout_open) {
            closeFd(stdout_pipe[0]);
          }
        }
      }
    }

    if (!child_exited) {
      const pid_t wait_result = ::waitpid(pid, &child_status, WNOHANG);
      if (wait_result == pid) {
        child_exited = true;
        if (stdin_open && input_offset < input.size()) {
          closeFd(stdin_pipe[1]);
          stdin_open = false;
        }
      } else if (wait_result < 0 && errno != EINTR) {
        closeFd(stdin_pipe[1]);
        closeFd(stdout_pipe[0]);
        return CommandResult{EncodeStatus::kCommandFailed, -1, {}};
      }
    }
  }

  closeFd(stdin_pipe[1]);
  closeFd(stdout_pipe[0]);

  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    const int exit_code = WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1;
    return CommandResult{EncodeStatus::kCommandFailed, exit_code, {}};
  }
  return CommandResult{EncodeStatus::kOk, WEXITSTATUS(child_status), std::move(output)};
}
}  // namespace

const char * frameContractStatusName(const FrameContractStatus status)
{
  switch (status) {
    case FrameContractStatus::kOk:
      return "ok";
    case FrameContractStatus::kInvalidSampleRate:
      return "invalid_sample_rate";
    case FrameContractStatus::kInvalidChannels:
      return "invalid_channels";
    case FrameContractStatus::kUnsupportedInputEncoding:
      return "unsupported_input_encoding";
    case FrameContractStatus::kUnsupportedInputBitDepth:
      return "unsupported_input_bit_depth";
    case FrameContractStatus::kUnsupportedLayout:
      return "unsupported_layout";
    case FrameContractStatus::kEmptyData:
      return "empty_data";
    case FrameContractStatus::kUnalignedData:
      return "unaligned_data";
  }
  return "unknown";
}

const char * encodeStatusMessage(const EncodeStatus status)
{
  switch (status) {
    case EncodeStatus::kOk:
      return "ok";
    case EncodeStatus::kInvalidFrameContract:
      return "invalid PCM frame contract";
    case EncodeStatus::kEmptyInput:
      return "encoder input is empty";
    case EncodeStatus::kCommandStartFailed:
      return "failed to start external codec encoder command";
    case EncodeStatus::kCommandWriteFailed:
      return "failed to write PCM input to external codec encoder command";
    case EncodeStatus::kCommandReadFailed:
      return "failed to read encoded output from external codec encoder command";
    case EncodeStatus::kCommandTimeout:
      return "external codec encoder command timed out";
    case EncodeStatus::kCommandFailed:
      return "external codec encoder command failed";
    case EncodeStatus::kEmptyOutput:
      return "external codec encoder produced empty output";
    case EncodeStatus::kOutputTooLarge:
      return "external codec encoder output exceeded configured maximum";
  }
  return "unknown encoder status";
}

bool isSupportedPcmFormat(const std::string & encoding, const int bit_depth)
{
  return (encoding == kEncodingPcm16Le && bit_depth == 16) ||
         (encoding == kEncodingPcm32Le && bit_depth == 32) ||
         (encoding == kEncodingFloat32Le && bit_depth == 32);
}

size_t bytesPerSample(const int bit_depth)
{
  if (bit_depth <= 0 || (bit_depth % 8) != 0) {
    return 0;
  }
  return static_cast<size_t>(bit_depth / 8);
}

ExternalCodecEncoderBackend::ExternalCodecEncoderBackend(const ExternalCodecEncoderConfig & config)
: config_(config)
{
  if (config_.executable.empty()) {
    throw std::runtime_error("backend.command.executable is required");
  }
  if (config_.timeout_ms <= 0) {
    throw std::runtime_error("backend.command.timeout_ms must be > 0");
  }
  if (config_.max_output_bytes <= 0) {
    throw std::runtime_error("backend.command.max_output_bytes must be > 0");
  }
  if (config_.input_sample_rate <= 0) {
    throw std::runtime_error("input.sample_rate must be > 0");
  }
  if (config_.input_channels <= 0) {
    throw std::runtime_error("input.channels must be > 0");
  }
  if (!isSupportedPcmFormat(config_.input_encoding, config_.input_bit_depth)) {
    throw std::runtime_error("input encoding must be PCM16LE/16, PCM32LE/32, or FLOAT32LE/32");
  }
  if (config_.input_layout != kInterleavedLayout) {
    throw std::runtime_error("input.layout must be interleaved");
  }
  if (config_.output_codec.empty()) {
    throw std::runtime_error("output.codec is required");
  }
  if (config_.output_container.empty()) {
    throw std::runtime_error("output.container is required");
  }
  if (config_.output_payload_format.empty()) {
    throw std::runtime_error("output.payload_format is required");
  }
}

FrameContractStatus ExternalCodecEncoderBackend::validateContract(
  const PcmFrameContract & contract) const
{
  if (contract.sample_rate != static_cast<uint32_t>(config_.input_sample_rate)) {
    return FrameContractStatus::kInvalidSampleRate;
  }
  if (contract.channels != static_cast<uint32_t>(config_.input_channels)) {
    return FrameContractStatus::kInvalidChannels;
  }
  if (contract.encoding != config_.input_encoding) {
    return FrameContractStatus::kUnsupportedInputEncoding;
  }
  if (contract.bit_depth != static_cast<uint32_t>(config_.input_bit_depth)) {
    return FrameContractStatus::kUnsupportedInputBitDepth;
  }
  if (contract.layout != config_.input_layout) {
    return FrameContractStatus::kUnsupportedLayout;
  }
  if (contract.data_size == 0) {
    return FrameContractStatus::kEmptyData;
  }
  const size_t bytes_per_frame =
    static_cast<size_t>(config_.input_channels) * bytesPerSample(config_.input_bit_depth);
  if (bytes_per_frame == 0 || (contract.data_size % bytes_per_frame) != 0) {
    return FrameContractStatus::kUnalignedData;
  }
  return FrameContractStatus::kOk;
}

EncodeResult ExternalCodecEncoderBackend::encode(
  const std::vector<uint8_t> & input,
  const PcmFrameContract & contract) const
{
  const FrameContractStatus contract_status = validateContract(contract);
  if (contract_status != FrameContractStatus::kOk) {
    return EncodeResult{EncodeStatus::kInvalidFrameContract, contract_status, -1, {}, {}, {}, 0, 0, {}};
  }
  if (input.empty()) {
    return EncodeResult{EncodeStatus::kEmptyInput, FrameContractStatus::kOk, -1, {}, {}, {}, 0, 0, {}};
  }

  CommandResult command_result = runExternalCommand(config_, input);
  if (command_result.status != EncodeStatus::kOk) {
    return EncodeResult{
      command_result.status,
      FrameContractStatus::kOk,
      command_result.exit_code,
      {},
      {},
      {},
      0,
      0,
      {}};
  }
  if (command_result.output.empty()) {
    return EncodeResult{
      EncodeStatus::kEmptyOutput,
      FrameContractStatus::kOk,
      command_result.exit_code,
      {},
      {},
      {},
      0,
      0,
      {}};
  }

  return EncodeResult{
    EncodeStatus::kOk,
    FrameContractStatus::kOk,
    command_result.exit_code,
    config_.output_codec,
    config_.output_container,
    config_.output_payload_format,
    static_cast<uint32_t>(config_.input_sample_rate),
    static_cast<uint32_t>(config_.input_channels),
    std::move(command_result.output)};
}

}  // namespace fa_encode::backends
