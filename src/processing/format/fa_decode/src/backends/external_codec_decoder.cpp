#include "fa_decode/backends/external_codec_decoder.hpp"

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

namespace fa_decode::backends
{

namespace
{
struct CommandResult
{
  DecodeStatus status{DecodeStatus::kOk};
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

DecodeStatus readFromFd(
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
        return DecodeStatus::kOutputTooLarge;
      }
      continue;
    }
    if (read_count == 0) {
      open = false;
      return DecodeStatus::kOk;
    }
    if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) {
      return DecodeStatus::kOk;
    }
    return DecodeStatus::kCommandReadFailed;
  }
}

CommandResult commandStartFailed()
{
  return CommandResult{DecodeStatus::kCommandStartFailed, -1, {}};
}

CommandResult runExternalCommand(
  const ExternalCodecDecoderConfig & config,
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
      return CommandResult{DecodeStatus::kCommandTimeout, -1, {}};
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
        return CommandResult{DecodeStatus::kCommandReadFailed, -1, {}};
      }
      for (nfds_t index = 0; index < fd_count; ++index) {
        if (fds[index].fd == stdin_pipe[1] && (fds[index].revents & POLLOUT) != 0) {
          if (!writeToFd(stdin_pipe[1], input, input_offset)) {
            closeFd(stdin_pipe[1]);
            closeFd(stdout_pipe[0]);
            terminateChild(pid);
            return CommandResult{DecodeStatus::kCommandWriteFailed, -1, {}};
          }
        }
        if (fds[index].fd == stdout_pipe[0] &&
            (fds[index].revents & (POLLIN | POLLHUP)) != 0)
        {
          const DecodeStatus status =
            readFromFd(stdout_pipe[0], output, max_output_bytes, stdout_open);
          if (status != DecodeStatus::kOk) {
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
        return CommandResult{DecodeStatus::kCommandFailed, -1, {}};
      }
    }
  }

  closeFd(stdin_pipe[1]);
  closeFd(stdout_pipe[0]);

  if (!WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
    const int exit_code = WIFEXITED(child_status) ? WEXITSTATUS(child_status) : -1;
    return CommandResult{DecodeStatus::kCommandFailed, exit_code, {}};
  }
  return CommandResult{DecodeStatus::kOk, WEXITSTATUS(child_status), std::move(output)};
}
}  // namespace

const char * encodedChunkContractStatusName(const EncodedChunkContractStatus status)
{
  switch (status) {
    case EncodedChunkContractStatus::kOk:
      return "ok";
    case EncodedChunkContractStatus::kInvalidCodec:
      return "invalid_codec";
    case EncodedChunkContractStatus::kInvalidContainer:
      return "invalid_container";
    case EncodedChunkContractStatus::kInvalidPayloadFormat:
      return "invalid_payload_format";
    case EncodedChunkContractStatus::kInvalidSampleRate:
      return "invalid_sample_rate";
    case EncodedChunkContractStatus::kInvalidChannels:
      return "invalid_channels";
    case EncodedChunkContractStatus::kInvalidDuration:
      return "invalid_duration";
    case EncodedChunkContractStatus::kEmptyData:
      return "empty_data";
  }
  throw std::logic_error("unhandled decoder encoded chunk contract status");
}

const char * decodeStatusMessage(const DecodeStatus status)
{
  switch (status) {
    case DecodeStatus::kOk:
      return "ok";
    case DecodeStatus::kInvalidChunkContract:
      return "invalid encoded chunk contract";
    case DecodeStatus::kEmptyInput:
      return "decoder input is empty";
    case DecodeStatus::kCommandStartFailed:
      return "failed to start external codec decoder command";
    case DecodeStatus::kCommandWriteFailed:
      return "failed to write encoded input to external codec decoder command";
    case DecodeStatus::kCommandReadFailed:
      return "failed to read PCM output from external codec decoder command";
    case DecodeStatus::kCommandTimeout:
      return "external codec decoder command timed out";
    case DecodeStatus::kCommandFailed:
      return "external codec decoder command failed";
    case DecodeStatus::kEmptyOutput:
      return "external codec decoder produced empty output";
    case DecodeStatus::kOutputTooLarge:
      return "external codec decoder output exceeded configured maximum";
    case DecodeStatus::kUnalignedOutput:
      return "external codec decoder output is not aligned to PCM frame size";
  }
  throw std::logic_error("unhandled decoder process status");
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

ExternalCodecDecoderBackend::ExternalCodecDecoderBackend(const ExternalCodecDecoderConfig & config)
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
  if (config_.input_codec.empty()) {
    throw std::runtime_error("input.codec is required");
  }
  if (config_.input_container.empty()) {
    throw std::runtime_error("input.container is required");
  }
  if (config_.input_payload_format.empty()) {
    throw std::runtime_error("input.payload_format is required");
  }
  if (config_.input_sample_rate <= 0) {
    throw std::runtime_error("input.sample_rate must be > 0");
  }
  if (config_.input_channels <= 0) {
    throw std::runtime_error("input.channels must be > 0");
  }
  if (config_.output_sample_rate != config_.input_sample_rate) {
    throw std::runtime_error("output.sample_rate must equal input.sample_rate");
  }
  if (config_.output_channels != config_.input_channels) {
    throw std::runtime_error("output.channels must equal input.channels");
  }
  if (!isSupportedPcmFormat(config_.output_encoding, config_.output_bit_depth)) {
    throw std::runtime_error("output encoding must be PCM16LE/16, PCM32LE/32, or FLOAT32LE/32");
  }
  if (config_.output_layout != kInterleavedLayout) {
    throw std::runtime_error("output.layout must be interleaved");
  }
}

EncodedChunkContractStatus ExternalCodecDecoderBackend::validateContract(
  const EncodedChunkContract & contract) const
{
  if (contract.codec != config_.input_codec) {
    return EncodedChunkContractStatus::kInvalidCodec;
  }
  if (contract.container != config_.input_container) {
    return EncodedChunkContractStatus::kInvalidContainer;
  }
  if (contract.payload_format != config_.input_payload_format) {
    return EncodedChunkContractStatus::kInvalidPayloadFormat;
  }
  if (contract.sample_rate != static_cast<uint32_t>(config_.input_sample_rate)) {
    return EncodedChunkContractStatus::kInvalidSampleRate;
  }
  if (contract.channels != static_cast<uint32_t>(config_.input_channels)) {
    return EncodedChunkContractStatus::kInvalidChannels;
  }
  if (contract.duration_ns == 0) {
    return EncodedChunkContractStatus::kInvalidDuration;
  }
  if (contract.data_size == 0) {
    return EncodedChunkContractStatus::kEmptyData;
  }
  return EncodedChunkContractStatus::kOk;
}

DecodeResult ExternalCodecDecoderBackend::decode(
  const std::vector<uint8_t> & input,
  const EncodedChunkContract & contract) const
{
  const EncodedChunkContractStatus contract_status = validateContract(contract);
  if (contract_status != EncodedChunkContractStatus::kOk) {
    return DecodeResult{
      DecodeStatus::kInvalidChunkContract,
      contract_status,
      -1,
      {},
      0,
      0,
      0,
      {},
      {}};
  }
  if (input.empty()) {
    return DecodeResult{
      DecodeStatus::kEmptyInput,
      EncodedChunkContractStatus::kOk,
      -1,
      {},
      0,
      0,
      0,
      {},
      {}};
  }

  CommandResult command_result = runExternalCommand(config_, input);
  if (command_result.status != DecodeStatus::kOk) {
    return DecodeResult{
      command_result.status,
      EncodedChunkContractStatus::kOk,
      command_result.exit_code,
      {},
      0,
      0,
      0,
      {},
      {}};
  }
  if (command_result.output.empty()) {
    return DecodeResult{
      DecodeStatus::kEmptyOutput,
      EncodedChunkContractStatus::kOk,
      command_result.exit_code,
      {},
      0,
      0,
      0,
      {},
      {}};
  }

  const size_t bytes_per_frame =
    static_cast<size_t>(config_.output_channels) * bytesPerSample(config_.output_bit_depth);
  if (bytes_per_frame == 0 || (command_result.output.size() % bytes_per_frame) != 0) {
    return DecodeResult{
      DecodeStatus::kUnalignedOutput,
      EncodedChunkContractStatus::kOk,
      command_result.exit_code,
      {},
      0,
      0,
      0,
      {},
      {}};
  }

  return DecodeResult{
    DecodeStatus::kOk,
    EncodedChunkContractStatus::kOk,
    command_result.exit_code,
    config_.output_encoding,
    static_cast<uint32_t>(config_.output_bit_depth),
    static_cast<uint32_t>(config_.output_sample_rate),
    static_cast<uint32_t>(config_.output_channels),
    config_.output_layout,
    std::move(command_result.output)};
}

}  // namespace fa_decode::backends
