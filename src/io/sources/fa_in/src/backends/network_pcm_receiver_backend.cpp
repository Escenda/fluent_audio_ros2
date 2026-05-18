#include "fa_in/backends/network_pcm_receiver_backend.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/socket.h>
#include <unistd.h>

namespace fa_in::backends
{

namespace
{
constexpr const char * kUdpPrefix = "udp://";

uint16_t parsePort(const std::string & port_text)
{
  if (port_text.empty()) {
    throw BackendError("endpoint.uri port is required");
  }
  unsigned long value = 0;
  for (const char character : port_text) {
    if (character < '0' || character > '9') {
      throw BackendError("endpoint.uri port must be numeric");
    }
    value = (value * 10UL) + static_cast<unsigned long>(character - '0');
    if (value > 65535UL) {
      throw BackendError("endpoint.uri port must be <= 65535");
    }
  }
  if (value == 0UL) {
    throw BackendError("endpoint.uri port must be > 0");
  }
  return static_cast<uint16_t>(value);
}

sockaddr_in parseUdpEndpoint(const std::string & endpoint_uri)
{
  if (endpoint_uri.empty()) {
    throw BackendError("endpoint.uri is required");
  }
  if (endpoint_uri.rfind(kUdpPrefix, 0) != 0) {
    throw BackendError("endpoint.uri must use udp://host:port");
  }

  const std::string endpoint = endpoint_uri.substr(std::strlen(kUdpPrefix));
  const auto separator = endpoint.rfind(':');
  if (separator == std::string::npos) {
    throw BackendError("endpoint.uri must use udp://host:port");
  }
  const std::string host = endpoint.substr(0, separator);
  const std::string port = endpoint.substr(separator + 1);
  if (host.empty()) {
    throw BackendError("endpoint.uri host is required");
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(parsePort(port));
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    throw BackendError("endpoint.uri host must be an IPv4 address");
  }
  return address;
}

size_t bytesPerFrame(const AudioFormat & format)
{
  if (format.channels == 0 || format.bit_depth == 0) {
    throw BackendError("audio.channels and audio.bit_depth must be > 0");
  }
  if ((format.bit_depth % 8U) != 0) {
    throw BackendError("audio.bit_depth must be byte aligned");
  }
  const size_t bytes_per_sample = static_cast<size_t>(format.bit_depth / 8U);
  if (static_cast<size_t>(format.channels) >
    std::numeric_limits<size_t>::max() / bytes_per_sample)
  {
    throw BackendError("audio.channels * audio.bit_depth exceeds size_t range");
  }
  return static_cast<size_t>(format.channels) * bytes_per_sample;
}
}  // namespace

NetworkPcmReceiverBackend::~NetworkPcmReceiverBackend()
{
  close();
}

std::vector<DeviceInfo> NetworkPcmReceiverBackend::listDevices() const
{
  if (endpoint_uri_.empty()) {
    return {};
  }
  return {DeviceInfo{endpoint_uri_, "network_pcm_receiver", format_.channels, format_.sample_rate}};
}

DeviceInfo NetworkPcmReceiverBackend::selectDevice(const DeviceSelector & selector) const
{
  if (selector.identifier.empty()) {
    throw BackendError("endpoint.uri is required");
  }
  return DeviceInfo{selector.identifier, "network_pcm_receiver", format_.channels, format_.sample_rate};
}

size_t NetworkPcmReceiverBackend::open(
  const std::string & device_id,
  const AudioFormat & format,
  const size_t requested_frames)
{
  close();
  if (requested_frames == 0) {
    throw BackendError("network receiver requested frame count must be > 0");
  }
  endpoint_uri_ = device_id;
  format_ = format;
  bytes_per_frame_ = bytesPerFrame(format_);
  if (requested_frames > std::numeric_limits<size_t>::max() / bytes_per_frame_) {
    throw BackendError("network.max_packet_bytes exceeds size_t range");
  }
  max_packet_bytes_ = requested_frames * bytes_per_frame_;
  bind_address_ = parseUdpEndpoint(endpoint_uri_);
  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0) {
    throw BackendError("failed to create UDP socket: " + std::string(std::strerror(errno)));
  }
  if (::bind(socket_fd_, reinterpret_cast<const sockaddr *>(&bind_address_), sizeof(bind_address_)) < 0) {
    const std::string message = std::strerror(errno);
    close();
    throw BackendError("failed to bind UDP endpoint: " + message);
  }
  packets_received_ = 0;
  bytes_received_ = 0;
  return requested_frames;
}

void NetworkPcmReceiverBackend::close()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

void NetworkPcmReceiverBackend::drop()
{
  close();
}

ReadResult NetworkPcmReceiverBackend::read(uint8_t * data, const size_t frames)
{
  if (socket_fd_ < 0) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver is not open"};
  }
  if (data == nullptr) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver destination is null"};
  }
  if (frames == 0) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver frame count must be > 0"};
  }
  if (bytes_per_frame_ == 0) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver bytes_per_frame is zero"};
  }
  if (frames > std::numeric_limits<size_t>::max() / bytes_per_frame_) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver byte count is too large"};
  }
  const size_t max_byte_count = frames * bytes_per_frame_;
  if (max_byte_count > max_packet_bytes_) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver max byte count exceeds configured packet size"};
  }
  if (max_byte_count > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
    return ReadResult{ReadStatus::kError, 0, "network PCM receiver max byte count is too large"};
  }

  const auto received = ::recvfrom(
    socket_fd_,
    data,
    max_byte_count,
    MSG_DONTWAIT | MSG_TRUNC,
    nullptr,
    nullptr);
  if (received < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return ReadResult{ReadStatus::kNoData, 0, ""};
    }
    return ReadResult{
      ReadStatus::kError, 0, "failed to receive UDP packet: " + std::string(std::strerror(errno))};
  }
  if (received == 0) {
    return ReadResult{ReadStatus::kError, 0, "received empty UDP packet"};
  }
  if (static_cast<size_t>(received) > max_byte_count) {
    return ReadResult{ReadStatus::kError, 0, "received UDP packet exceeds network.max_packet_bytes"};
  }
  if ((static_cast<size_t>(received) % bytes_per_frame_) != 0) {
    return ReadResult{
      ReadStatus::kError, 0, "received UDP packet byte size is not divisible by expected frame byte size"};
  }

  packets_received_ += 1;
  bytes_received_ += static_cast<uint64_t>(received);
  return ReadResult{ReadStatus::kOk, static_cast<size_t>(received) / bytes_per_frame_, ""};
}

bool NetworkPcmReceiverBackend::isOpen() const
{
  return socket_fd_ >= 0;
}

uint64_t NetworkPcmReceiverBackend::packetsReceived() const
{
  return packets_received_;
}

uint64_t NetworkPcmReceiverBackend::bytesReceived() const
{
  return bytes_received_;
}

}  // namespace fa_in::backends
