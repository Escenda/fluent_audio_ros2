#include "fa_out/backends/network_pcm_sender_backend.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>

namespace fa_out::backends
{

namespace
{
constexpr const char * kUdpPrefix = "udp://";

uint16_t parsePort(const std::string & port_text)
{
  if (port_text.empty()) {
    throw SinkBackendError("endpoint.uri port is required");
  }
  unsigned long value = 0;
  for (const char character : port_text) {
    if (character < '0' || character > '9') {
      throw SinkBackendError("endpoint.uri port must be numeric");
    }
    value = (value * 10UL) + static_cast<unsigned long>(character - '0');
    if (value > 65535UL) {
      throw SinkBackendError("endpoint.uri port must be <= 65535");
    }
  }
  if (value == 0UL) {
    throw SinkBackendError("endpoint.uri port must be > 0");
  }
  return static_cast<uint16_t>(value);
}

sockaddr_in parseUdpEndpoint(const std::string & endpoint_uri)
{
  if (endpoint_uri.empty()) {
    throw SinkBackendError("endpoint.uri is required");
  }
  if (endpoint_uri.rfind(kUdpPrefix, 0) != 0) {
    throw SinkBackendError("endpoint.uri must use udp://host:port");
  }

  const std::string endpoint = endpoint_uri.substr(std::strlen(kUdpPrefix));
  const auto separator = endpoint.rfind(':');
  if (separator == std::string::npos) {
    throw SinkBackendError("endpoint.uri must use udp://host:port");
  }
  const std::string host = endpoint.substr(0, separator);
  const std::string port = endpoint.substr(separator + 1);
  if (host.empty()) {
    throw SinkBackendError("endpoint.uri host is required");
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(parsePort(port));
  if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
    throw SinkBackendError("endpoint.uri host must be an IPv4 address");
  }
  return address;
}

size_t bytesPerFrame(const NetworkPcmSenderConfig & config)
{
  if (config.channels == 0 || config.bit_depth == 0) {
    throw SinkBackendError("audio.channels and audio.bit_depth must be > 0");
  }
  if ((config.bit_depth % 8U) != 0) {
    throw SinkBackendError("audio.bit_depth must be byte aligned");
  }
  const size_t bytes_per_sample = static_cast<size_t>(config.bit_depth / 8U);
  if (static_cast<size_t>(config.channels) >
    std::numeric_limits<size_t>::max() / bytes_per_sample)
  {
    throw SinkBackendError("audio.channels * audio.bit_depth exceeds size_t range");
  }
  return static_cast<size_t>(config.channels) * bytes_per_sample;
}
}  // namespace

NetworkPcmSenderBackend::NetworkPcmSenderBackend(NetworkPcmSenderConfig config)
: config_(std::move(config))
{
  bytes_per_frame_ = bytesPerFrame(config_);
  if (config_.max_packet_bytes == 0) {
    throw SinkBackendError("network.max_packet_bytes must be > 0");
  }
  if ((config_.max_packet_bytes % bytes_per_frame_) != 0) {
    throw SinkBackendError("network.max_packet_bytes must be divisible by expected frame byte size");
  }
}

NetworkPcmSenderBackend::~NetworkPcmSenderBackend()
{
  close();
}

SinkOpenInfo NetworkPcmSenderBackend::open()
{
  close();
  endpoint_address_ = parseUdpEndpoint(config_.endpoint_uri);
  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0) {
    throw SinkBackendError("failed to create UDP socket: " + std::string(std::strerror(errno)));
  }
  packets_sent_ = 0;
  bytes_sent_ = 0;
  SinkOpenInfo info;
  info.info_messages.push_back("network PCM sender opened endpoint " + config_.endpoint_uri);
  return info;
}

void NetworkPcmSenderBackend::close()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

bool NetworkPcmSenderBackend::isOpen() const
{
  return socket_fd_ >= 0;
}

bool NetworkPcmSenderBackend::isRunning() const
{
  return isOpen();
}

size_t NetworkPcmSenderBackend::writeFrames(const uint8_t * data, const size_t frame_count)
{
  if (socket_fd_ < 0) {
    throw SinkBackendError("network PCM sender is not open");
  }
  if (data == nullptr) {
    throw SinkBackendError("network PCM sender data is null");
  }
  if (frame_count == 0) {
    throw SinkBackendError("network PCM sender frame_count must be > 0");
  }
  if (frame_count > std::numeric_limits<size_t>::max() / bytes_per_frame_) {
    throw SinkBackendError("network PCM sender byte count is too large");
  }
  const size_t byte_count = frame_count * bytes_per_frame_;
  if (byte_count > config_.max_packet_bytes) {
    throw SinkBackendError("network PCM sender packet exceeds network.max_packet_bytes");
  }
  if (byte_count > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
    throw SinkBackendError("network PCM sender packet is too large");
  }

  const auto sent = ::sendto(
    socket_fd_,
    data,
    byte_count,
    0,
    reinterpret_cast<const sockaddr *>(&endpoint_address_),
    sizeof(endpoint_address_));
  if (sent < 0) {
    throw SinkBackendError("failed to send UDP packet: " + std::string(std::strerror(errno)));
  }
  if (static_cast<size_t>(sent) != byte_count) {
    throw SinkBackendError("UDP send accepted fewer bytes than requested");
  }
  packets_sent_ += 1;
  bytes_sent_ += static_cast<uint64_t>(byte_count);
  return frame_count;
}

uint64_t NetworkPcmSenderBackend::packetsSent() const
{
  return packets_sent_;
}

uint64_t NetworkPcmSenderBackend::bytesSent() const
{
  return bytes_sent_;
}

}  // namespace fa_out::backends
