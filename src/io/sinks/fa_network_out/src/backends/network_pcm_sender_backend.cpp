#include "fa_network_out/backends/network_pcm_sender_backend.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/socket.h>
#include <unistd.h>

namespace fa_network_out::backends
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
}  // namespace

BackendError::BackendError(const std::string & message)
: std::runtime_error(message)
{
}

NetworkPcmSenderBackend::~NetworkPcmSenderBackend()
{
  close();
}

void NetworkPcmSenderBackend::open(const std::string & endpoint_uri)
{
  close();
  endpoint_address_ = parseUdpEndpoint(endpoint_uri);
  socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
  if (socket_fd_ < 0) {
    throw BackendError("failed to create UDP socket: " + std::string(std::strerror(errno)));
  }
  packets_sent_ = 0;
  bytes_sent_ = 0;
}

void NetworkPcmSenderBackend::close()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

void NetworkPcmSenderBackend::send(const uint8_t * data, const size_t byte_count)
{
  if (socket_fd_ < 0) {
    throw BackendError("network PCM sender is not open");
  }
  if (data == nullptr) {
    throw BackendError("network PCM sender data is null");
  }
  if (byte_count == 0) {
    throw BackendError("network PCM sender byte_count must be > 0");
  }
  if (byte_count > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
    throw BackendError("network PCM sender packet is too large");
  }

  const auto sent = ::sendto(
    socket_fd_,
    data,
    byte_count,
    0,
    reinterpret_cast<const sockaddr *>(&endpoint_address_),
    sizeof(endpoint_address_));
  if (sent < 0) {
    throw BackendError("failed to send UDP packet: " + std::string(std::strerror(errno)));
  }
  if (static_cast<size_t>(sent) != byte_count) {
    throw BackendError("UDP send accepted fewer bytes than requested");
  }
  packets_sent_ += 1;
  bytes_sent_ += static_cast<uint64_t>(byte_count);
}

bool NetworkPcmSenderBackend::isOpen() const
{
  return socket_fd_ >= 0;
}

uint64_t NetworkPcmSenderBackend::packetsSent() const
{
  return packets_sent_;
}

uint64_t NetworkPcmSenderBackend::bytesSent() const
{
  return bytes_sent_;
}

}  // namespace fa_network_out::backends
