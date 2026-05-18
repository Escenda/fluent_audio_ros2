#include "fa_network_in/backends/network_pcm_receiver_backend.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <limits>
#include <sys/socket.h>
#include <unistd.h>

namespace fa_network_in::backends
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

NetworkPcmReceiverBackend::~NetworkPcmReceiverBackend()
{
  close();
}

void NetworkPcmReceiverBackend::open(const std::string & endpoint_uri)
{
  close();
  bind_address_ = parseUdpEndpoint(endpoint_uri);
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
}

void NetworkPcmReceiverBackend::close()
{
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

ReceiveResult NetworkPcmReceiverBackend::receive(uint8_t * destination, const size_t max_byte_count)
{
  if (socket_fd_ < 0) {
    throw BackendError("network PCM receiver is not open");
  }
  if (destination == nullptr) {
    throw BackendError("network PCM receiver destination is null");
  }
  if (max_byte_count == 0) {
    throw BackendError("network PCM receiver max_byte_count must be > 0");
  }
  if (max_byte_count > static_cast<size_t>(std::numeric_limits<ssize_t>::max())) {
    throw BackendError("network PCM receiver max_byte_count is too large");
  }

  const auto received = ::recvfrom(
    socket_fd_,
    destination,
    max_byte_count,
    MSG_DONTWAIT | MSG_TRUNC,
    nullptr,
    nullptr);
  if (received < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return ReceiveResult{};
    }
    throw BackendError("failed to receive UDP packet: " + std::string(std::strerror(errno)));
  }
  if (received == 0) {
    throw BackendError("received empty UDP packet");
  }
  if (static_cast<size_t>(received) > max_byte_count) {
    throw BackendError("received UDP packet exceeds network.max_packet_bytes");
  }

  packets_received_ += 1;
  bytes_received_ += static_cast<uint64_t>(received);
  return ReceiveResult{true, static_cast<size_t>(received)};
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

}  // namespace fa_network_in::backends
