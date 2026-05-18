#pragma once

#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <stdexcept>
#include <string>

namespace fa_network_in::backends
{

class BackendError : public std::runtime_error
{
public:
  explicit BackendError(const std::string & message);
};

struct ReceiveResult
{
  bool has_packet{false};
  size_t byte_count{0};
};

class NetworkPcmReceiverBackend
{
public:
  NetworkPcmReceiverBackend() = default;
  ~NetworkPcmReceiverBackend();

  void open(const std::string & endpoint_uri);
  void close();
  ReceiveResult receive(uint8_t * destination, size_t max_byte_count);

  bool isOpen() const;
  uint64_t packetsReceived() const;
  uint64_t bytesReceived() const;

private:
  sockaddr_in bind_address_{};
  int socket_fd_{-1};
  uint64_t packets_received_{0};
  uint64_t bytes_received_{0};
};

}  // namespace fa_network_in::backends
