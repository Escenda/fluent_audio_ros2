#pragma once

#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <stdexcept>
#include <string>

namespace fa_network_out::backends
{

class BackendError : public std::runtime_error
{
public:
  explicit BackendError(const std::string & message);
};

class NetworkPcmSenderBackend
{
public:
  NetworkPcmSenderBackend() = default;
  ~NetworkPcmSenderBackend();

  void open(const std::string & endpoint_uri);
  void close();
  void send(const uint8_t * data, size_t byte_count);

  bool isOpen() const;
  uint64_t packetsSent() const;
  uint64_t bytesSent() const;

private:
  sockaddr_in endpoint_address_{};
  int socket_fd_{-1};
  uint64_t packets_sent_{0};
  uint64_t bytes_sent_{0};
};

}  // namespace fa_network_out::backends
