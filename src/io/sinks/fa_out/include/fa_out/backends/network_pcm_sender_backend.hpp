#pragma once

#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <string>

#include "fa_out/backends/sink_backend.hpp"

namespace fa_out::backends
{

struct NetworkPcmSenderConfig
{
  std::string endpoint_uri{};
  std::string encoding{};
  uint32_t channels{0};
  uint32_t bit_depth{0};
  size_t max_packet_bytes{0};
};

class NetworkPcmSenderBackend final : public SinkBackend
{
public:
  explicit NetworkPcmSenderBackend(NetworkPcmSenderConfig config);
  ~NetworkPcmSenderBackend() override;

  SinkOpenInfo open() override;
  void close() override;
  bool isOpen() const override;
  bool isRunning() const override;
  size_t writeFrames(const uint8_t * data, size_t frame_count) override;

  uint64_t packetsSent() const;
  uint64_t bytesSent() const;

private:
  NetworkPcmSenderConfig config_{};
  sockaddr_in endpoint_address_{};
  int socket_fd_{-1};
  size_t bytes_per_frame_{0};
  uint64_t packets_sent_{0};
  uint64_t bytes_sent_{0};
};

}  // namespace fa_out::backends
