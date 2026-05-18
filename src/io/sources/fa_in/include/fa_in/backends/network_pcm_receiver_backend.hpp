#pragma once

#include <cstddef>
#include <cstdint>
#include <netinet/in.h>
#include <string>
#include <vector>

#include "fa_in/backends/source_backend.hpp"

namespace fa_in::backends
{

class NetworkPcmReceiverBackend final : public SourceBackend
{
public:
  NetworkPcmReceiverBackend() = default;
  ~NetworkPcmReceiverBackend() override;

  std::vector<DeviceInfo> listDevices() const override;
  DeviceInfo selectDevice(const DeviceSelector & selector) const override;
  size_t open(const std::string & device_id, const AudioFormat & format, size_t requested_frames) override;
  void close() override;
  void drop() override;
  ReadResult read(uint8_t * data, size_t frames) override;

  bool isOpen() const;
  uint64_t packetsReceived() const;
  uint64_t bytesReceived() const;

private:
  sockaddr_in bind_address_{};
  int socket_fd_{-1};
  std::string endpoint_uri_{};
  AudioFormat format_{};
  size_t bytes_per_frame_{0};
  size_t max_packet_bytes_{0};
  uint64_t packets_received_{0};
  uint64_t bytes_received_{0};
};

}  // namespace fa_in::backends
