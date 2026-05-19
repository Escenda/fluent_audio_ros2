#include "fa_in/backends/network_pcm_receiver_backend.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{
struct TestEndpoint
{
  uint16_t port{0};
  std::string uri{};
};

class ScopedSocket
{
public:
  ScopedSocket()
  {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create test UDP socket: " + std::string(std::strerror(errno)));
    }
  }

  ~ScopedSocket()
  {
    if (socket_fd_ >= 0) {
      ::close(socket_fd_);
    }
  }

  ScopedSocket(const ScopedSocket &) = delete;
  ScopedSocket & operator=(const ScopedSocket &) = delete;

  int get() const
  {
    return socket_fd_;
  }

private:
  int socket_fd_{-1};
};

TestEndpoint reserveLoopbackEndpoint()
{
  ScopedSocket socket;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = 0;
  if (::bind(socket.get(), reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0) {
    throw std::runtime_error("failed to bind temporary UDP socket: " + std::string(std::strerror(errno)));
  }

  socklen_t address_length = sizeof(address);
  if (::getsockname(socket.get(), reinterpret_cast<sockaddr *>(&address), &address_length) < 0) {
    throw std::runtime_error("failed to inspect temporary UDP socket: " + std::string(std::strerror(errno)));
  }

  TestEndpoint endpoint;
  endpoint.port = ntohs(address.sin_port);
  endpoint.uri = "udp://127.0.0.1:" + std::to_string(endpoint.port);
  return endpoint;
}

fa_in::backends::AudioFormat pcm16MonoFormat()
{
  fa_in::backends::AudioFormat format;
  format.sample_rate = 16000;
  format.channels = 1;
  format.bit_depth = 16;
  format.chunk_ms = 20;
  format.encoding = "PCM16LE";
  format.layout = "interleaved";
  return format;
}

void sendUdpPacket(const uint16_t port, const std::vector<uint8_t> & payload)
{
  ScopedSocket socket;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  address.sin_port = htons(port);
  const uint8_t empty_payload_address = 0;
  const void * payload_data =
    payload.empty() ? static_cast<const void *>(&empty_payload_address) : payload.data();
  const auto sent = ::sendto(
    socket.get(),
    payload_data,
    payload.size(),
    0,
    reinterpret_cast<const sockaddr *>(&address),
    sizeof(address));
  if (sent < 0) {
    throw std::runtime_error("failed to send test UDP packet: " + std::string(std::strerror(errno)));
  }
  if (static_cast<size_t>(sent) != payload.size()) {
    throw std::runtime_error("test UDP send accepted fewer bytes than requested");
  }
}

template<typename Callable>
void expectBackendError(Callable action, const std::string & expected_message)
{
  try {
    action();
    FAIL() << "expected BackendError: " << expected_message;
  } catch (const fa_in::backends::BackendError & error) {
    EXPECT_STREQ(expected_message.c_str(), error.what());
  }
}
}  // namespace

TEST(NetworkPcmReceiverBackendTest, RejectsInvalidOpenConfiguration)
{
  const TestEndpoint endpoint = reserveLoopbackEndpoint();
  fa_in::backends::NetworkPcmReceiverBackend backend;

  expectBackendError(
    [&backend, &endpoint]() {
      static_cast<void>(backend.open(endpoint.uri, pcm16MonoFormat(), 0));
    },
    "network receiver requested frame count must be > 0");

  auto invalid_format = pcm16MonoFormat();
  invalid_format.bit_depth = 12;
  expectBackendError(
    [&backend, &endpoint, &invalid_format]() {
      static_cast<void>(backend.open(endpoint.uri, invalid_format, 2));
    },
    "audio.bit_depth must be byte aligned");
}

TEST(NetworkPcmReceiverBackendTest, RejectsInvalidEndpointUriOnOpen)
{
  const std::vector<std::pair<std::string, std::string>> cases{
    {"", "endpoint.uri is required"},
    {"tcp://127.0.0.1:40000", "endpoint.uri must use udp://host:port"},
    {"udp://127.0.0.1", "endpoint.uri must use udp://host:port"},
    {"udp://:40000", "endpoint.uri host is required"},
    {"udp://localhost:40000", "endpoint.uri host must be an IPv4 address"},
    {"udp://127.0.0.1:", "endpoint.uri port is required"},
    {"udp://127.0.0.1:abc", "endpoint.uri port must be numeric"},
    {"udp://127.0.0.1:0", "endpoint.uri port must be > 0"},
    {"udp://127.0.0.1:70000", "endpoint.uri port must be <= 65535"},
  };

  for (const auto & test_case : cases) {
    fa_in::backends::NetworkPcmReceiverBackend backend;
    expectBackendError(
      [&backend, &test_case]() {
        static_cast<void>(backend.open(test_case.first, pcm16MonoFormat(), 2));
      },
      test_case.second);
  }
}

TEST(NetworkPcmReceiverBackendTest, ReturnsNoDataWhenNoPacketIsAvailable)
{
  const TestEndpoint endpoint = reserveLoopbackEndpoint();
  fa_in::backends::NetworkPcmReceiverBackend backend;
  EXPECT_EQ(backend.open(endpoint.uri, pcm16MonoFormat(), 2), 2u);
  ASSERT_TRUE(backend.isOpen());

  std::vector<uint8_t> buffer(4);
  const auto result = backend.read(buffer.data(), 2);

  EXPECT_EQ(result.status, fa_in::backends::ReadStatus::kNoData);
  EXPECT_EQ(result.frames, 0u);
  EXPECT_TRUE(result.message.empty());
}

TEST(NetworkPcmReceiverBackendTest, RejectsInvalidReadInputs)
{
  const TestEndpoint endpoint = reserveLoopbackEndpoint();
  fa_in::backends::NetworkPcmReceiverBackend backend;
  EXPECT_EQ(backend.open(endpoint.uri, pcm16MonoFormat(), 2), 2u);

  auto null_destination = backend.read(nullptr, 2);
  EXPECT_EQ(null_destination.status, fa_in::backends::ReadStatus::kError);
  EXPECT_EQ(null_destination.message, "network PCM receiver destination is null");

  std::vector<uint8_t> buffer(4);
  auto zero_frames = backend.read(buffer.data(), 0);
  EXPECT_EQ(zero_frames.status, fa_in::backends::ReadStatus::kError);
  EXPECT_EQ(zero_frames.message, "network PCM receiver frame count must be > 0");
}

TEST(NetworkPcmReceiverBackendTest, RejectsEmptyPartialAndOversizedPackets)
{
  const TestEndpoint endpoint = reserveLoopbackEndpoint();
  fa_in::backends::NetworkPcmReceiverBackend backend;
  EXPECT_EQ(backend.open(endpoint.uri, pcm16MonoFormat(), 2), 2u);
  std::vector<uint8_t> buffer(4);

  sendUdpPacket(endpoint.port, {});
  const auto empty_packet = backend.read(buffer.data(), 2);
  EXPECT_EQ(empty_packet.status, fa_in::backends::ReadStatus::kError);
  EXPECT_EQ(empty_packet.message, "received empty UDP packet");

  sendUdpPacket(endpoint.port, {0x01, 0x02, 0x03});
  const auto partial_packet = backend.read(buffer.data(), 2);
  EXPECT_EQ(partial_packet.status, fa_in::backends::ReadStatus::kError);
  EXPECT_EQ(partial_packet.message, "received UDP packet byte size is not divisible by expected frame byte size");

  sendUdpPacket(endpoint.port, {0x01, 0x02, 0x03, 0x04, 0x05, 0x06});
  const auto oversized_packet = backend.read(buffer.data(), 2);
  EXPECT_EQ(oversized_packet.status, fa_in::backends::ReadStatus::kError);
  EXPECT_EQ(oversized_packet.message, "received UDP packet exceeds network.max_packet_bytes");
}

TEST(NetworkPcmReceiverBackendTest, ReceivesOneUdpPacketWithoutByteMutation)
{
  const TestEndpoint endpoint = reserveLoopbackEndpoint();
  fa_in::backends::NetworkPcmReceiverBackend backend;
  EXPECT_EQ(backend.open(endpoint.uri, pcm16MonoFormat(), 2), 2u);

  const std::vector<uint8_t> payload{0x10, 0x20, 0x30, 0x40};
  std::vector<uint8_t> buffer(4);
  sendUdpPacket(endpoint.port, payload);

  const auto result = backend.read(buffer.data(), 2);

  EXPECT_EQ(result.status, fa_in::backends::ReadStatus::kOk);
  EXPECT_EQ(result.frames, 2u);
  EXPECT_TRUE(result.message.empty());
  EXPECT_EQ(buffer, payload);
  EXPECT_EQ(backend.packetsReceived(), 1u);
  EXPECT_EQ(backend.bytesReceived(), payload.size());
}
