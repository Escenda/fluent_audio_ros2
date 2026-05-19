#include "fa_out/backends/network_pcm_sender_backend.hpp"

#include <arpa/inet.h>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace
{
class UdpReceiver
{
public:
  UdpReceiver()
  {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create test UDP socket: " + std::string(std::strerror(errno)));
    }

    timeval timeout{};
    timeout.tv_sec = 1;
    if (::setsockopt(socket_fd_, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout)) < 0) {
      close();
      throw std::runtime_error("failed to set test UDP receive timeout: " + std::string(std::strerror(errno)));
    }

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = 0;
    if (::bind(socket_fd_, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) < 0) {
      close();
      throw std::runtime_error("failed to bind test UDP socket: " + std::string(std::strerror(errno)));
    }

    socklen_t address_length = sizeof(address);
    if (::getsockname(socket_fd_, reinterpret_cast<sockaddr *>(&address), &address_length) < 0) {
      close();
      throw std::runtime_error("failed to inspect test UDP socket: " + std::string(std::strerror(errno)));
    }
    port_ = ntohs(address.sin_port);
  }

  ~UdpReceiver()
  {
    close();
  }

  UdpReceiver(const UdpReceiver &) = delete;
  UdpReceiver & operator=(const UdpReceiver &) = delete;

  std::string endpointUri() const
  {
    return "udp://127.0.0.1:" + std::to_string(port_);
  }

  std::vector<uint8_t> receive(const size_t max_bytes)
  {
    std::vector<uint8_t> buffer(max_bytes);
    const auto received = ::recvfrom(socket_fd_, buffer.data(), buffer.size(), 0, nullptr, nullptr);
    if (received < 0) {
      throw std::runtime_error("failed to receive test UDP packet: " + std::string(std::strerror(errno)));
    }
    buffer.resize(static_cast<size_t>(received));
    return buffer;
  }

private:
  void close()
  {
    if (socket_fd_ >= 0) {
      ::close(socket_fd_);
      socket_fd_ = -1;
    }
  }

  int socket_fd_{-1};
  uint16_t port_{0};
};

fa_out::backends::NetworkPcmSenderConfig validConfig(const std::string & endpoint_uri)
{
  fa_out::backends::NetworkPcmSenderConfig config;
  config.endpoint_uri = endpoint_uri;
  config.encoding = "PCM16LE";
  config.channels = 1;
  config.bit_depth = 16;
  config.max_packet_bytes = 8;
  return config;
}

template<typename Callable>
void expectSinkBackendError(Callable action, const std::string & expected_message)
{
  try {
    action();
    FAIL() << "expected SinkBackendError: " << expected_message;
  } catch (const fa_out::backends::SinkBackendError & error) {
    EXPECT_STREQ(expected_message.c_str(), error.what());
  }
}
}  // namespace

TEST(NetworkPcmSenderBackendTest, RejectsInvalidPacketSizeConfiguration)
{
  auto zero_packet_config = validConfig("udp://127.0.0.1:40000");
  zero_packet_config.max_packet_bytes = 0;
  expectSinkBackendError(
    [&zero_packet_config]() {
      fa_out::backends::NetworkPcmSenderBackend backend(zero_packet_config);
    },
    "network.max_packet_bytes must be > 0");

  auto unaligned_packet_config = validConfig("udp://127.0.0.1:40000");
  unaligned_packet_config.max_packet_bytes = 3;
  expectSinkBackendError(
    [&unaligned_packet_config]() {
      fa_out::backends::NetworkPcmSenderBackend backend(unaligned_packet_config);
    },
    "network.max_packet_bytes must be divisible by expected frame byte size");
}

TEST(NetworkPcmSenderBackendTest, RejectsInvalidEndpointUriOnOpen)
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
    auto config = validConfig(test_case.first);
    fa_out::backends::NetworkPcmSenderBackend backend(config);
    expectSinkBackendError(
      [&backend]() {
        static_cast<void>(backend.open());
      },
      test_case.second);
  }
}

TEST(NetworkPcmSenderBackendTest, RejectsPacketLargerThanConfiguredMaximum)
{
  UdpReceiver receiver;
  auto config = validConfig(receiver.endpointUri());
  config.max_packet_bytes = 4;
  fa_out::backends::NetworkPcmSenderBackend backend(config);
  static_cast<void>(backend.open());

  const std::vector<uint8_t> payload{0x01, 0x02, 0x03, 0x04, 0x05, 0x06};
  expectSinkBackendError(
    [&backend, &payload]() {
      static_cast<void>(backend.writeFrames(payload.data(), 3));
    },
    "network PCM sender packet exceeds network.max_packet_bytes");
}

TEST(NetworkPcmSenderBackendTest, SendsOneAcceptedFrameAsOneUdpPacketWithoutByteMutation)
{
  UdpReceiver receiver;
  fa_out::backends::NetworkPcmSenderBackend backend(validConfig(receiver.endpointUri()));
  const auto open_info = backend.open();
  ASSERT_TRUE(backend.isOpen());
  ASSERT_TRUE(backend.isRunning());
  ASSERT_EQ(open_info.info_messages.size(), 1u);

  const std::vector<uint8_t> payload{0x10, 0x20, 0x30, 0x40};
  EXPECT_EQ(backend.writeFrames(payload.data(), 2), 2u);
  EXPECT_EQ(backend.packetsSent(), 1u);
  EXPECT_EQ(backend.bytesSent(), payload.size());

  EXPECT_EQ(receiver.receive(16), payload);
}
