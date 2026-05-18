#include "fa_network_in/fa_network_in_node.hpp"

#include <arpa/inet.h>
#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <netinet/in.h>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace
{
using namespace std::chrono_literals;

constexpr const char * kOutputTopic = "audio/test/network_in";
constexpr const char * kSourceId = "network_contract_source";
constexpr const char * kStreamId = "audio/test/network_in";

class LoopbackUdpPort
{
public:
  LoopbackUdpPort()
  {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create UDP socket for free port allocation");
    }

    sockaddr_in bind_address{};
    bind_address.sin_family = AF_INET;
    bind_address.sin_port = htons(0);
    if (inet_pton(AF_INET, "127.0.0.1", &bind_address.sin_addr) != 1) {
      throw std::runtime_error("failed to parse loopback address");
    }
    if (::bind(socket_fd_, reinterpret_cast<const sockaddr *>(&bind_address), sizeof(bind_address)) < 0) {
      throw std::runtime_error("failed to bind UDP socket for free port allocation");
    }

    sockaddr_in actual_address{};
    socklen_t actual_length = sizeof(actual_address);
    if (::getsockname(socket_fd_, reinterpret_cast<sockaddr *>(&actual_address), &actual_length) < 0) {
      throw std::runtime_error("failed to read allocated UDP port");
    }
    port_ = ntohs(actual_address.sin_port);
    ::close(socket_fd_);
    socket_fd_ = -1;
  }

  ~LoopbackUdpPort()
  {
    if (socket_fd_ >= 0) {
      ::close(socket_fd_);
    }
  }

  uint16_t port() const
  {
    return port_;
  }

  std::string endpointUri() const
  {
    return "udp://127.0.0.1:" + std::to_string(port_);
  }

private:
  int socket_fd_{-1};
  uint16_t port_{0};
};

class UdpSender
{
public:
  explicit UdpSender(const uint16_t port)
  {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create UDP sender socket");
    }
    endpoint_address_.sin_family = AF_INET;
    endpoint_address_.sin_port = htons(port);
    if (inet_pton(AF_INET, "127.0.0.1", &endpoint_address_.sin_addr) != 1) {
      throw std::runtime_error("failed to parse loopback address");
    }
  }

  ~UdpSender()
  {
    if (socket_fd_ >= 0) {
      ::close(socket_fd_);
    }
  }

  void send(const std::vector<uint8_t> & packet)
  {
    const auto sent = ::sendto(
      socket_fd_,
      packet.data(),
      packet.size(),
      0,
      reinterpret_cast<const sockaddr *>(&endpoint_address_),
      sizeof(endpoint_address_));
    if (sent < 0 || static_cast<size_t>(sent) != packet.size()) {
      throw std::runtime_error("failed to send UDP fixture packet");
    }
  }

private:
  int socket_fd_{-1};
  sockaddr_in endpoint_address_{};
};

std::vector<rclcpp::Parameter> validParameters(const std::string & endpoint_uri)
{
  return {
    rclcpp::Parameter("backend.name", "network_pcm_receiver"),
    rclcpp::Parameter("endpoint.uri", endpoint_uri),
    rclcpp::Parameter("transport.identity", "network_in_contract"),
    rclcpp::Parameter("output_topic", kOutputTopic),
    rclcpp::Parameter("audio.source_id", kSourceId),
    rclcpp::Parameter("audio.stream_id", kStreamId),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.bit_depth", 16),
    rclcpp::Parameter("expected.layout", "interleaved"),
    rclcpp::Parameter("network.max_packet_bytes", 1024),
    rclcpp::Parameter("polling.period_ms", 10),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", true),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
  };
}

rclcpp::NodeOptions optionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options;
  options.parameter_overrides(std::move(parameters));
  return options;
}

bool spinUntil(
  rclcpp::executors::SingleThreadedExecutor & executor,
  const std::function<bool()> & predicate)
{
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    executor.spin_some(10ms);
    std::this_thread::sleep_for(10ms);
  }
  return predicate();
}

class RclcppContractTest : public ::testing::Test
{
protected:
  void SetUp() override
  {
    if (!rclcpp::ok()) {
      int argc = 0;
      char ** argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  void TearDown() override
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

TEST_F(RclcppContractTest, FailsClosedWhenBackendNameIsUnknown)
{
  auto parameters = validParameters("udp://127.0.0.1:9");
  parameters[0] = rclcpp::Parameter("backend.name", "hidden_streamer");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_network_in::FaNetworkInNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenEndpointIsMissing)
{
  auto parameters = validParameters("udp://127.0.0.1:9");
  parameters[1] = rclcpp::Parameter("endpoint.uri", "");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_network_in::FaNetworkInNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenTransportIdentityIsMissing)
{
  auto parameters = validParameters("udp://127.0.0.1:9");
  parameters[2] = rclcpp::Parameter("transport.identity", "");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_network_in::FaNetworkInNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenEndpointHostIsNotExplicitIpv4)
{
  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_network_in::FaNetworkInNode>(
        optionsWith(validParameters("udp://localhost:9000")));
    },
    std::runtime_error);
}

TEST_F(RclcppContractTest, PublishesRawUdpPacketAsAudioFrameWithoutFormatMutation)
{
  LoopbackUdpPort allocated_port;
  const auto endpoint_uri = allocated_port.endpointUri();
  const auto port = allocated_port.port();

  auto node = std::make_shared<fa_network_in::FaNetworkInNode>(
    optionsWith(validParameters(endpoint_uri)));
  auto io_node = std::make_shared<rclcpp::Node>("fa_network_in_contract_io");

  std::vector<fa_interfaces::msg::AudioFrame> received;
  auto subscription = io_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    kOutputTopic,
    rclcpp::QoS(10).reliable(),
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received.push_back(*msg);
    });
  ASSERT_NE(subscription, nullptr);

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&subscription]() {
    return subscription->get_publisher_count() > 0;
  }));

  UdpSender sender(port);
  sender.send({0x10, 0x00, 0x20, 0x00});

  ASSERT_TRUE(spinUntil(executor, [&received]() {
    return !received.empty();
  }));
  ASSERT_EQ(received.size(), 1U);
  EXPECT_EQ(received[0].source_id, kSourceId);
  EXPECT_EQ(received[0].stream_id, kStreamId);
  EXPECT_EQ(received[0].encoding, "PCM16LE");
  EXPECT_EQ(received[0].sample_rate, 16000U);
  EXPECT_EQ(received[0].channels, 1U);
  EXPECT_EQ(received[0].bit_depth, 16U);
  EXPECT_EQ(received[0].layout, "interleaved");
  EXPECT_EQ(received[0].data, (std::vector<uint8_t>{0x10, 0x00, 0x20, 0x00}));
}

TEST_F(RclcppContractTest, FailsClosedWhenPacketPayloadSizeDoesNotMatchExpectedFrameSize)
{
  LoopbackUdpPort allocated_port;
  const auto endpoint_uri = allocated_port.endpointUri();
  const auto port = allocated_port.port();

  auto node = std::make_shared<fa_network_in::FaNetworkInNode>(
    optionsWith(validParameters(endpoint_uri)));

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  UdpSender sender(port);
  sender.send({0x10, 0x00, 0x20});

  ASSERT_TRUE(spinUntil(executor, [&node]() {
    return node->hasFatalError();
  }));
}
}  // namespace
