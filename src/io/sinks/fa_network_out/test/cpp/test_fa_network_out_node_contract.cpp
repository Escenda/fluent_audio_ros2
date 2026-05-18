#include "fa_network_out/fa_network_out_node.hpp"

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

constexpr const char * kInputTopic = "audio/test/network_out";

class UdpReceiver
{
public:
  UdpReceiver()
  {
    socket_fd_ = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (socket_fd_ < 0) {
      throw std::runtime_error("failed to create UDP receiver socket");
    }

    sockaddr_in bind_address{};
    bind_address.sin_family = AF_INET;
    bind_address.sin_port = htons(0);
    if (inet_pton(AF_INET, "127.0.0.1", &bind_address.sin_addr) != 1) {
      throw std::runtime_error("failed to parse loopback address");
    }
    if (::bind(socket_fd_, reinterpret_cast<const sockaddr *>(&bind_address), sizeof(bind_address)) < 0) {
      throw std::runtime_error("failed to bind UDP receiver socket");
    }

    sockaddr_in actual_address{};
    socklen_t actual_length = sizeof(actual_address);
    if (::getsockname(socket_fd_, reinterpret_cast<sockaddr *>(&actual_address), &actual_length) < 0) {
      throw std::runtime_error("failed to read UDP receiver port");
    }
    port_ = ntohs(actual_address.sin_port);
  }

  ~UdpReceiver()
  {
    if (socket_fd_ >= 0) {
      ::close(socket_fd_);
    }
  }

  std::string endpointUri() const
  {
    return "udp://127.0.0.1:" + std::to_string(port_);
  }

  bool tryReceive(std::vector<uint8_t> & packet)
  {
    std::vector<uint8_t> buffer(1024);
    const auto received = ::recvfrom(
      socket_fd_,
      buffer.data(),
      buffer.size(),
      MSG_DONTWAIT,
      nullptr,
      nullptr);
    if (received <= 0) {
      return false;
    }
    buffer.resize(static_cast<size_t>(received));
    packet = buffer;
    return true;
  }

private:
  int socket_fd_{-1};
  uint16_t port_{0};
};

std::vector<rclcpp::Parameter> validParameters(const std::string & endpoint_uri)
{
  return {
    rclcpp::Parameter("backend.name", "network_pcm_sender"),
    rclcpp::Parameter("endpoint.uri", endpoint_uri),
    rclcpp::Parameter("transport.identity", "network_out_contract"),
    rclcpp::Parameter("input_topic", kInputTopic),
    rclcpp::Parameter("expected.sample_rate", 16000),
    rclcpp::Parameter("expected.channels", 1),
    rclcpp::Parameter("expected.encoding", "PCM16LE"),
    rclcpp::Parameter("expected.bit_depth", 16),
    rclcpp::Parameter("expected.layout", "interleaved"),
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

fa_interfaces::msg::AudioFrame validFrame()
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = rclcpp::Clock().now();
  frame.header.frame_id = "fixture_source";
  frame.source_id = "fixture_source";
  frame.stream_id = "fixture_stream";
  frame.encoding = "PCM16LE";
  frame.sample_rate = 16000;
  frame.channels = 1;
  frame.bit_depth = 16;
  frame.layout = "interleaved";
  frame.data = {0x10, 0x00, 0x20, 0x00};
  return frame;
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
    { auto node = std::make_shared<fa_network_out::FaNetworkOutNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenEndpointIsMissing)
{
  auto parameters = validParameters("udp://127.0.0.1:9");
  parameters[1] = rclcpp::Parameter("endpoint.uri", "");

  EXPECT_THROW(
    { auto node = std::make_shared<fa_network_out::FaNetworkOutNode>(optionsWith(parameters)); },
    std::runtime_error);
}

TEST_F(RclcppContractTest, FailsClosedWhenEndpointHostIsNotExplicitIpv4)
{
  EXPECT_THROW(
    {
      auto node = std::make_shared<fa_network_out::FaNetworkOutNode>(
        optionsWith(validParameters("udp://localhost:9000")));
    },
    std::runtime_error);
}

TEST_F(RclcppContractTest, SendsRawPcmPayloadWithoutFormatMutation)
{
  UdpReceiver receiver;
  auto node = std::make_shared<fa_network_out::FaNetworkOutNode>(
    optionsWith(validParameters(receiver.endpointUri())));
  auto io_node = std::make_shared<rclcpp::Node>("fa_network_out_contract_io");
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic,
    rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  publisher->publish(validFrame());

  std::vector<uint8_t> packet;
  ASSERT_TRUE(spinUntil(executor, [&receiver, &packet]() {
    return receiver.tryReceive(packet);
  }));
  EXPECT_EQ(packet, (std::vector<uint8_t>{0x10, 0x00, 0x20, 0x00}));
}

TEST_F(RclcppContractTest, FailsClosedWhenIncomingFrameFormatDoesNotMatch)
{
  UdpReceiver receiver;
  auto node = std::make_shared<fa_network_out::FaNetworkOutNode>(
    optionsWith(validParameters(receiver.endpointUri())));
  auto io_node = std::make_shared<rclcpp::Node>("fa_network_out_mismatch_contract_io");
  auto publisher = io_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    kInputTopic,
    rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);
  executor.add_node(io_node);
  ASSERT_TRUE(spinUntil(executor, [&publisher]() {
    return publisher->get_subscription_count() > 0;
  }));

  auto frame = validFrame();
  frame.layout = "planar";
  publisher->publish(frame);
  ASSERT_TRUE(spinUntil(executor, [&node]() {
    return node->hasFatalError();
  }));

  std::vector<uint8_t> packet;
  EXPECT_FALSE(receiver.tryReceive(packet));
}
}  // namespace
