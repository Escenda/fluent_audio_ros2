#include "fa_resample/fa_resample_node.hpp"
#include "fa_resample/backends/internal_linear_resampler.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <gtest/gtest.h>
#include <rclcpp/rclcpp.hpp>

namespace
{

using namespace std::chrono_literals;

using DiagnosticValues = std::map<std::string, std::string>;

constexpr double kPi = 3.14159265358979323846;

struct BackendGraphSmokeConfig
{
  std::string backend_name;
  std::optional<rclcpp::Parameter> backend_quality;
  std::string expected_quality;
};

rclcpp::NodeOptions quietGraphNodeOptions()
{
  rclcpp::NodeOptions options;
  options.enable_rosout(false);
  options.start_parameter_services(false);
  options.start_parameter_event_publisher(false);
  return options;
}

fa_interfaces::msg::AudioFrame makeFloat32FrameFromSamples(
  const rclcpp::Node & node,
  const std::vector<float> & samples)
{
  fa_interfaces::msg::AudioFrame frame;
  frame.header.stamp = node.now();
  frame.source_id = "test-mic";
  frame.stream_id = "audio/test/float32";
  frame.encoding = "FLOAT32LE";
  frame.sample_rate = 48000;
  frame.channels = 1;
  frame.bit_depth = 32;
  frame.layout = "interleaved";
  frame.data = fa_resample::backends::encodeFloat32Le(samples);
  frame.epoch = 11;
  return frame;
}

fa_interfaces::msg::AudioFrame makeFloat32Frame(const rclcpp::Node & node)
{
  std::vector<float> samples;
  samples.reserve(480);
  for (int i = 0; i < 480; ++i) {
    samples.push_back(static_cast<float>(i) / 480.0F);
  }

  return makeFloat32FrameFromSamples(node, samples);
}

fa_interfaces::msg::AudioFrame makeLowAmplitudeFloat32Frame(const rclcpp::Node & node)
{
  constexpr double kSampleRate = 48000.0;
  constexpr double kFundamentalHz = 440.0;
  constexpr double kSecondToneHz = 880.0;
  constexpr double kThirdToneHz = 1320.0;

  std::vector<float> samples;
  samples.reserve(480);
  for (int i = 0; i < 480; ++i) {
    const double t = static_cast<double>(i) / kSampleRate;
    const double sample =
      0.08 * std::sin(2.0 * kPi * kFundamentalHz * t) +
      0.03 * std::sin(2.0 * kPi * kSecondToneHz * t) +
      0.01 * std::sin(2.0 * kPi * kThirdToneHz * t);
    samples.push_back(static_cast<float>(sample));
  }

  return makeFloat32FrameFromSamples(node, samples);
}

std::vector<rclcpp::Parameter> validResampleParameters()
{
  return {
    rclcpp::Parameter("target_sample_rate", 16000),
    rclcpp::Parameter("backend.name", "internal_linear_resampler"),
    rclcpp::Parameter("input.encoding", "FLOAT32LE"),
    rclcpp::Parameter("input.bit_depth", 32),
    rclcpp::Parameter("input.layout", "interleaved"),
    rclcpp::Parameter("output.encoding", "FLOAT32LE"),
    rclcpp::Parameter("output.bit_depth", 32),
    rclcpp::Parameter("mic.enabled", true),
    rclcpp::Parameter("mic.input_topic", "/fa_resample_test/input"),
    rclcpp::Parameter("mic.output_topic", "/fa_resample_test/output"),
    rclcpp::Parameter("mic.input_stream_id", "audio/test/float32"),
    rclcpp::Parameter("mic.output.stream_id", "audio/test/mono16k"),
    rclcpp::Parameter("ref.enabled", false),
    rclcpp::Parameter("ref.input_topic", "/fa_resample_test/ref_in"),
    rclcpp::Parameter("ref.output_topic", "/fa_resample_test/ref_out"),
    rclcpp::Parameter("ref.input_stream_id", "audio/test/ref_float32"),
    rclcpp::Parameter("ref.output.stream_id", "audio/test/ref16k"),
    rclcpp::Parameter("qos.depth", 10),
    rclcpp::Parameter("qos.reliable", false),
    rclcpp::Parameter("diagnostics.publish_period_ms", 1000),
    rclcpp::Parameter("diagnostics.qos.depth", 10),
    rclcpp::Parameter("diagnostics.qos.reliable", true),
  };
}

rclcpp::NodeOptions graphNodeOptionsWith(std::vector<rclcpp::Parameter> parameters)
{
  rclcpp::NodeOptions options = quietGraphNodeOptions();
  options.parameter_overrides(parameters);
  return options;
}

void replaceParameter(
  std::vector<rclcpp::Parameter> & parameters,
  const rclcpp::Parameter & replacement)
{
  for (auto & parameter : parameters) {
    if (parameter.get_name() == replacement.get_name()) {
      parameter = replacement;
      return;
    }
  }
  parameters.push_back(replacement);
}

bool diagnosticsMatchBackendConfig(
  const DiagnosticValues & values,
  const BackendGraphSmokeConfig & config)
{
  const auto backend_name = values.find("backend.name");
  const auto backend_quality = values.find("backend.quality");
  if (backend_name == values.end() || backend_quality == values.end()) {
    return false;
  }

  return backend_name->second == config.backend_name &&
         backend_quality->second == config.expected_quality;
}

std::optional<DiagnosticValues> findFaResampleDiagnostics(
  const diagnostic_msgs::msg::DiagnosticArray & diagnostics,
  const BackendGraphSmokeConfig & config)
{
  for (const auto & status : diagnostics.status) {
    if (status.name != "fa_resample") {
      continue;
    }

    DiagnosticValues values;
    for (const auto & value : status.values) {
      values.emplace(value.key, value.value);
    }
    if (!diagnosticsMatchBackendConfig(values, config)) {
      continue;
    }
    return values;
  }

  return std::nullopt;
}

bool hasProcessedFrameMetrics(const std::optional<DiagnosticValues> & values)
{
  if (!values.has_value()) {
    return false;
  }

  const auto input_frames = values->find("input_frames_total");
  const auto output_frames = values->find("output_frames_total");
  const auto expected_output_frames = values->find("expected_output_frames");
  if (
    input_frames == values->end() ||
    output_frames == values->end() ||
    expected_output_frames == values->end()) {
    return false;
  }

  return std::stoull(input_frames->second) > 0ULL &&
         std::stoull(output_frames->second) > 0ULL &&
         std::stod(expected_output_frames->second) > 0.0;
}

std::string diagnosticValue(const DiagnosticValues & values, const std::string & key)
{
  const auto value = values.find(key);
  if (value == values.end()) {
    throw std::runtime_error("missing diagnostics key: " + key);
  }
  return value->second;
}

uint64_t diagnosticUnsignedValue(const DiagnosticValues & values, const std::string & key)
{
  return static_cast<uint64_t>(std::stoull(diagnosticValue(values, key)));
}

double diagnosticDoubleValue(const DiagnosticValues & values, const std::string & key)
{
  return std::stod(diagnosticValue(values, key));
}

void expectFiniteNormalizedFloat32Payload(
  const fa_interfaces::msg::AudioFrame & frame,
  const DiagnosticValues & diagnostics_values)
{
  ASSERT_EQ(frame.data.size() % sizeof(float), 0U);
  const std::vector<float> decoded = fa_resample::backends::decodeFloat32Le(frame.data);
  ASSERT_FALSE(decoded.empty());

  for (const float sample : decoded) {
    EXPECT_TRUE(std::isfinite(sample));
    EXPECT_GE(sample, -1.0F);
    EXPECT_LE(sample, 1.0F);
  }

  const uint64_t output_frames_total =
    diagnosticUnsignedValue(diagnostics_values, "output_frames_total");
  EXPECT_GE(output_frames_total, static_cast<uint64_t>(decoded.size() / frame.channels));
}

void runBackendGraphSmoke(const BackendGraphSmokeConfig & config)
{
  std::vector<rclcpp::Parameter> parameters = validResampleParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", config.backend_name));
  if (config.backend_quality.has_value()) {
    replaceParameter(parameters, config.backend_quality.value());
  }
  replaceParameter(parameters, rclcpp::Parameter("diagnostics.publish_period_ms", 50));

  auto resample_node = std::make_shared<fa_resample::FaResampleNode>(
    graphNodeOptionsWith(parameters));
  auto test_node = std::make_shared<rclcpp::Node>(
    "fa_resample_" + config.backend_name + "_graph_smoke_test",
    quietGraphNodeOptions());

  rclcpp::QoS audio_qos(10);
  audio_qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_resample_test/input",
    audio_qos);

  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_resample_test/output",
    audio_qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::QoS diagnostics_qos(10);
  diagnostics_qos.reliable();
  std::optional<DiagnosticValues> diagnostics_values;
  auto diagnostics_subscriber =
    test_node->create_subscription<diagnostic_msgs::msg::DiagnosticArray>(
      "/diagnostics",
      diagnostics_qos,
      [&diagnostics_values, &config](const diagnostic_msgs::msg::DiagnosticArray::SharedPtr msg) {
        const std::optional<DiagnosticValues> values = findFaResampleDiagnostics(*msg, config);
        if (values.has_value()) {
          diagnostics_values = values;
        }
      });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(resample_node);
  executor.add_node(test_node);

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (
    (!received.has_value() || !hasProcessedFrameMetrics(diagnostics_values)) &&
    std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeLowAmplitudeFloat32Frame(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(resample_node);
  diagnostics_subscriber.reset();
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value()) << config.backend_name;
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "audio/test/mono16k");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 11U);
  EXPECT_FALSE(received->data.empty());

  ASSERT_TRUE(hasProcessedFrameMetrics(diagnostics_values)) << config.backend_name;
  const DiagnosticValues values = diagnostics_values.value();

  EXPECT_EQ(diagnosticValue(values, "backend.name"), config.backend_name);
  EXPECT_EQ(diagnosticValue(values, "backend.quality"), config.expected_quality);
  EXPECT_EQ(diagnosticValue(values, "target_sample_rate"), "16000");
  EXPECT_EQ(diagnosticValue(values, "output.encoding"), "FLOAT32LE");
  EXPECT_EQ(diagnosticValue(values, "output.bit_depth"), "32");

  expectFiniteNormalizedFloat32Payload(received.value(), values);
  EXPECT_GE(diagnosticDoubleValue(values, "algorithmic_delay_ms"), 0.0);
  EXPECT_GE(diagnosticDoubleValue(values, "processing_time_mean_ms"), 0.0);
  EXPECT_GT(diagnosticUnsignedValue(values, "input_frames_total"), 0ULL);
  EXPECT_GT(diagnosticUnsignedValue(values, "output_frames_total"), 0ULL);
  EXPECT_GT(diagnosticDoubleValue(values, "expected_output_frames"), 0.0);
  EXPECT_NO_THROW((void)std::stoll(diagnosticValue(values, "frame_count_error_samples")));
}

class RclcppFixture : public ::testing::Test
{
protected:
  static void SetUpTestSuite()
  {
    if (!rclcpp::ok()) {
      int argc = 0;
      char ** argv = nullptr;
      rclcpp::init(argc, argv);
    }
  }

  static void TearDownTestSuite()
  {
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
  }
};

}  // namespace

TEST_F(RclcppFixture, ExplicitInternalLinearBackendStarts)
{
  auto resample_node = std::make_shared<fa_resample::FaResampleNode>(
    graphNodeOptionsWith(validResampleParameters()));

  EXPECT_EQ(resample_node->get_name(), std::string("fa_resample"));
}

TEST_F(RclcppFixture, UnknownBackendNameFailsStartup)
{
  std::vector<rclcpp::Parameter> parameters = validResampleParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", "unknown_backend"));

  EXPECT_THROW(
    {
      auto resample_node = std::make_shared<fa_resample::FaResampleNode>(
        graphNodeOptionsWith(parameters));
      (void)resample_node;
    },
    std::runtime_error);
}

TEST_F(RclcppFixture, SpeexDspQualityOutsideZeroToTenFailsStartup)
{
  std::vector<rclcpp::Parameter> parameters = validResampleParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", "speexdsp"));
  replaceParameter(parameters, rclcpp::Parameter("backend.quality", 11));

  EXPECT_THROW(
    {
      auto resample_node = std::make_shared<fa_resample::FaResampleNode>(
        graphNodeOptionsWith(parameters));
      (void)resample_node;
    },
    std::runtime_error);
}

TEST_F(RclcppFixture, SoxrQualityOutsideAllowedSetFailsStartup)
{
  std::vector<rclcpp::Parameter> parameters = validResampleParameters();
  replaceParameter(parameters, rclcpp::Parameter("backend.name", "soxr"));
  replaceParameter(parameters, rclcpp::Parameter("backend.quality", "BEST"));

  EXPECT_THROW(
    {
      auto resample_node = std::make_shared<fa_resample::FaResampleNode>(
        graphNodeOptionsWith(parameters));
      (void)resample_node;
    },
    std::runtime_error);
}

TEST_F(RclcppFixture, PublishesResampledFloat32Frame)
{
  rclcpp::NodeOptions options = graphNodeOptionsWith(validResampleParameters());

  auto resample_node = std::make_shared<fa_resample::FaResampleNode>(options);
  auto test_node = std::make_shared<rclcpp::Node>("fa_resample_graph_test", quietGraphNodeOptions());

  rclcpp::QoS qos(10);
  qos.best_effort();
  auto publisher = test_node->create_publisher<fa_interfaces::msg::AudioFrame>(
    "/fa_resample_test/input",
    qos);
  std::optional<fa_interfaces::msg::AudioFrame> received;
  auto subscriber = test_node->create_subscription<fa_interfaces::msg::AudioFrame>(
    "/fa_resample_test/output",
    qos,
    [&received](const fa_interfaces::msg::AudioFrame::SharedPtr msg) {
      received = *msg;
    });

  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(resample_node);
  executor.add_node(test_node);

  auto wrong_stream = makeFloat32Frame(*test_node);
  wrong_stream.stream_id = "audio/test/other";
  for (int i = 0; i < 4; ++i) {
    publisher->publish(wrong_stream);
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }
  EXPECT_FALSE(received.has_value());

  const auto deadline = std::chrono::steady_clock::now() + 3s;
  while (!received.has_value() && std::chrono::steady_clock::now() < deadline) {
    publisher->publish(makeFloat32Frame(*test_node));
    executor.spin_some(20ms);
    std::this_thread::sleep_for(10ms);
  }

  executor.remove_node(test_node);
  executor.remove_node(resample_node);
  subscriber.reset();
  publisher.reset();

  ASSERT_TRUE(received.has_value());
  EXPECT_EQ(received->source_id, "test-mic");
  EXPECT_EQ(received->stream_id, "audio/test/mono16k");
  EXPECT_EQ(received->encoding, "FLOAT32LE");
  EXPECT_EQ(received->sample_rate, 16000U);
  EXPECT_EQ(received->channels, 1U);
  EXPECT_EQ(received->bit_depth, 32U);
  EXPECT_EQ(received->layout, "interleaved");
  EXPECT_EQ(received->epoch, 11U);
  EXPECT_EQ(received->data.size(), 160U * sizeof(float));
}

TEST_F(RclcppFixture, PublishesDiagnosticsWithBackendAndMetricsValues)
{
  runBackendGraphSmoke(
    BackendGraphSmokeConfig{
      "internal_linear_resampler",
      std::nullopt,
      "debug_reference"});
}

TEST_F(RclcppFixture, SoxrBackendProcessesGraphAndPublishesDiagnostics)
{
  runBackendGraphSmoke(
    BackendGraphSmokeConfig{
      "soxr",
      rclcpp::Parameter("backend.quality", "MQ"),
      "MQ"});
}

TEST_F(RclcppFixture, SpeexDspBackendProcessesGraphAndPublishesDiagnostics)
{
  runBackendGraphSmoke(
    BackendGraphSmokeConfig{
      "speexdsp",
      rclcpp::Parameter("backend.quality", 6),
      "6"});
}
