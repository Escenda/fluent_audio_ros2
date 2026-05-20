#include "fa_resample/backends/internal_linear_resampler.hpp"
#include "fa_resample/backends/backend_factory.hpp"
#include "fa_resample/backends/soxr_resampler.hpp"
#include "fa_resample/backends/speexdsp_resampler.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

std::vector<uint8_t> float32LeBytes(const std::vector<float> & samples)
{
  std::vector<uint8_t> bytes;
  bytes.reserve(samples.size() * sizeof(float));
  for (const float sample : samples) {
    fa_resample::backends::appendFloat32Le(sample, bytes);
  }
  return bytes;
}

fa_resample::backends::FrameContract frameContract(
  uint32_t sample_rate,
  uint32_t channels,
  const std::vector<uint8_t> & bytes)
{
  return fa_resample::backends::FrameContract{
    "FLOAT32LE",
    sample_rate,
    channels,
    32,
    "interleaved",
    bytes.size()};
}

fa_resample::backends::ProcessRequest processRequest(
  const std::string & stream_id,
  const std::vector<uint8_t> & input,
  const fa_resample::backends::FrameContract & contract)
{
  return fa_resample::backends::ProcessRequest{stream_id, input, contract};
}

std::vector<float> simpleFrameSamples()
{
  std::vector<float> samples;
  samples.reserve(480);
  for (int i = 0; i < 480; ++i) {
    samples.push_back(static_cast<float>(i) / 960.0F);
  }
  return samples;
}

std::unique_ptr<fa_resample::backends::ResamplerBackend> createSelectedBackend(
  const fa_resample::backends::BackendSelection & selection)
{
  std::unique_ptr<fa_resample::backends::ResamplerBackend> backend =
    fa_resample::backends::createResamplerBackend(selection);
  if (backend == nullptr) {
    throw std::runtime_error("resampler backend factory returned null backend");
  }
  return backend;
}

void expectSelectedBackendDoesNotFallback(
  const fa_resample::backends::BackendSelection & selection,
  const std::string & expected_name)
{
  const std::unique_ptr<fa_resample::backends::ResamplerBackend> backend =
    createSelectedBackend(selection);
  EXPECT_EQ(backend->name(), expected_name);
  EXPECT_NE(backend->name(), fa_resample::backends::InternalLinearResamplerBackend::kName);
}

void expectBackendProcessesSimpleFloat32Frame(
  fa_resample::backends::ResamplerBackend & backend,
  const std::string & stream_id)
{
  const std::vector<uint8_t> input = float32LeBytes(simpleFrameSamples());
  const fa_resample::backends::FrameContract contract = frameContract(48000, 1, input);
  std::vector<uint8_t> output;
  fa_resample::backends::ProcessResult last_result;

  for (int i = 0; i < 20 && output.empty(); ++i) {
    last_result = backend.process(processRequest(stream_id, input, contract), output);
    ASSERT_EQ(last_result.status, fa_resample::backends::ProcessStatus::kOk);
  }

  ASSERT_FALSE(output.empty());
  EXPECT_EQ(output.size() % sizeof(float), 0U);
  const std::vector<float> decoded = fa_resample::backends::decodeFloat32Le(output);
  EXPECT_FALSE(decoded.empty());
  EXPECT_TRUE(fa_resample::backends::containsOnlyFiniteNormalizedSamples(decoded));
  EXPECT_GT(backend.metrics().process_call_count, 0U);
}

}  // namespace

TEST(ResamplerBackendSelectionContract, ExplicitInternalLinearSelectionSucceeds)
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kInternalLinearResampler;
  selection.name = fa_resample::backends::InternalLinearResamplerBackend::kName;
  selection.target_sample_rate = 16000;

  const std::unique_ptr<fa_resample::backends::ResamplerBackend> backend =
    fa_resample::backends::createResamplerBackend(selection);

  ASSERT_NE(backend, nullptr);
  EXPECT_EQ(backend->name(), fa_resample::backends::InternalLinearResamplerBackend::kName);
  EXPECT_EQ(backend->quality(), "debug_reference");
  EXPECT_EQ(backend->targetSampleRate(), 16000);
}

TEST(ResamplerBackendSelectionContract, UnknownBackendNameFailsValidation)
{
  EXPECT_THROW(
    fa_resample::backends::parseBackendKind("unknown_backend"),
    std::runtime_error);
}

TEST(ResamplerBackendSelectionContract, SpeexDspQualityOutsideZeroToTenIsRejected)
{
  EXPECT_THROW(fa_resample::backends::validateSpeexDspQuality(-1), std::runtime_error);
  EXPECT_THROW(fa_resample::backends::validateSpeexDspQuality(11), std::runtime_error);
  EXPECT_EQ(fa_resample::backends::validateSpeexDspQuality(0), 0);
  EXPECT_EQ(fa_resample::backends::validateSpeexDspQuality(10), 10);
}

TEST(ResamplerBackendSelectionContract, SoxrQualityOutsideAllowedSetIsRejected)
{
  EXPECT_THROW(fa_resample::backends::parseSoxrQuality(""), std::runtime_error);
  EXPECT_THROW(fa_resample::backends::parseSoxrQuality("mq"), std::runtime_error);
  EXPECT_THROW(fa_resample::backends::parseSoxrQuality("BEST"), std::runtime_error);
  EXPECT_EQ(
    fa_resample::backends::soxrQualityName(
      fa_resample::backends::parseSoxrQuality("QQ")),
    "QQ");
  EXPECT_EQ(
    fa_resample::backends::soxrQualityName(
      fa_resample::backends::parseSoxrQuality("LQ")),
    "LQ");
  EXPECT_EQ(
    fa_resample::backends::soxrQualityName(
      fa_resample::backends::parseSoxrQuality("MQ")),
    "MQ");
  EXPECT_EQ(
    fa_resample::backends::soxrQualityName(
      fa_resample::backends::parseSoxrQuality("HQ")),
    "HQ");
  EXPECT_EQ(
    fa_resample::backends::soxrQualityName(
      fa_resample::backends::parseSoxrQuality("VHQ")),
    "VHQ");
}

TEST(ResamplerBackendSelectionContract, SpeexDspSelectionDoesNotFallbackToInternalLinear)
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kSpeexDsp;
  selection.name = fa_resample::backends::SpeexDspResamplerBackend::kName;
  selection.target_sample_rate = 16000;
  selection.speex_quality = 6;
  selection.quality_label = "6";

  try {
    expectSelectedBackendDoesNotFallback(
      selection,
      fa_resample::backends::SpeexDspResamplerBackend::kName);
  } catch (const std::runtime_error & error) {
    GTEST_SKIP() << "speexdsp optional runtime unavailable: " << error.what();
  }
}

TEST(ResamplerBackendSelectionContract, SoxrSelectionDoesNotFallbackToInternalLinear)
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kSoxr;
  selection.name = fa_resample::backends::SoxrResamplerBackend::kName;
  selection.target_sample_rate = 16000;
  selection.soxr_quality = fa_resample::backends::SoxrQuality::kMq;
  selection.quality_label = "MQ";

  expectSelectedBackendDoesNotFallback(
    selection,
    fa_resample::backends::SoxrResamplerBackend::kName);
}

TEST(InternalLinearResamplerBackendContract, RejectsInvalidConfig)
{
  EXPECT_THROW(
    fa_resample::backends::InternalLinearResamplerBackend(
      fa_resample::backends::InternalLinearResamplerConfig{0}),
    std::runtime_error);
}

TEST(InternalLinearResamplerBackendContract, AcceptsOnlyFloat32LeInterleavedFrames)
{
  const fa_resample::backends::FrameContract valid{
    "FLOAT32LE",
    48000,
    1,
    32,
    "interleaved",
    sizeof(float) * 160};

  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(valid),
    fa_resample::backends::FrameContractStatus::kOk);

  auto pcm16 = valid;
  pcm16.encoding = "PCM16LE";
  pcm16.bit_depth = 16;
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(pcm16),
    fa_resample::backends::FrameContractStatus::kUnsupportedEncoding);

  auto pcm32 = valid;
  pcm32.encoding = "PCM32LE";
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(pcm32),
    fa_resample::backends::FrameContractStatus::kUnsupportedEncoding);

  auto planar = valid;
  planar.layout = "planar";
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(planar),
    fa_resample::backends::FrameContractStatus::kUnsupportedLayout);

  auto unaligned = valid;
  unaligned.data_size = sizeof(float) * 2 + 1;
  EXPECT_EQ(
    fa_resample::backends::validateFloat32InterleavedContract(unaligned),
    fa_resample::backends::FrameContractStatus::kUnalignedData);
}

TEST(InternalLinearResamplerBackendContract, DecodesAndEncodesFloat32LeWithoutPcmConversion)
{
  const std::vector<float> samples{-1.0F, -0.5F, 0.0F, 0.5F, 1.0F};
  const std::vector<uint8_t> bytes = float32LeBytes(samples);

  const std::vector<float> decoded = fa_resample::backends::decodeFloat32Le(bytes);
  ASSERT_EQ(decoded.size(), samples.size());
  for (size_t i = 0; i < samples.size(); ++i) {
    EXPECT_FLOAT_EQ(decoded.at(i), samples.at(i));
  }

  const std::vector<uint8_t> encoded = fa_resample::backends::encodeFloat32Le(decoded);
  EXPECT_EQ(encoded, bytes);
}

TEST(InternalLinearResamplerBackendContract, RejectsNonFiniteOrOutOfRangeSamplesInsteadOfClamping)
{
  EXPECT_FALSE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({0.0F, 1.25F}));
  EXPECT_FALSE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({0.0F, -1.25F}));
  EXPECT_FALSE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({0.0F, NAN}));
  EXPECT_TRUE(fa_resample::backends::containsOnlyFiniteNormalizedSamples({-1.0F, 0.0F, 1.0F}));

  EXPECT_TRUE(fa_resample::backends::encodeFloat32Le({0.0F, 1.25F}).empty());

  uint32_t out_frames = 0;
  EXPECT_TRUE(
    fa_resample::backends::resampleLinear({0.0F, 1.25F}, 48000, 16000, 1, 2, out_frames)
    .empty());
}

TEST(InternalLinearResamplerBackendContract, ResamplesLinearFloat32WithoutChangingChannels)
{
  std::vector<float> input;
  input.reserve(480);
  for (int i = 0; i < 480; ++i) {
    input.push_back(static_cast<float>(i) / 480.0F);
  }

  uint32_t out_frames = 0;
  const std::vector<float> output =
    fa_resample::backends::resampleLinear(input, 48000, 16000, 1, 480, out_frames);

  EXPECT_EQ(out_frames, 160U);
  ASSERT_EQ(output.size(), 160U);
  EXPECT_FLOAT_EQ(output.front(), input.front());
  EXPECT_GT(output.back(), 0.0F);
  EXPECT_LT(output.back(), 1.0F);
}

TEST(InternalLinearResamplerBackendContract, ResamplesToConfiguredHigherSampleRate)
{
  const std::vector<float> input{0.0F, 0.25F, 0.5F, 0.75F, 1.0F};

  uint32_t out_frames = 0;
  const std::vector<float> output =
    fa_resample::backends::resampleLinear(input, 16000, 48000, 1, 5, out_frames);

  EXPECT_EQ(out_frames, 15U);
  ASSERT_EQ(output.size(), 15U);
  EXPECT_FLOAT_EQ(output.front(), input.front());
  EXPECT_LE(output.back(), 1.0F);
}

TEST(InternalLinearResamplerBackendContract, ReturnsTypedStatusAndLeavesOutputOnFailure)
{
  fa_resample::backends::InternalLinearResamplerBackend backend(
    fa_resample::backends::InternalLinearResamplerConfig{16000});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  const std::vector<uint8_t> invalid_samples = float32LeBytes({1.25F});
  const std::string stream_id = "audio/test/internal_linear_invalid_samples";
  const fa_resample::backends::ProcessResult result = backend.process(
    processRequest(stream_id, invalid_samples, frameContract(48000, 1, invalid_samples)),
    output);

  EXPECT_EQ(result.status, fa_resample::backends::ProcessStatus::kInvalidInputSamples);
  EXPECT_EQ(output, float32LeBytes({0.125F}));
}

TEST(InternalLinearResamplerBackendContract, RejectsUnsupportedFrameWithoutOutputMutation)
{
  fa_resample::backends::InternalLinearResamplerBackend backend(
    fa_resample::backends::InternalLinearResamplerConfig{16000});

  std::vector<uint8_t> output = float32LeBytes({0.125F});
  const std::vector<uint8_t> input = float32LeBytes({0.0F, 0.25F, 0.5F});
  fa_resample::backends::FrameContract contract = frameContract(48000, 1, input);
  contract.encoding = "PCM16LE";
  contract.bit_depth = 16;
  const std::string stream_id = "audio/test/internal_linear_unsupported";

  const fa_resample::backends::ProcessResult result =
    backend.process(processRequest(stream_id, input, contract), output);

  EXPECT_EQ(result.status, fa_resample::backends::ProcessStatus::kInvalidFrameContract);
  EXPECT_EQ(
    result.frame_contract_status,
    fa_resample::backends::FrameContractStatus::kUnsupportedEncoding);
  EXPECT_EQ(output, float32LeBytes({0.125F}));
}

TEST(InternalLinearResamplerBackendContract, RejectsUnhandledStatusValues)
{
  EXPECT_THROW(
    fa_resample::backends::frameContractStatusName(
      static_cast<fa_resample::backends::FrameContractStatus>(999)),
    std::logic_error);
  EXPECT_THROW(
    fa_resample::backends::processStatusMessage(
      static_cast<fa_resample::backends::ProcessStatus>(999)),
    std::logic_error);
}

TEST(InternalLinearResamplerBackendContract, ProcessesBytesThroughBackend)
{
  fa_resample::backends::InternalLinearResamplerBackend backend(
    fa_resample::backends::InternalLinearResamplerConfig{16000});

  std::vector<float> samples;
  samples.reserve(480);
  for (int i = 0; i < 480; ++i) {
    samples.push_back(static_cast<float>(i) / 480.0F);
  }
  const std::vector<uint8_t> input = float32LeBytes(samples);

  std::vector<uint8_t> output;
  const std::string stream_id = "audio/test/internal_linear_process";
  const fa_resample::backends::ProcessResult result =
    backend.process(processRequest(stream_id, input, frameContract(48000, 1, input)), output);

  EXPECT_EQ(result.status, fa_resample::backends::ProcessStatus::kOk);
  EXPECT_EQ(result.output_frames, 160U);
  EXPECT_EQ(output.size(), 160U * sizeof(float));
}

TEST(InternalLinearResamplerBackendContract, TracksFrameCountDriftAndProcessingMetrics)
{
  fa_resample::backends::InternalLinearResamplerBackend backend(
    fa_resample::backends::InternalLinearResamplerConfig{16000});

  const std::vector<uint8_t> input = float32LeBytes(simpleFrameSamples());
  const fa_resample::backends::FrameContract contract = frameContract(48000, 1, input);
  const std::string stream_id = "audio/test/internal_linear_metrics";
  uint64_t output_frames_total = 0;

  for (int i = 0; i < 10; ++i) {
    std::vector<uint8_t> output;
    const fa_resample::backends::ProcessResult result =
      backend.process(processRequest(stream_id, input, contract), output);
    ASSERT_EQ(result.status, fa_resample::backends::ProcessStatus::kOk);
    EXPECT_EQ(result.output_frames, 160U);
    output_frames_total += result.output_frames;
  }

  const fa_resample::backends::BackendMetrics metrics = backend.metrics();
  EXPECT_EQ(metrics.process_call_count, 10U);
  EXPECT_EQ(metrics.input_frames_total, 4800U);
  EXPECT_EQ(metrics.output_frames_total, output_frames_total);
  EXPECT_DOUBLE_EQ(metrics.expected_output_frames, 1600.0);
  EXPECT_EQ(metrics.frame_count_error_samples, 0);
  EXPECT_GT(metrics.processing_time_total_ns, 0U);
  EXPECT_GT(fa_resample::backends::processingTimeMeanMs(metrics), 0.0);
}

TEST(SpeexDspResamplerBackendContract, ProcessesSimpleFloat32FrameWhenRuntimeLibraryIsAvailable)
{
  std::unique_ptr<fa_resample::backends::ResamplerBackend> backend;
  try {
    fa_resample::backends::BackendSelection selection;
    selection.kind = fa_resample::backends::BackendKind::kSpeexDsp;
    selection.name = fa_resample::backends::SpeexDspResamplerBackend::kName;
    selection.target_sample_rate = 16000;
    selection.speex_quality = 6;
    selection.quality_label = "6";
    backend = fa_resample::backends::createResamplerBackend(selection);
  } catch (const std::runtime_error & error) {
    GTEST_SKIP() << error.what();
  }

  ASSERT_NE(backend, nullptr);
  ASSERT_EQ(backend->name(), fa_resample::backends::SpeexDspResamplerBackend::kName);
  expectBackendProcessesSimpleFloat32Frame(*backend, "audio/test/speexdsp_smoke");
}

TEST(SoxrResamplerBackendContract, ProcessesSimpleFloat32FrameWhenRuntimeLibraryIsAvailable)
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kSoxr;
  selection.name = fa_resample::backends::SoxrResamplerBackend::kName;
  selection.target_sample_rate = 16000;
  selection.soxr_quality = fa_resample::backends::SoxrQuality::kMq;
  selection.quality_label = "MQ";
  std::unique_ptr<fa_resample::backends::ResamplerBackend> backend =
    createSelectedBackend(selection);

  ASSERT_NE(backend, nullptr);
  ASSERT_EQ(backend->name(), fa_resample::backends::SoxrResamplerBackend::kName);
  expectBackendProcessesSimpleFloat32Frame(*backend, "audio/test/soxr_smoke");
}
