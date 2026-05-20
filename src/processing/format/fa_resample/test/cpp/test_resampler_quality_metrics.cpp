#include "fa_resample/backends/backend_factory.hpp"
#include "fa_resample/backends/internal_linear_resampler.hpp"
#include "fa_resample/backends/resampler_backend.hpp"
#include "fa_resample/backends/soxr_resampler.hpp"
#include "fa_resample/backends/speexdsp_resampler.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace
{

constexpr uint32_t kInputSampleRate = 48000;
constexpr uint32_t kTargetSampleRate = 16000;
constexpr uint32_t kChannels = 1;
constexpr size_t kChunkFrames = 480;
constexpr size_t kMetricGuardSamples = 96;
constexpr size_t kMetricMaxSamples = 12000;
constexpr double kPi = 3.141592653589793238462643383279502884;

struct Tone
{
  double frequency_hz{0.0};
  double amplitude{0.0};
};

struct QualityMetrics
{
  double rms_error{0.0};
  double peak_error{0.0};
  double snr_db{0.0};
  size_t compared_samples{0};
};

struct BackendOutput
{
  std::string label;
  std::vector<float> samples;
  size_t impulse_peak_offset_samples{0};
  fa_resample::backends::BackendMetrics backend_metrics;
};

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
  const uint32_t sample_rate,
  const uint32_t channels,
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

std::vector<float> generateMultiTone(
  const uint32_t sample_rate,
  const double duration_seconds,
  const std::vector<Tone> & tones)
{
  const size_t frame_count = static_cast<size_t>(
    std::llround(static_cast<double>(sample_rate) * duration_seconds));
  const size_t fade_frames = static_cast<size_t>(sample_rate / 100U);
  std::vector<float> samples;
  samples.reserve(frame_count);

  for (size_t frame = 0; frame < frame_count; ++frame) {
    const double t = static_cast<double>(frame) / static_cast<double>(sample_rate);
    double value = 0.0;
    for (const Tone & tone : tones) {
      value += tone.amplitude * std::sin(2.0 * kPi * tone.frequency_hz * t);
    }

    double envelope = 1.0;
    if (fade_frames > 0U && frame < fade_frames) {
      envelope = static_cast<double>(frame) / static_cast<double>(fade_frames);
    } else if (fade_frames > 0U && frame_count - frame <= fade_frames) {
      envelope = static_cast<double>(frame_count - frame) / static_cast<double>(fade_frames);
    }
    samples.push_back(static_cast<float>(value * envelope));
  }
  return samples;
}

std::vector<float> generateImpulse()
{
  const size_t frame_count = static_cast<size_t>(kInputSampleRate / 2U);
  constexpr size_t kImpulseInputOffset = 960;
  std::vector<float> samples(frame_count, 0.0F);
  samples.at(kImpulseInputOffset) = 0.75F;
  return samples;
}

fa_resample::backends::ProcessRequest processRequest(
  const std::string & stream_id,
  const std::vector<uint8_t> & input,
  const fa_resample::backends::FrameContract & contract)
{
  return fa_resample::backends::ProcessRequest{stream_id, input, contract};
}

std::vector<float> processSignal(
  fa_resample::backends::ResamplerBackend & backend,
  const std::vector<float> & samples,
  const std::string & stream_id)
{
  std::vector<float> output_samples;
  for (size_t frame_offset = 0; frame_offset < samples.size(); frame_offset += kChunkFrames) {
    const size_t frame_count = std::min(kChunkFrames, samples.size() - frame_offset);
    const auto begin = samples.begin() + static_cast<std::vector<float>::difference_type>(frame_offset);
    const auto end = begin + static_cast<std::vector<float>::difference_type>(frame_count);
    const std::vector<float> chunk(begin, end);
    const std::vector<uint8_t> input = float32LeBytes(chunk);
    std::vector<uint8_t> output;
    const fa_resample::backends::ProcessResult result = backend.process(
      processRequest(stream_id, input, frameContract(kInputSampleRate, kChannels, input)),
      output);
    if (result.status != fa_resample::backends::ProcessStatus::kOk) {
      throw std::runtime_error(
        "resampler process failed for " + backend.name() + "/" + backend.quality() +
        ": " + fa_resample::backends::processStatusMessage(result.status));
    }
    if (!output.empty()) {
      const std::vector<float> decoded = fa_resample::backends::decodeFloat32Le(output);
      output_samples.insert(output_samples.end(), decoded.begin(), decoded.end());
    }
  }
  return output_samples;
}

size_t impulsePeakOffsetSamples(const std::vector<float> & samples)
{
  if (samples.empty()) {
    throw std::runtime_error("impulse output is empty");
  }

  size_t peak_index = 0;
  float peak_value = 0.0F;
  for (size_t i = 0; i < samples.size(); ++i) {
    const float value = std::fabs(samples.at(i));
    if (value > peak_value) {
      peak_value = value;
      peak_index = i;
    }
  }
  if (peak_value <= 0.0F) {
    throw std::runtime_error("impulse output has no non-zero peak");
  }
  return peak_index;
}

std::unique_ptr<fa_resample::backends::ResamplerBackend> createBackend(
  const fa_resample::backends::BackendSelection & selection)
{
  std::unique_ptr<fa_resample::backends::ResamplerBackend> backend =
    fa_resample::backends::createResamplerBackend(selection);
  if (backend == nullptr) {
    throw std::runtime_error("resampler backend factory returned null backend");
  }
  return backend;
}

fa_resample::backends::BackendSelection internalLinearSelection()
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kInternalLinearResampler;
  selection.name = fa_resample::backends::InternalLinearResamplerBackend::kName;
  selection.target_sample_rate = static_cast<int>(kTargetSampleRate);
  selection.quality_label = fa_resample::backends::InternalLinearResamplerBackend::kQuality;
  return selection;
}

fa_resample::backends::BackendSelection soxrSelection(
  const fa_resample::backends::SoxrQuality quality)
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kSoxr;
  selection.name = fa_resample::backends::SoxrResamplerBackend::kName;
  selection.target_sample_rate = static_cast<int>(kTargetSampleRate);
  selection.soxr_quality = quality;
  selection.quality_label = fa_resample::backends::soxrQualityName(quality);
  return selection;
}

fa_resample::backends::BackendSelection speexSelection(const int quality)
{
  fa_resample::backends::BackendSelection selection;
  selection.kind = fa_resample::backends::BackendKind::kSpeexDsp;
  selection.name = fa_resample::backends::SpeexDspResamplerBackend::kName;
  selection.target_sample_rate = static_cast<int>(kTargetSampleRate);
  selection.speex_quality = quality;
  selection.quality_label = std::to_string(quality);
  return selection;
}

BackendOutput backendOutput(
  const fa_resample::backends::BackendSelection & selection,
  const std::vector<float> & signal,
  const std::string & stream_suffix)
{
  std::unique_ptr<fa_resample::backends::ResamplerBackend> signal_backend = createBackend(selection);
  const std::string label = signal_backend->name() + "/" + signal_backend->quality();
  const std::vector<float> signal_output = processSignal(
    *signal_backend,
    signal,
    "audio/test/quality/" + stream_suffix + "/signal/" + label);
  const fa_resample::backends::BackendMetrics metrics = signal_backend->metrics();

  std::unique_ptr<fa_resample::backends::ResamplerBackend> impulse_backend = createBackend(selection);
  const std::vector<float> impulse_output = processSignal(
    *impulse_backend,
    generateImpulse(),
    "audio/test/quality/" + stream_suffix + "/impulse/" + label);

  return BackendOutput{label, signal_output, impulsePeakOffsetSamples(impulse_output), metrics};
}

QualityMetrics compareAligned(
  const BackendOutput & reference,
  const BackendOutput & candidate)
{
  const size_t reference_start = reference.impulse_peak_offset_samples + kMetricGuardSamples;
  const size_t candidate_start = candidate.impulse_peak_offset_samples + kMetricGuardSamples;
  if (reference.samples.size() <= reference_start || candidate.samples.size() <= candidate_start) {
    throw std::runtime_error("not enough output samples to compare " + candidate.label);
  }

  const size_t shared_length = std::min(
    kMetricMaxSamples,
    std::min(reference.samples.size() - reference_start, candidate.samples.size() - candidate_start));
  if (shared_length < 1024U) {
    throw std::runtime_error("aligned comparison window is too short for " + candidate.label);
  }

  double signal_energy = 0.0;
  double error_energy = 0.0;
  double peak_error = 0.0;
  for (size_t i = 0; i < shared_length; ++i) {
    const double ref = static_cast<double>(reference.samples.at(reference_start + i));
    const double candidate_value = static_cast<double>(candidate.samples.at(candidate_start + i));
    const double error = candidate_value - ref;
    signal_energy += ref * ref;
    error_energy += error * error;
    peak_error = std::max(peak_error, std::fabs(error));
  }

  const double rms_error = std::sqrt(error_energy / static_cast<double>(shared_length));
  const double signal_power = signal_energy / static_cast<double>(shared_length);
  const double error_power = error_energy / static_cast<double>(shared_length);
  const double snr_db = 10.0 * std::log10(signal_power / error_power);
  return QualityMetrics{rms_error, peak_error, snr_db, shared_length};
}

void expectFiniteMetrics(const QualityMetrics & metrics, const std::string & label)
{
  EXPECT_GT(metrics.compared_samples, 1024U) << label;
  EXPECT_TRUE(std::isfinite(metrics.rms_error)) << label;
  EXPECT_TRUE(std::isfinite(metrics.peak_error)) << label;
  EXPECT_TRUE(std::isfinite(metrics.snr_db)) << label;
  EXPECT_GE(metrics.rms_error, 0.0) << label;
  EXPECT_GE(metrics.peak_error, 0.0) << label;
}

std::string metricValueString(const double value)
{
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(15) << value;
  return stream.str();
}

void recordQualityMetrics(const std::string & prefix, const QualityMetrics & metrics)
{
  testing::Test::RecordProperty(prefix + "_rms_error", metricValueString(metrics.rms_error));
  testing::Test::RecordProperty(prefix + "_peak_error", metricValueString(metrics.peak_error));
  testing::Test::RecordProperty(prefix + "_snr_db", metricValueString(metrics.snr_db));
  testing::Test::RecordProperty(
    prefix + "_compared_samples",
    static_cast<int>(metrics.compared_samples));
}

}  // namespace

TEST(ResamplerQualityMetrics, ComputesPassbandMetricsAgainstRequiredSoxrVhqReference)
{
  const std::vector<float> signal = generateMultiTone(
    kInputSampleRate,
    1.0,
    {
      Tone{320.0, 0.18},
      Tone{1180.0, 0.16},
      Tone{2700.0, 0.14},
      Tone{6100.0, 0.10},
    });

  const BackendOutput reference = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kVhq),
    signal,
    "passband");
  const BackendOutput internal = backendOutput(internalLinearSelection(), signal, "passband");
  const BackendOutput soxr_mq = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kMq),
    signal,
    "passband");
  const BackendOutput soxr_hq = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kHq),
    signal,
    "passband");

  const QualityMetrics internal_metrics = compareAligned(reference, internal);
  const QualityMetrics mq_metrics = compareAligned(reference, soxr_mq);
  const QualityMetrics hq_metrics = compareAligned(reference, soxr_hq);

  expectFiniteMetrics(internal_metrics, internal.label);
  expectFiniteMetrics(mq_metrics, soxr_mq.label);
  expectFiniteMetrics(hq_metrics, soxr_hq.label);
  EXPECT_GT(internal_metrics.snr_db, 15.0);
  EXPECT_GT(mq_metrics.snr_db, 35.0);
  EXPECT_GT(hq_metrics.snr_db, 35.0);
  EXPECT_LT(mq_metrics.peak_error, 0.04);
  EXPECT_LT(hq_metrics.peak_error, 0.04);

  recordQualityMetrics("passband_internal_linear", internal_metrics);
  recordQualityMetrics("passband_soxr_mq", mq_metrics);
  recordQualityMetrics("passband_soxr_hq", hq_metrics);
  testing::Test::RecordProperty(
    "reference_impulse_peak_offset_samples",
    static_cast<int>(reference.impulse_peak_offset_samples));
}

TEST(ResamplerQualityMetrics, SoxrIsCloserToVhqThanInternalLinearForOutOfBandAliasProbe)
{
  const std::vector<float> signal = generateMultiTone(
    kInputSampleRate,
    1.0,
    {
      Tone{1000.0, 0.20},
      Tone{11000.0, 0.16},
    });

  const BackendOutput reference = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kVhq),
    signal,
    "alias_probe");
  const BackendOutput internal = backendOutput(internalLinearSelection(), signal, "alias_probe");
  const BackendOutput soxr_mq = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kMq),
    signal,
    "alias_probe");
  const BackendOutput soxr_hq = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kHq),
    signal,
    "alias_probe");

  const QualityMetrics internal_metrics = compareAligned(reference, internal);
  const QualityMetrics mq_metrics = compareAligned(reference, soxr_mq);
  const QualityMetrics hq_metrics = compareAligned(reference, soxr_hq);

  expectFiniteMetrics(internal_metrics, internal.label);
  expectFiniteMetrics(mq_metrics, soxr_mq.label);
  expectFiniteMetrics(hq_metrics, soxr_hq.label);
  EXPECT_LT(mq_metrics.rms_error, internal_metrics.rms_error * 0.70);
  EXPECT_LT(hq_metrics.rms_error, internal_metrics.rms_error * 0.70);
  EXPECT_GT(mq_metrics.snr_db, internal_metrics.snr_db + 3.0);
  EXPECT_GT(hq_metrics.snr_db, internal_metrics.snr_db + 3.0);

  recordQualityMetrics("alias_internal_linear", internal_metrics);
  recordQualityMetrics("alias_soxr_mq", mq_metrics);
  recordQualityMetrics("alias_soxr_hq", hq_metrics);
}

TEST(SpeexDspResamplerQualityMetrics, ComputesMetricsAgainstSoxrVhqWhenRuntimeLibraryIsAvailable)
{
  std::unique_ptr<fa_resample::backends::ResamplerBackend> probe;
  try {
    probe = createBackend(speexSelection(6));
  } catch (const std::runtime_error & error) {
    GTEST_SKIP() << "speexdsp optional runtime unavailable: " << error.what();
  }
  ASSERT_NE(probe, nullptr);

  const std::vector<float> signal = generateMultiTone(
    kInputSampleRate,
    1.0,
    {
      Tone{400.0, 0.18},
      Tone{1450.0, 0.14},
      Tone{5200.0, 0.10},
    });

  const BackendOutput reference = backendOutput(
    soxrSelection(fa_resample::backends::SoxrQuality::kVhq),
    signal,
    "speex_optional");
  const BackendOutput speex = backendOutput(speexSelection(6), signal, "speex_optional");
  const QualityMetrics speex_metrics = compareAligned(reference, speex);

  expectFiniteMetrics(speex_metrics, speex.label);
  EXPECT_GT(speex_metrics.snr_db, 15.0);
  EXPECT_LT(speex_metrics.peak_error, 0.12);
  recordQualityMetrics("speex_q6_passband", speex_metrics);
}
