#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "fa_interfaces/msg/audio_frame.hpp"
#include "fa_interfaces/msg/vad_state.hpp"
#include "fa_interfaces/msg/wake_word_result.hpp"
#include "rclcpp/rclcpp.hpp"

#include "fa_kws/audio_utils.hpp"
#include "fa_kws/backends/kws_backend.hpp"
#include "fa_kws/backends/sherpa_onnx_kws_backend.hpp"
#include "fa_kws/vad_gate.hpp"

namespace fa_kws
{

using fa_interfaces::msg::AudioFrame;
using fa_interfaces::msg::VadState;
using fa_interfaces::msg::WakeWordResult;

namespace
{

bool traceEnabled()
{
  static const bool enabled = []() {
    const char *v = std::getenv("FA_KWS_TRACE");
    return v != nullptr && v[0] != '\0' && std::atoi(v) != 0;
  }();
  return enabled;
}

void tracef(const char *fmt, ...)
{
  if (!traceEnabled()) {
    return;
  }
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fflush(stderr);
}

}  // namespace

class FaKwsNode : public rclcpp::Node
{
public:
  FaKwsNode()
  : rclcpp::Node("fa_kws"),
    current_vad_prob_(0.0f)
  {
    loadParameters();
    validateBackendOrThrow();
    validateVadInputOrThrow();
    validateExecutionProviderOrThrow();
    validateModelFilesOrThrow();

    SherpaOnnxKwsBackendConfig cfg;
    cfg.target_sample_rate = target_sample_rate_;
    cfg.model_num_threads = model_num_threads_;
    cfg.execution_provider = execution_provider_;
    cfg.encoder_path = encoder_path_;
    cfg.decoder_path = decoder_path_;
    cfg.joiner_path = joiner_path_;
    cfg.tokens_path = tokens_path_;
    cfg.keywords_path = keywords_path_;
    cfg.max_active_paths = kws_max_active_paths_;
    cfg.num_trailing_blanks = kws_num_trailing_blanks_;
    cfg.keywords_score = static_cast<float>(kws_keywords_score_);
    cfg.keywords_threshold = static_cast<float>(kws_keywords_threshold_);
    cfg.vad_threshold = static_cast<float>(probability_gate_);
    cfg.cooldown = std::chrono::milliseconds(cooldown_ms_);

    kws_backend_ = std::make_unique<SherpaOnnxKwsBackend>(cfg);
    setupCommunication();
    setupDebug();

    RCLCPP_INFO(
      this->get_logger(),
      "fa_kws initialized: audio_topic=%s vad_topic=%s output_topic=%s target_sr=%d provider=%s",
      audio_topic_.c_str(),
      vad_topic_.c_str(),
      output_topic_.c_str(),
      target_sample_rate_,
      execution_provider_.c_str());
  }

  ~FaKwsNode() override
  {
    if (dump_audio_enable_ && !capture_buffer_.empty()) {
      try {
        const std::uint32_t sr =
          capture_sample_rate_ > 0 ? static_cast<std::uint32_t>(capture_sample_rate_)
                                   : static_cast<std::uint32_t>(target_sample_rate_);
        writeWav(dump_audio_path_, capture_buffer_, sr);
        RCLCPP_INFO(
          this->get_logger(),
          "Dumped %zu samples to %s",
          capture_buffer_.size(),
          dump_audio_path_.c_str());
      } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Failed to dump wav: %s", e.what());
      }
    }
  }

private:
  void validateBackendOrThrow() const
  {
    if (backend_name_.empty()) {
      throw std::runtime_error("backend.name is required");
    }
    if (backend_name_ != "sherpa_onnx_kws") {
      throw std::runtime_error("unsupported fa_kws backend.name: " + backend_name_);
    }
  }

  void validateModelFilesOrThrow() const
  {
    auto check_file = [](const char *label, const std::string &path, std::ostringstream &oss) {
      if (path.empty()) {
        oss << "  - " << label << ": (empty)\n";
        return;
      }
      std::error_code ec;
      const bool ok = std::filesystem::exists(path, ec) && !ec;
      if (!ok) {
        oss << "  - " << label << ": not found (" << path << ")\n";
      }
    };

    std::ostringstream errors;
    check_file("model.encoder", encoder_path_, errors);
    check_file("model.decoder", decoder_path_, errors);
    check_file("model.joiner", joiner_path_, errors);
    check_file("model.tokens", tokens_path_, errors);
    check_file("kws.keywords_file", keywords_path_, errors);

    const std::string missing = errors.str();
    if (!missing.empty()) {
      std::ostringstream oss;
      oss << "fa_kws model/keywords files are missing:\n" << missing
          << "Hint: check fa_kws/config/default.yaml and ensure models are installed under ros2_ws/models.";
      throw std::runtime_error(oss.str());
    }
  }

  void validateExecutionProviderOrThrow() const
  {
    if (execution_provider_.empty()) {
      throw std::runtime_error("backend.execution_provider is required");
    }
    if (!isSupportedSherpaOnnxExecutionProvider(execution_provider_)) {
      throw std::runtime_error(
        "unsupported fa_kws backend.execution_provider: " + execution_provider_ +
        "; supported providers: " + supportedSherpaOnnxExecutionProvidersForMessage());
    }
  }

  void validateVadInputOrThrow() const
  {
    if (!isValidVadGateThreshold(probability_gate_)) {
      throw std::runtime_error("vad.probability_gate must be finite and in (0.0, 1.0]");
    }
    if (vad_max_age_ms_ <= 0) {
      throw std::runtime_error("vad.max_age_ms must be greater than zero");
    }
  }

  static void writeWav(const std::string &path,
                       const std::vector<float> &samples,
                       std::uint32_t sample_rate)
  {
    std::vector<std::int16_t> pcm(samples.size());
    for (std::size_t i = 0; i < samples.size(); ++i) {
      float v = samples[i];
      if (v > 1.0f) v = 1.0f;
      if (v < -1.0f) v = -1.0f;
      pcm[i] = static_cast<std::int16_t>(v * 32767.0f);
    }
    const std::uint16_t channels = 1;
    const std::uint32_t byte_rate = sample_rate * channels * 2;
    const std::uint16_t block_align = channels * 2;
    const std::uint32_t data_size = static_cast<std::uint32_t>(pcm.size() * 2);
    const std::uint32_t riff_size = 36 + data_size;

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("failed to open wav for writing");
    }
    ofs.write("RIFF", 4);
    ofs.write(reinterpret_cast<const char *>(&riff_size), 4);
    ofs.write("WAVE", 4);
    ofs.write("fmt ", 4);
    std::uint32_t fmt_chunk_size = 16;
    std::uint16_t audio_format = 1;
    std::uint16_t bits_per_sample = 16;
    ofs.write(reinterpret_cast<const char *>(&fmt_chunk_size), 4);
    ofs.write(reinterpret_cast<const char *>(&audio_format), 2);
    ofs.write(reinterpret_cast<const char *>(&channels), 2);
    ofs.write(reinterpret_cast<const char *>(&sample_rate), 4);
    ofs.write(reinterpret_cast<const char *>(&byte_rate), 4);
    ofs.write(reinterpret_cast<const char *>(&block_align), 2);
    ofs.write(reinterpret_cast<const char *>(&bits_per_sample), 2);
    ofs.write("data", 4);
    ofs.write(reinterpret_cast<const char *>(&data_size), 4);
    ofs.write(reinterpret_cast<const char *>(pcm.data()),
              static_cast<std::streamsize>(pcm.size() * sizeof(std::int16_t)));
  }

  void loadParameters()
  {
    audio_topic_ = this->declare_parameter<std::string>("audio_topic", "audio/frame");
    vad_topic_ = this->declare_parameter<std::string>("vad_topic", "voice/vad_state");
    output_topic_ = this->declare_parameter<std::string>("output_topic", "voice/wake_word");
    backend_name_ = this->declare_parameter<std::string>("backend.name", "");

    target_sample_rate_ = this->declare_parameter<int>("target_sample_rate", 16000);
    probability_gate_ = this->declare_parameter<double>("vad.probability_gate", 0.35);
    vad_max_age_ms_ = this->declare_parameter<int>("vad.max_age_ms", 1000);
    cooldown_ms_ = this->declare_parameter<int>("cooldown_ms", 2000);
    dump_audio_enable_ = this->declare_parameter<bool>("dump_audio.enable", false);
    dump_audio_path_ = this->declare_parameter<std::string>("dump_audio.path", "/tmp/fa_kws_capture.wav");
    debug_status_period_sec_ = this->declare_parameter<double>("debug.status_period_sec", 0.0);

    encoder_path_ = this->declare_parameter<std::string>("model.encoder", "");
    decoder_path_ = this->declare_parameter<std::string>("model.decoder", "");
    joiner_path_ = this->declare_parameter<std::string>("model.joiner", "");
    tokens_path_ = this->declare_parameter<std::string>("model.tokens", "");
    keywords_path_ = this->declare_parameter<std::string>("kws.keywords_file", "");

    model_num_threads_ = this->declare_parameter<int>("model.num_threads", 4);
    execution_provider_ = this->declare_parameter<std::string>(
      "backend.execution_provider", "");

    kws_max_active_paths_ = this->declare_parameter<int>("kws.max_active_paths", 4);
    kws_num_trailing_blanks_ = this->declare_parameter<int>("kws.num_trailing_blanks", 1);
    kws_keywords_score_ = this->declare_parameter<double>("kws.keywords_score", 1.0);
    kws_keywords_threshold_ = this->declare_parameter<double>("kws.keywords_threshold", 0.25);
  }

  void setupCommunication()
  {
    rclcpp::QoS qos_audio(rclcpp::KeepLast(10));
    qos_audio.best_effort();

    rclcpp::QoS qos_vad(rclcpp::KeepLast(20));
    qos_vad.best_effort();

    wake_pub_ = this->create_publisher<WakeWordResult>(output_topic_, qos_audio);

    vad_sub_ = this->create_subscription<VadState>(
      vad_topic_, qos_vad,
      std::bind(&FaKwsNode::onVad, this, std::placeholders::_1));

    audio_sub_ = this->create_subscription<AudioFrame>(
      audio_topic_, qos_audio,
      std::bind(&FaKwsNode::onAudio, this, std::placeholders::_1));
  }

  void setupDebug()
  {
    if (debug_status_period_sec_ <= 0.0) {
      return;
    }
    debug_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(debug_status_period_sec_),
      std::bind(&FaKwsNode::onDebugTimer, this));
  }

  void onDebugTimer()
  {
    const auto frames = audio_frames_received_.load(std::memory_order_relaxed);
    const auto samples = audio_samples_received_.load(std::memory_order_relaxed);
    const auto last_rx_ns = last_audio_rx_ns_.load(std::memory_order_relaxed);

    double since_audio_sec = -1.0;
    if (last_rx_ns > 0) {
      const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
      since_audio_sec = static_cast<double>(now_ns - last_rx_ns) / 1e9;
    }

    const float vad_prob = current_vad_prob_.load(std::memory_order_relaxed);
    const int32_t sr = last_sample_rate_.load(std::memory_order_relaxed);

    RCLCPP_INFO(
      this->get_logger(),
      "fa_kws status: frames=%llu samples=%llu last_audio=%.2fs vad_prob=%.3f sr=%d",
      static_cast<unsigned long long>(frames),
      static_cast<unsigned long long>(samples),
      since_audio_sec,
      static_cast<double>(vad_prob),
      static_cast<int>(sr));
  }

  void onVad(const VadState::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    if (!isValidVadProbability(msg->probability)) {
      current_vad_prob_.store(0.0f, std::memory_order_relaxed);
      last_vad_rx_ns_.store(0, std::memory_order_relaxed);
      RCLCPP_ERROR_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Rejecting invalid VadState.probability %.6f; probability must be finite in [0.0, 1.0]",
        static_cast<double>(msg->probability));
      return;
    }
    current_vad_prob_.store(msg->probability, std::memory_order_relaxed);
    const auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::steady_clock::now().time_since_epoch())
                          .count();
    last_vad_rx_ns_.store(now_ns, std::memory_order_relaxed);
  }

  bool readFreshVadProbability(std::int64_t now_ns, float &vad_prob)
  {
    const auto last_vad_ns = last_vad_rx_ns_.load(std::memory_order_relaxed);
    if (last_vad_ns <= 0) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Dropping AudioFrame until first VadState is received on %s",
        vad_topic_.c_str());
      return false;
    }

    const auto vad_age_ms = (now_ns - last_vad_ns) / 1000000;
    if (vad_age_ms > vad_max_age_ms_) {
      RCLCPP_WARN_THROTTLE(
        this->get_logger(),
        *this->get_clock(),
        2000,
        "Dropping AudioFrame because latest VadState is stale: age_ms=%lld max_age_ms=%d",
        static_cast<long long>(vad_age_ms),
        vad_max_age_ms_);
      return false;
    }

    vad_prob = current_vad_prob_.load(std::memory_order_relaxed);
    return true;
  }

  void onAudio(const AudioFrame::SharedPtr msg)
  {
    if (!msg || !kws_backend_) {
      return;
    }

    tracef(
      "fa_kws: onAudio stamp=%d.%09u source=%s stream=%s sr=%u ch=%u bit=%u layout=%s bytes=%zu\n",
      static_cast<int>(msg->header.stamp.sec),
      static_cast<unsigned int>(msg->header.stamp.nanosec),
      msg->source_id.c_str(),
      msg->stream_id.c_str(),
      msg->sample_rate,
      msg->channels,
      msg->bit_depth,
      msg->layout.c_str(),
      msg->data.size());

    audio_frames_received_.fetch_add(1, std::memory_order_relaxed);
    last_sample_rate_.store(static_cast<int32_t>(msg->sample_rate), std::memory_order_relaxed);
    const auto now = std::chrono::steady_clock::now();
    const auto now_rx_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             now.time_since_epoch())
                             .count();
    last_audio_rx_ns_.store(now_rx_ns, std::memory_order_relaxed);

    std::vector<float> samples;
    try {
      samples = frameToCanonicalFloat(*msg);
    } catch (const std::invalid_argument &e) {
      RCLCPP_ERROR(this->get_logger(), "Dropping invalid AudioFrame: %s", e.what());
      return;
    }
    if (samples.empty()) {
      return;
    }

    const std::int32_t src_rate = static_cast<std::int32_t>(msg->sample_rate);
    if (src_rate <= 0) {
      RCLCPP_ERROR(this->get_logger(), "Dropping AudioFrame with invalid sample_rate=%d", src_rate);
      return;
    }

    if (src_rate != target_sample_rate_) {
      RCLCPP_ERROR(
        this->get_logger(),
        "Dropping AudioFrame with sample_rate=%d; expected target_sample_rate=%d",
        static_cast<int>(src_rate),
        target_sample_rate_);
      return;
    }

    if (dump_audio_enable_) {
      if (capture_sample_rate_ != 0 && capture_sample_rate_ != src_rate) {
        RCLCPP_WARN(
          this->get_logger(),
          "Audio sample rate changed from %d to %d; resetting capture buffer.",
          static_cast<int>(capture_sample_rate_),
          static_cast<int>(src_rate));
        capture_buffer_.clear();
      }
      capture_sample_rate_ = src_rate;
      capture_buffer_.insert(capture_buffer_.end(), samples.begin(), samples.end());
    }

    audio_samples_received_.fetch_add(samples.size(), std::memory_order_relaxed);

    float vad_prob = 0.0f;
    if (!readFreshVadProbability(now_rx_ns, vad_prob)) {
      return;
    }
    if (!passesVadGate(vad_prob, static_cast<float>(probability_gate_))) {
      RCLCPP_DEBUG(
        this->get_logger(),
        "Dropping AudioFrame below VAD probability gate: vad_prob=%.6f threshold=%.6f",
        static_cast<double>(vad_prob),
        probability_gate_);
      return;
    }

    auto detection = kws_backend_->process(samples, target_sample_rate_, vad_prob, now);
    if (!detection) {
      return;
    }

    WakeWordResult out;
    out.header = msg->header;
    out.keyword = detection->keyword;
    out.score = detection->score;
    out.detected = true;

    wake_pub_->publish(out);

    RCLCPP_INFO(
      this->get_logger(),
      "Wake word detected: keyword=%s vad_prob=%.3f",
      out.keyword.c_str(),
      static_cast<double>(vad_prob));
  }

  std::string audio_topic_;
  std::string vad_topic_;
  std::string output_topic_;
  std::string backend_name_;

  int target_sample_rate_{16000};
  double probability_gate_{0.35};
  int vad_max_age_ms_{1000};
  int cooldown_ms_{2000};

  std::string encoder_path_;
  std::string decoder_path_;
  std::string joiner_path_;
  std::string tokens_path_;
  std::string keywords_path_;

  int model_num_threads_{4};
  std::string execution_provider_{};

  int kws_max_active_paths_{4};
  int kws_num_trailing_blanks_{1};
  double kws_keywords_score_{1.0};
  double kws_keywords_threshold_{0.25};

  std::atomic<float> current_vad_prob_;
  std::atomic<std::int64_t> last_vad_rx_ns_{0};

  std::unique_ptr<KwsBackend> kws_backend_;

  double debug_status_period_sec_{0.0};
  rclcpp::TimerBase::SharedPtr debug_timer_;
  std::atomic<std::uint64_t> audio_frames_received_{0};
  std::atomic<std::uint64_t> audio_samples_received_{0};
  std::atomic<std::int64_t> last_audio_rx_ns_{0};
  std::atomic<std::int32_t> last_sample_rate_{0};

  bool dump_audio_enable_{false};
  std::string dump_audio_path_;
  std::int32_t capture_sample_rate_{0};
  std::vector<float> capture_buffer_;

  rclcpp::Subscription<AudioFrame>::SharedPtr audio_sub_;
  rclcpp::Subscription<VadState>::SharedPtr vad_sub_;
  rclcpp::Publisher<WakeWordResult>::SharedPtr wake_pub_;
};

}  // namespace fa_kws

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_kws::FaKwsNode>();
    rclcpp::spin(node);
  } catch (const std::exception &e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_kws"), "Exception in fa_kws: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
  rclcpp::shutdown();
  return EXIT_SUCCESS;
}
