#include <chrono>
#include <cmath>
#include <memory>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "fv_audio/msg/audio_frame.hpp"

namespace fv_audio
{

class SineWaveGenerator : public rclcpp::Node
{
public:
  SineWaveGenerator()
  : rclcpp::Node("fv_sine_wave_generator"),
    phase_(0.0)
  {
    // パラメータの宣言と取得
    this->declare_parameter("frequency", 440.0);  // A4音
    this->declare_parameter("amplitude", 0.5);    // 最大振幅の50%
    this->declare_parameter("sample_rate", 48000);
    this->declare_parameter("channels", 1);
    this->declare_parameter("bit_depth", 16);
    this->declare_parameter("chunk_ms", 20);

    frequency_ = this->get_parameter("frequency").as_double();
    amplitude_ = this->get_parameter("amplitude").as_double();
    sample_rate_ = this->get_parameter("sample_rate").as_int();
    channels_ = this->get_parameter("channels").as_int();
    bit_depth_ = this->get_parameter("bit_depth").as_int();
    chunk_ms_ = this->get_parameter("chunk_ms").as_int();

    // サンプル数を計算
    samples_per_chunk_ = (sample_rate_ * chunk_ms_) / 1000;

    RCLCPP_INFO(this->get_logger(),
                "Sine wave generator initialized: freq=%.1fHz, amp=%.2f, rate=%dHz, channels=%d, bits=%d",
                frequency_, amplitude_, sample_rate_, channels_, bit_depth_);

    // パブリッシャーの作成
    audio_pub_ = this->create_publisher<fv_audio::msg::AudioFrame>(
      "audio/frame", rclcpp::SensorDataQoS());

    // タイマーの作成（chunk_msごとに発火）
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(chunk_ms_),
      std::bind(&SineWaveGenerator::generateAndPublish, this));
  }

private:
  void generateAndPublish()
  {
    fv_audio::msg::AudioFrame frame_msg;
    frame_msg.header.stamp = this->now();
    frame_msg.encoding = (bit_depth_ == 16) ? "pcm16" : "float32";
    frame_msg.sample_rate = sample_rate_;
    frame_msg.channels = channels_;
    frame_msg.bit_depth = bit_depth_;

    // サイン波データの生成
    if (bit_depth_ == 16) {
      generateSineWaveInt16(frame_msg);
    } else {
      generateSineWaveFloat32(frame_msg);
    }

    audio_pub_->publish(frame_msg);
  }

  void generateSineWaveInt16(fv_audio::msg::AudioFrame &frame_msg)
  {
    constexpr double kInt16Scale = 32767.0;
    const size_t total_samples = samples_per_chunk_ * channels_;
    const size_t byte_count = total_samples * sizeof(int16_t);

    frame_msg.data.resize(byte_count);
    int16_t *samples = reinterpret_cast<int16_t*>(frame_msg.data.data());

    double peak = 0.0;
    double rms_accum = 0.0;

    for (size_t i = 0; i < samples_per_chunk_; ++i) {
      // サイン波の値を計算
      double value = amplitude_ * std::sin(2.0 * M_PI * phase_);

      // 位相を更新
      phase_ += frequency_ / static_cast<double>(sample_rate_);
      if (phase_ >= 1.0) {
        phase_ -= 1.0;
      }

      // int16に変換
      int16_t sample = static_cast<int16_t>(value * kInt16Scale);

      // 全チャンネルに同じ値を設定
      for (uint32_t ch = 0; ch < channels_; ++ch) {
        samples[i * channels_ + ch] = sample;
      }

      // RMSとピークの計算
      double normalized = value;
      peak = std::max(peak, std::abs(normalized));
      rms_accum += normalized * normalized;
    }

    frame_msg.rms = static_cast<float>(std::sqrt(rms_accum / samples_per_chunk_));
    frame_msg.peak = static_cast<float>(peak);
    frame_msg.vad = false;
  }

  void generateSineWaveFloat32(fv_audio::msg::AudioFrame &frame_msg)
  {
    const size_t total_samples = samples_per_chunk_ * channels_;
    const size_t byte_count = total_samples * sizeof(float);

    frame_msg.data.resize(byte_count);
    float *samples = reinterpret_cast<float*>(frame_msg.data.data());

    double peak = 0.0;
    double rms_accum = 0.0;

    for (size_t i = 0; i < samples_per_chunk_; ++i) {
      // サイン波の値を計算
      double value = amplitude_ * std::sin(2.0 * M_PI * phase_);

      // 位相を更新
      phase_ += frequency_ / static_cast<double>(sample_rate_);
      if (phase_ >= 1.0) {
        phase_ -= 1.0;
      }

      // 全チャンネルに同じ値を設定
      for (uint32_t ch = 0; ch < channels_; ++ch) {
        samples[i * channels_ + ch] = static_cast<float>(value);
      }

      // RMSとピークの計算
      peak = std::max(peak, std::abs(value));
      rms_accum += value * value;
    }

    frame_msg.rms = static_cast<float>(std::sqrt(rms_accum / samples_per_chunk_));
    frame_msg.peak = static_cast<float>(peak);
    frame_msg.vad = false;
  }

  rclcpp::Publisher<fv_audio::msg::AudioFrame>::SharedPtr audio_pub_;
  rclcpp::TimerBase::SharedPtr timer_;

  double frequency_;
  double amplitude_;
  int sample_rate_;
  int channels_;
  int bit_depth_;
  int chunk_ms_;
  size_t samples_per_chunk_;
  double phase_;
};

}  // namespace fv_audio

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fv_audio::SineWaveGenerator>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
