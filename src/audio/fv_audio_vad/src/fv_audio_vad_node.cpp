#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "fv_audio/msg/audio_frame.hpp"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

namespace
{
constexpr double kInt16Scale = 32768.0;
}

class FvAudioVadNode : public rclcpp::Node
{
public:
  FvAudioVadNode()
  : rclcpp::Node("fv_audio_vad_node")
  {
    threshold_ = this->declare_parameter("vad.threshold", 0.05);
    release_ms_ = this->declare_parameter("vad.release_ms", 200);
    min_active_ms_ = this->declare_parameter("vad.min_active_ms", 100);

    vad_pub_ = this->create_publisher<std_msgs::msg::Bool>("audio/vad", rclcpp::SystemDefaultsQoS());

    auto qos = rclcpp::SensorDataQoS().keep_last(10);
    audio_sub_ = this->create_subscription<fv_audio::msg::AudioFrame>(
      "audio/frame",
      qos,
      std::bind(&FvAudioVadNode::audioCallback, this, std::placeholders::_1));

    last_active_time_ = std::chrono::steady_clock::now();
    state_change_time_ = last_active_time_;

    RCLCPP_INFO(this->get_logger(), "VAD threshold=%.3f release=%dms min_active=%dms",
      threshold_, release_ms_, min_active_ms_);
  }

private:
  void audioCallback(const fv_audio::msg::AudioFrame::SharedPtr msg)
  {
    if (!msg) {
      return;
    }

    double rms = msg->rms;
    if (rms <= 0.0) {
      rms = computeRms(msg->data, msg->bit_depth, msg->channels);
    }

    auto now = std::chrono::steady_clock::now();
    bool above = rms >= threshold_;
    bool publish = false;

    if (above) {
      last_active_time_ = now;
      if (!current_state_) {
        current_state_ = true;
        state_change_time_ = now;
        publish = true;
      }
    } else if (current_state_) {
      auto release_duration = std::chrono::milliseconds(release_ms_);
      auto min_active_duration = std::chrono::milliseconds(min_active_ms_);
      if ((now - last_active_time_) > release_duration &&
          (now - state_change_time_) > min_active_duration)
      {
        current_state_ = false;
        state_change_time_ = now;
        publish = true;
      }
    }

    if (publish) {
      std_msgs::msg::Bool msg_out;
      msg_out.data = current_state_;
      vad_pub_->publish(msg_out);
    }
  }

  static double computeRms(const std::vector<uint8_t> &buffer, uint32_t bit_depth, uint32_t channels)
  {
    if (buffer.empty() || channels == 0) {
      return 0.0;
    }

    if (bit_depth == 16) {
      const int16_t *samples = reinterpret_cast<const int16_t *>(buffer.data());
      size_t sample_count = buffer.size() / sizeof(int16_t);
      double accum = 0.0;
      for (size_t i = 0; i < sample_count; ++i) {
        double normalized = samples[i] / kInt16Scale;
        accum += normalized * normalized;
      }
      return std::sqrt(accum / static_cast<double>(sample_count));
    }

    const float *samples = reinterpret_cast<const float *>(buffer.data());
    size_t sample_count = buffer.size() / sizeof(float);
    double accum = 0.0;
    for (size_t i = 0; i < sample_count; ++i) {
      accum += samples[i] * samples[i];
    }
    return std::sqrt(accum / static_cast<double>(sample_count));
  }

  double threshold_;
  int release_ms_;
  int min_active_ms_;
  bool current_state_{false};
  std::chrono::steady_clock::time_point last_active_time_;
  std::chrono::steady_clock::time_point state_change_time_;

  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr vad_pub_;
  rclcpp::Subscription<fv_audio::msg::AudioFrame>::SharedPtr audio_sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FvAudioVadNode>());
  rclcpp::shutdown();
  return 0;
}
