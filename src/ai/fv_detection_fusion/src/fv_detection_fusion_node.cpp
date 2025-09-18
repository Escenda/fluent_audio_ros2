#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <fv_msgs/msg/detection_array.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace fv_detection_fusion {

class DetectionFusionNode : public rclcpp::Node {
public:
  explicit DetectionFusionNode(const rclcpp::NodeOptions &options)
      : rclcpp::Node("fv_detection_fusion_node", options) {
    declareParameters();
    readParameters();

    output_pub_ = this->create_publisher<fv_msgs::msg::DetectionArray>(output_topic_, rclcpp::QoS(output_qos_depth_));

    for (const auto &source : sources_) {
      auto sub = this->create_subscription<vision_msgs::msg::Detection2DArray>(
          source.topic, rclcpp::QoS(source.qos_depth),
          [this, cfg = source](const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
            handleDetections(cfg, msg);
          });
      source_subs_.push_back(sub);
    }

    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(diagnostic_interval_ms_)),
        std::bind(&DetectionFusionNode::publishFusion, this));

    RCLCPP_INFO(get_logger(), "fv_detection_fusion_node ready (sources=%zu)", sources_.size());
  }

private:
  struct SourceConfig {
    std::string label;
    std::string topic;
    float confidence_multiplier{1.0f};
    float min_confidence{0.0f};
    int qos_depth{5};
  };

  struct DetRecord {
    fv_msgs::msg::Detection2D detection;
    rclcpp::Time last_update{0};
    int hold_frames{0};
  };

  template <typename T>
  void declareIfMissing(const std::string &name, const T &default_value) {
    if (!this->has_parameter(name)) {
      this->declare_parameter<T>(name, default_value);
    }
  }

  void declareParameters() {
    declareIfMissing<std::string>("output_topic", "/fv/d405/detection_fusion/rois");
    declareIfMissing<int>("output_qos_depth", 5);
    declareIfMissing<double>("diagnostic_interval_ms", 100.0);
    declareIfMissing<int>("hold_frames", 5);
    declareIfMissing<std::string>("frame_id", "");
  }

  void readParameters() {
    output_topic_ = this->get_parameter("output_topic").as_string();
    output_qos_depth_ = this->get_parameter("output_qos_depth").as_int();
    diagnostic_interval_ms_ = this->get_parameter("diagnostic_interval_ms").as_double();
    hold_frames_ = this->get_parameter("hold_frames").as_int();
    frame_id_override_ = this->get_parameter("frame_id").as_string();

    auto param_list = this->list_parameters({"sources"}, 10);
    std::unordered_map<std::string, SourceConfig> configs;
    for (const auto &name : param_list.names) {
      if (name.rfind("sources.", 0) != 0) {
        continue;
      }
      const std::string without_prefix = name.substr(std::string("sources.").size());
      const auto dot = without_prefix.find('.');
      if (dot == std::string::npos) {
        continue;
      }
      const std::string key = without_prefix.substr(0, dot);
      const std::string param = without_prefix.substr(dot + 1);
      SourceConfig &cfg = configs[key];
      if (param == "label") {
        cfg.label = this->get_parameter(name).as_string();
      } else if (param == "topic") {
        cfg.topic = this->get_parameter(name).as_string();
      } else if (param == "confidence_multiplier") {
        cfg.confidence_multiplier = static_cast<float>(this->get_parameter(name).as_double());
      } else if (param == "min_confidence") {
        cfg.min_confidence = static_cast<float>(this->get_parameter(name).as_double());
      } else if (param == "qos_depth") {
        cfg.qos_depth = this->get_parameter(name).as_int();
      }
    }

    for (auto &entry : configs) {
      auto &cfg = entry.second;
      if (cfg.topic.empty()) {
        RCLCPP_WARN(get_logger(), "sources.%s.topic missing, skipping", entry.first.c_str());
        continue;
      }
      if (cfg.label.empty()) {
        cfg.label = entry.first;
      }
      sources_.push_back(cfg);
    }

    if (sources_.empty()) {
      RCLCPP_WARN(get_logger(), "No detection sources configured; fusion will output empty arrays");
    }
  }

  void handleDetections(const SourceConfig &cfg, const vision_msgs::msg::Detection2DArray::SharedPtr msg) {
    const rclcpp::Time stamp(msg->header.stamp);
    std::lock_guard<std::mutex> lock(data_mutex_);

    for (const auto &det : msg->detections) {
      const int32_t key = makeKey(cfg.label, det);
      auto [it, inserted] = active_detections_.try_emplace(key);
      auto &record = it->second;
      record.detection.header = msg->header;
      record.detection.id = key;
      record.detection.label = cfg.label;
      record.detection.class_id = 0;

      float score = 1.0f;
      if (!det.results.empty()) {
        score = static_cast<float>(det.results.front().hypothesis.score);
      }
      score *= cfg.confidence_multiplier;
      if (score < cfg.min_confidence) {
        continue;
      }
      record.detection.conf_object = score;
      record.detection.conf_fused = score;

      record.detection.bbox_min.x = det.bbox.center.position.x - det.bbox.size_x * 0.5;
      record.detection.bbox_min.y = det.bbox.center.position.y - det.bbox.size_y * 0.5;
      record.detection.bbox_max.x = det.bbox.center.position.x + det.bbox.size_x * 0.5;
      record.detection.bbox_max.y = det.bbox.center.position.y + det.bbox.size_y * 0.5;
      record.detection.mask_instance_id = 0;
      record.detection.mask_semantic_id = 0;
      record.detection.depth_hint_m = std::numeric_limits<float>::quiet_NaN();
      record.detection.observed_at = msg->header.stamp;

      record.last_update = stamp;
      record.hold_frames = hold_frames_;
    }
  }

  void publishFusion() {
    if (!output_pub_) {
      return;
    }

    fv_msgs::msg::DetectionArray fused;
    fused.header.stamp = this->now();
    fused.header.frame_id = frame_id_override_;

    {
      std::lock_guard<std::mutex> lock(data_mutex_);
      for (auto it = active_detections_.begin(); it != active_detections_.end();) {
        auto &record = it->second;
        if (record.hold_frames <= 0) {
          it = active_detections_.erase(it);
          continue;
        }
        record.hold_frames -= 1;
        fused.detections.emplace_back(record.detection);
        ++it;
      }
    }

    output_pub_->publish(fused);
  }

  int32_t makeKey(const std::string &label, const vision_msgs::msg::Detection2D &det) const {
    const double cx = det.bbox.center.position.x;
    const double cy = det.bbox.center.position.y;
    const double w = det.bbox.size_x;
    const double h = det.bbox.size_y;
    std::size_t hash = std::hash<std::string>{}(label);
    hash ^= std::hash<long long>{}(static_cast<long long>(cx * 1000.0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<long long>{}(static_cast<long long>(cy * 1000.0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<long long>{}(static_cast<long long>(w * 1000.0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    hash ^= std::hash<long long>{}(static_cast<long long>(h * 1000.0)) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    return static_cast<int32_t>(hash & 0x7FFFFFFF);
  }

  std::string output_topic_;
  int output_qos_depth_{5};
  double diagnostic_interval_ms_{100.0};
  int hold_frames_{5};
  std::string frame_id_override_;

  std::vector<SourceConfig> sources_;
  std::vector<rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr> source_subs_;

  std::mutex data_mutex_;
  std::unordered_map<int32_t, DetRecord> active_detections_;

  rclcpp::Publisher<fv_msgs::msg::DetectionArray>::SharedPtr output_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace fv_detection_fusion

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::NodeOptions options;
  options.allow_undeclared_parameters(true);
  options.automatically_declare_parameters_from_overrides(true);
  auto node = std::make_shared<fv_detection_fusion::DetectionFusionNode>(options);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}

