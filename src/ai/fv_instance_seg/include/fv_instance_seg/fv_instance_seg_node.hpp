#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <opencv2/core.hpp>
#include <memory>
#include <vector>

#include "fv_instance_seg/backends/inferencer.hpp"

namespace fv_instance_seg {

class InstanceSegNode : public rclcpp::Node {
 public:
  explicit InstanceSegNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());

 private:
  struct TrackState {
    int id = 0;
    cv::Rect bbox;
    cv::Mat mask;
    float score = 0.f;
    int cls = 0;
    cv::Point2f center;
    int misses = 0;
    bool active = false;
    cv::Scalar color{0, 255, 0};
    rclcpp::Time first_seen;
    rclcpp::Time last_seen;
    int age_frames = 0;
  };

  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  void publishOverlay(const cv::Mat& bgr, const std_msgs::msg::Header& header);
  void publishMask(const cv::Mat& mask_mono, const std_msgs::msg::Header& header);
  void publishDetections(const std::vector<TrackState>& tracks, const std_msgs::msg::Header& header);
  void updateTracking(const InferResult& res, const rclcpp::Time& stamp);

  cv::Scalar nextColor();
  void updateStats(double inference_ms, double total_ms, std::size_t detection_count);
  void drawStats(cv::Mat& image);


  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr overlay_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;
  rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr dets_pub_;

  std::unique_ptr<Inferencer> inferencer_;

  std::string backend_;
  std::string model_path_;
  std::string device_;
  std::string fallback_device_;
  std::string input_image_topic_;
  double conf_thres_;
  double iou_thres_;
  bool publish_detections_;
  bool publish_overlay_;
  bool nms_class_agnostic_ = true;
  int max_detections_ = 100;
  bool debug_shapes_ = false;

  std::vector<TrackState> tracks_;
  int next_track_id_ = 1;
  int hold_frames_ = 3;
  int drop_frames_ = 10;
  double match_distance_px_ = 80.0;
  std::vector<cv::Scalar> palette_;
  std::size_t palette_index_ = 0;

  double stats_fps_ = 0.0;
  double stats_inference_ms_ = 0.0;
  double stats_total_ms_ = 0.0;
  std::size_t stats_detection_count_ = 0;
};

} // namespace fv_instance_seg
