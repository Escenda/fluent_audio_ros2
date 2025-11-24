#include "fv_instance_seg/fv_instance_seg_node.hpp"

#include <algorithm>
#include <cstdint>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <fluent_lib/cv_bridge_compat.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <opencv2/imgproc.hpp>
#include "fluent_text.hpp"

namespace fv_instance_seg {

using sensor_msgs::msg::Image;
using vision_msgs::msg::Detection2D;
using vision_msgs::msg::Detection2DArray;
using vision_msgs::msg::ObjectHypothesisWithPose;

static rclcpp::QoS make_qos(const std::string& reliability, int depth) {
  rclcpp::QoS qos(depth);
  if (reliability == "reliable") qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  else qos.reliability(RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT);
  qos.history(RMW_QOS_POLICY_HISTORY_KEEP_LAST);
  return qos;
}

InstanceSegNode::InstanceSegNode(const rclcpp::NodeOptions& options)
    : rclcpp::Node("instance_seg_node", options) {
  backend_ = this->declare_parameter<std::string>("backend", "openvino");
  model_path_ = this->declare_parameter<std::string>("model_path", "");
  device_ = this->declare_parameter<std::string>("device", "CPU");
  fallback_device_ = this->declare_parameter<std::string>("fallback_device", "");
  input_image_topic_ = this->declare_parameter<std::string>("input_image_topic", "/fv/d405/color/image_raw");
  conf_thres_ = this->declare_parameter<double>("conf_thres", 0.25);
  iou_thres_ = this->declare_parameter<double>("iou_thres", 0.5);
  publish_detections_ = this->declare_parameter<bool>("publish_detections", true);
  publish_overlay_ = this->declare_parameter<bool>("publish_overlay", true);
  nms_class_agnostic_ = this->declare_parameter<bool>("nms_class_agnostic", true);
  max_detections_ = this->declare_parameter<int>("max_detections", 100);
  debug_shapes_ = this->declare_parameter<bool>("debug_shapes", false);

  match_distance_px_ = this->declare_parameter<double>("tracking.match_max_distance_px", match_distance_px_);
  hold_frames_ = this->declare_parameter<int>("tracking.hold_frames", hold_frames_);
  drop_frames_ = this->declare_parameter<int>("tracking.drop_frames", drop_frames_);

  const std::vector<int64_t> default_palette_ints;
  std::vector<int64_t> palette_vals = this->declare_parameter<std::vector<int64_t>>(
      "tracking.color_palette_bgr", default_palette_ints);
  if (!palette_vals.empty()) {
    if (palette_vals.size() % 3 != 0) {
      RCLCPP_WARN(this->get_logger(),
                  "tracking.color_palette_bgr size (%zu) is not divisible by 3",
                  palette_vals.size());
    }
    for (std::size_t i = 0; i + 2 < palette_vals.size(); i += 3) {
      palette_.emplace_back(static_cast<double>(palette_vals[i]),
                           static_cast<double>(palette_vals[i + 1]),
                           static_cast<double>(palette_vals[i + 2]));
    }
  }
  if (palette_.empty()) {
    palette_ = {
        cv::Scalar(255, 0, 0),   cv::Scalar(0, 255, 0),   cv::Scalar(0, 0, 255),
        cv::Scalar(255, 255, 0), cv::Scalar(255, 0, 255), cv::Scalar(0, 255, 255)};
  }
  if (drop_frames_ < hold_frames_) {
    drop_frames_ = hold_frames_;
  }
  if (match_distance_px_ <= 0.0) {
    match_distance_px_ = 80.0;
  }

  std::string qos_rel = this->declare_parameter<std::string>("qos.reliability", "best_effort");
  int qos_depth = this->declare_parameter<int>("qos.queue_size", 10);
  auto qos = make_qos(qos_rel, qos_depth);

  overlay_pub_ = this->create_publisher<Image>("overlay", qos);
  mask_pub_ = this->create_publisher<Image>("mask", qos);
  id_mask_pub_ = this->create_publisher<Image>("mask_id", qos);
  if (publish_detections_) dets_pub_ = this->create_publisher<Detection2DArray>("detections", qos);

  inferencer_ = CreateInferencer(backend_);
  if (inferencer_) inferencer_->configure(nms_class_agnostic_, max_detections_, debug_shapes_);
  if (!model_path_.empty() && inferencer_) {
    bool loaded = inferencer_->load(model_path_, device_);
    if (!loaded && !fallback_device_.empty() && fallback_device_ != device_) {
      RCLCPP_WARN(this->get_logger(), "Failed to load model on %s, retrying on %s", device_.c_str(), fallback_device_.c_str());
      loaded = inferencer_->load(model_path_, fallback_device_);
      if (loaded) {
        device_ = fallback_device_;
      }
    }
    if (!loaded) {
      RCLCPP_WARN(this->get_logger(), "Failed to load model: %s on %s", model_path_.c_str(), device_.c_str());
    } else {
      RCLCPP_INFO(this->get_logger(), "Loaded model: %s on %s", model_path_.c_str(), device_.c_str());
    }
  }

  image_sub_ = this->create_subscription<Image>(
      input_image_topic_, qos, std::bind(&InstanceSegNode::imageCallback, this, std::placeholders::_1));
  RCLCPP_INFO(this->get_logger(), "Subscribed to %s", input_image_topic_.c_str());
}

void InstanceSegNode::imageCallback(const Image::SharedPtr msg) {
  auto callback_start = std::chrono::steady_clock::now();

  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
  } catch (const std::exception& e) {
    RCLCPP_WARN(this->get_logger(), "cv_bridge failed: %s", e.what());
    return;
  }

  auto infer_start = std::chrono::steady_clock::now();
  InferResult res;
  bool ok = inferencer_ && inferencer_->infer(cv_ptr->image, static_cast<float>(conf_thres_), static_cast<float>(iou_thres_), &res);
  auto infer_end = std::chrono::steady_clock::now();

  if (!ok) {
    res = InferResult();
  }

  rclcpp::Time stamp(msg->header.stamp);
  if (stamp.nanoseconds() == 0) {
    stamp = this->get_clock()->now();
  }

  updateTracking(res, stamp);

  cv::Mat combined(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1, cv::Scalar(0));
  cv::Mat id_img(cv_ptr->image.rows, cv_ptr->image.cols, CV_8UC1, cv::Scalar(0));
  cv::Mat color_layer(cv_ptr->image.size(), CV_8UC3, cv::Scalar(0, 0, 0));
  std::vector<TrackState> publish_tracks;
  publish_tracks.reserve(tracks_.size());

  for (const auto& track : tracks_) {
    bool keep = (track.active || track.misses <= hold_frames_) && !track.mask.empty();
    if (!keep) {
      continue;
    }

    cv::Mat mask = track.mask;
    if (!mask.empty() && mask.size() != combined.size()) {
      cv::Mat resized;
      cv::resize(mask, resized, combined.size(), 0, 0, cv::INTER_NEAREST);
      mask = resized;
    }

    publish_tracks.push_back(track);
    if (!mask.empty()) {
      combined |= mask;
      // インスタンスIDマスク: トラックID(1..255)を画素値として書き込む
      unsigned char vid = static_cast<unsigned char>(track.id & 0xFF);
      if (vid == 0) vid = 255; // 0は背景に予約
      id_img.setTo(cv::Scalar(vid), mask);
      if (publish_overlay_) {
        cv::Mat colored(color_layer.size(), CV_8UC3, track.color);
        colored.copyTo(color_layer, mask);
      }
    }
  }

  publishMask(combined, msg->header);
  // IDマスクの配信（常時）
  {
    cv_bridge::CvImage out;
    out.header = msg->header;
    out.encoding = sensor_msgs::image_encodings::MONO8;
    out.image = id_img;
    id_mask_pub_->publish(*out.toImageMsg());
  }

  if (publish_overlay_) {
    cv::Mat overlay;
    cv::addWeighted(cv_ptr->image, 0.6, color_layer, 0.4, 0.0, overlay);
    for (const auto& track : publish_tracks) {
      cv::rectangle(overlay, track.bbox, track.color, 2);

      int conf_pct = static_cast<int>(track.score * 100.0f + 0.5f);
      if (conf_pct < 0) conf_pct = 0;
      if (conf_pct > 100) conf_pct = 100;
      double duration_sec = (track.last_seen - track.first_seen).seconds();
      if (duration_sec < 0.0) duration_sec = 0.0;

      std::ostringstream label;
      label << "ID " << track.id
            << " " << conf_pct << "%"
            << " " << std::fixed << std::setprecision(1) << duration_sec << "s"
            << " " << track.age_frames << "f";

      int tx = track.bbox.x;
      int ty = std::max(15, track.bbox.y - 5);
      fluent::text::drawShadow(overlay, label.str(), cv::Point(tx, ty), cv::Scalar(255, 255, 255), cv::Scalar(0,0,0), 0.6, 2, 0);
    }

    drawStats(overlay);
    publishOverlay(overlay, msg->header);
  }

  if (publish_detections_) {
    publishDetections(publish_tracks, msg->header);
  }

  auto callback_end = std::chrono::steady_clock::now();
  double inference_ms = std::chrono::duration<double, std::milli>(infer_end - infer_start).count();
  double total_ms = std::chrono::duration<double, std::milli>(callback_end - callback_start).count();
  updateStats(inference_ms, total_ms, publish_tracks.size());
}

void InstanceSegNode::publishOverlay(const cv::Mat& bgr, const std_msgs::msg::Header& header){ cv_bridge::CvImage out; out.header=header; out.encoding=sensor_msgs::image_encodings::BGR8; out.image=bgr; overlay_pub_->publish(*out.toImageMsg()); }
void InstanceSegNode::publishMask(const cv::Mat& mask, const std_msgs::msg::Header& header){ cv::Mat mono; if(mask.type()!=CV_8UC1) mask.convertTo(mono, CV_8UC1, 255.0); else mono=mask; cv_bridge::CvImage out; out.header=header; out.encoding=sensor_msgs::image_encodings::MONO8; out.image=mono; mask_pub_->publish(*out.toImageMsg()); }

void InstanceSegNode::publishDetections(const std::vector<TrackState>& tracks, const std_msgs::msg::Header& header) {
  Detection2DArray arr;
  arr.header = header;
  arr.detections.reserve(tracks.size());

  for (const auto& track : tracks) {
    Detection2D det;
    det.header = header;
    det.id = std::to_string(track.id);
    det.bbox.center.position.x = track.bbox.x + track.bbox.width * 0.5;
    det.bbox.center.position.y = track.bbox.y + track.bbox.height * 0.5;
    det.bbox.size_x = static_cast<float>(track.bbox.width);
    det.bbox.size_y = static_cast<float>(track.bbox.height);

    ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(track.cls);
    hyp.hypothesis.score = track.score;
    det.results.push_back(hyp);

    arr.detections.push_back(det);
  }

  dets_pub_->publish(arr);
}

void InstanceSegNode::updateTracking(const InferResult& res, const rclcpp::Time& stamp) {
  for (auto& track : tracks_) {
    track.active = false;
  }

  auto toCvRect = [](const cv::Rect& r) {
    cv::Rect clipped = r;
    if (clipped.width < 0) clipped.width = 0;
    if (clipped.height < 0) clipped.height = 0;
    return clipped;
  };

  for (std::size_t i = 0; i < res.boxes.size(); ++i) {
    cv::Rect rect = toCvRect(res.boxes[i]);
    cv::Point2f center(rect.x + rect.width * 0.5f, rect.y + rect.height * 0.5f);
    cv::Mat mask = res.masks.size() > i ? res.masks[i] : cv::Mat();
    if (!mask.empty() && mask.type() != CV_8UC1) {
      cv::Mat tmp;
      mask.convertTo(tmp, CV_8UC1, 255.0);
      mask = tmp;
    }

    TrackState* best = nullptr;
    double best_dist = match_distance_px_;
    for (auto& track : tracks_) {
      double dist = cv::norm(track.center - center);
      if (dist <= best_dist) {
        best_dist = dist;
        best = &track;
      }
    }

    if (best) {
      best->bbox = rect;
      best->mask = mask.clone();
      best->score = res.scores.size() > i ? res.scores[i] : 0.f;
      best->cls = res.classes.size() > i ? res.classes[i] : 0;
      best->center = center;
      best->misses = 0;
      best->active = true;
      best->last_seen = stamp;
      best->age_frames += 1;
    } else {
      TrackState track;
      track.id = next_track_id_++;
      track.bbox = rect;
      track.mask = mask.clone();
      track.score = res.scores.size() > i ? res.scores[i] : 0.f;
      track.cls = res.classes.size() > i ? res.classes[i] : 0;
      track.center = center;
      track.misses = 0;
      track.active = true;
      track.color = nextColor();
      track.first_seen = stamp;
      track.last_seen = stamp;
      track.age_frames = 1;
      tracks_.push_back(std::move(track));
    }
  }

  for (auto& track : tracks_) {
    if (!track.active) {
      track.misses += 1;
    }
    if (track.active) {
      track.last_seen = stamp;
      if (track.age_frames <= 0) {
        track.age_frames = 1;
      }
      if (track.first_seen.nanoseconds() == 0) {
        track.first_seen = stamp;
      }
    }
  }

  tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(), [&](const TrackState& t) {
                     return t.misses > drop_frames_;
                   }),
                 tracks_.end());
}

cv::Scalar InstanceSegNode::nextColor() {
  auto isColorInUse = [&](const cv::Scalar& candidate) {
    for (const auto& track : tracks_) {
      if ((track.active || track.misses <= hold_frames_) && track.color == candidate) {
        return true;
      }
    }
    return false;
  };

  if (!palette_.empty()) {
    const std::size_t palette_size = palette_.size();
    for (std::size_t offset = 0; offset < palette_size; ++offset) {
      std::size_t idx = (palette_index_ + offset) % palette_size;
      const cv::Scalar& candidate = palette_[idx];
      if (!isColorInUse(candidate)) {
        palette_index_ = (idx + 1) % palette_size;
        return candidate;
      }
    }
  }

  for (int attempt = 0; attempt < 360; ++attempt) {
    int seed = next_track_id_ + attempt;
    cv::Scalar candidate(
        static_cast<double>((37 * seed) % 256),
        static_cast<double>((67 * seed) % 256),
        static_cast<double>((97 * seed) % 256));
    if (!isColorInUse(candidate)) {
      return candidate;
    }
  }

  return cv::Scalar(0, 255, 255);
}

void InstanceSegNode::updateStats(double inference_ms, double total_ms, std::size_t detection_count) {
  stats_inference_ms_ = inference_ms;
  stats_total_ms_ = total_ms;
  stats_detection_count_ = detection_count;

  if (total_ms > 0.0) {
    double inst_fps = 1000.0 / total_ms;
    if (stats_fps_ <= 0.0) {
      stats_fps_ = inst_fps;
    } else {
      const double alpha = 0.2;
      stats_fps_ = (1.0 - alpha) * stats_fps_ + alpha * inst_fps;
    }
  }
}

void InstanceSegNode::drawStats(cv::Mat& image) {
  int line = 0;
  auto put = [&](const std::string& s) {
    fluent::text::drawShadow(image, s, cv::Point(10, 30 + line * 22), cv::Scalar(0, 255, 0), cv::Scalar(0,0,0), 0.6, 2, 0);
    ++line;
  };

  put(std::string("デバイス: ") + device_);
  put(std::string("バックエンド: ") + backend_);
  put(std::string("FPS: ") + std::to_string(static_cast<int>(stats_fps_)));
  put(std::string("推論: ") + std::to_string(static_cast<int>(stats_inference_ms_)) + "ms");
  put(std::string("総処理: ") + std::to_string(static_cast<int>(stats_total_ms_)) + "ms");
  put(std::string("セグメント: ") + std::to_string(stats_detection_count_));
  if (!model_path_.empty()) {
    put(std::string("モデル: ") + model_path_);
  }
}

} // namespace fv_instance_seg

int main(int argc, char** argv){ rclcpp::init(argc, argv); auto node= std::make_shared<fv_instance_seg::InstanceSegNode>(); rclcpp::spin(node); rclcpp::shutdown(); return 0; }
