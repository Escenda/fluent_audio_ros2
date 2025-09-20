// fv_stalk_estimator_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/vector3.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <unordered_map>
#include <mutex>

#include <visualization_msgs/msg/marker_array.hpp>

#include "fv_msgs/msg/detection_array.hpp"
#include "fv_msgs/msg/detection_cloud_indices.hpp"
#include "fv_msgs/msg/stalk_metrics.hpp"
#include "fv_msgs/msg/stalk_metrics_array.hpp"

#include "fluent_lib/fluent_cloud/pipeline.hpp" // compute_pca_metrics

namespace fv_stalk_estimator {

using Cloud = pcl::PointCloud<pcl::PointXYZRGB>;
using CloudPtr = Cloud::Ptr;

struct Intrinsics {
  double fx{0}, fy{0}, cx{0}, cy{0};
  std::string frame_id;
  bool valid() const { return fx > 0.0 && fy > 0.0; }
};

class StalkEstimatorNode : public rclcpp::Node {
public:
  explicit StalkEstimatorNode(const rclcpp::NodeOptions &options)
  : rclcpp::Node("fv_stalk_estimator_node", options)
  {
    this->declare_parameter<std::string>("cloud_topic", "/fv/d405/pointcloud/filtered");
    this->declare_parameter<std::string>("counts_topic", "/fv/d405/pointcloud/filtered_counts");
    this->declare_parameter<std::string>("detections_topic", "/fv/d405/detection_fusion/rois");
    this->declare_parameter<std::string>("camera_info_topic", "/fv/d405/color/camera_info");
    this->declare_parameter<std::string>("output_topic", "/fv/d405/stalk/metrics");
    this->declare_parameter<bool>("publish_markers", true);
    this->declare_parameter<std::string>("marker_topic", "/fv/d405/stalk/markers");
    this->declare_parameter<double>("marker_scale_m", 0.015);
    this->declare_parameter<double>("pca_trim_low", 0.05);
    this->declare_parameter<double>("pca_trim_high", 0.95);
    this->declare_parameter<bool>("invert_vertical", false);
    // 根本選択ロジック: 2D投影でより下側(vが大)の端点を根本とする（安定性重視）
    this->declare_parameter<bool>("root_select.nearest_bottom", false);
    this->declare_parameter<double>("root_select.z_margin_m", 0.01);
    // 時間方向スムージング（微振れ抑制）
    this->declare_parameter<bool>("smooth.enable", true);
    this->declare_parameter<double>("smooth.alpha", 0.6);        // prevの寄与（0.0-1.0）
    this->declare_parameter<double>("smooth.max_jump_m", 0.08);  // 大ジャンプ時はリセット

    readParameters();

    // 点群Pubは通常 reliable。ミスマッチ回避のため購読もreliableに合わせる
    cloud_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(cloud_topic_, rclcpp::QoS(10).reliable(),
      std::bind(&StalkEstimatorNode::onCloud, this, std::placeholders::_1));
    counts_sub_ = this->create_subscription<fv_msgs::msg::DetectionCloudIndices>(counts_topic_, rclcpp::QoS(10),
      std::bind(&StalkEstimatorNode::onCounts, this, std::placeholders::_1));
    det_sub_ = this->create_subscription<fv_msgs::msg::DetectionArray>(detections_topic_, rclcpp::QoS(10),
      std::bind(&StalkEstimatorNode::onDetections, this, std::placeholders::_1));
    {
      rclcpp::QoS qos_info(5);
      qos_info.best_effort();
      info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(camera_info_topic_, qos_info,
        std::bind(&StalkEstimatorNode::onCameraInfo, this, std::placeholders::_1));
    }

    pub_ = this->create_publisher<fv_msgs::msg::StalkMetricsArray>(output_topic_, rclcpp::QoS(10));
    if (publish_markers_) {
      marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(marker_topic_, rclcpp::QoS(10));
    }

    RCLCPP_INFO(get_logger(), "fv_stalk_estimator_node ready: cloud=%s counts=%s det=%s out=%s",
                cloud_topic_.c_str(), counts_topic_.c_str(), detections_topic_.c_str(), output_topic_.c_str());
  }

private:
  void readParameters() {
    cloud_topic_ = this->get_parameter("cloud_topic").as_string();
    counts_topic_ = this->get_parameter("counts_topic").as_string();
    detections_topic_ = this->get_parameter("detections_topic").as_string();
    camera_info_topic_ = this->get_parameter("camera_info_topic").as_string();
    output_topic_ = this->get_parameter("output_topic").as_string();
    publish_markers_ = this->get_parameter("publish_markers").as_bool();
    marker_topic_ = this->get_parameter("marker_topic").as_string();
    marker_scale_m_ = this->get_parameter("marker_scale_m").as_double();
    pca_trim_low_ = this->get_parameter("pca_trim_low").as_double();
    pca_trim_high_ = this->get_parameter("pca_trim_high").as_double();
    invert_vertical_ = this->get_parameter("invert_vertical").as_bool();
    root_nearest_bottom_ = this->get_parameter("root_select.nearest_bottom").as_bool();
    root_z_margin_m_ = this->get_parameter("root_select.z_margin_m").as_double();
    smooth_enable_ = this->get_parameter("smooth.enable").as_bool();
    smooth_alpha_ = this->get_parameter("smooth.alpha").as_double();
    smooth_max_jump_m_ = this->get_parameter("smooth.max_jump_m").as_double();
    if (pca_trim_low_ < 0.0) pca_trim_low_ = 0.0;
    if (pca_trim_high_ > 1.0) pca_trim_high_ = 1.0;
    if (pca_trim_high_ <= pca_trim_low_) pca_trim_high_ = std::min(0.95, pca_trim_low_ + 0.8);
  }

  void onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    intr_.fx = msg->k[0]; intr_.fy = msg->k[4]; intr_.cx = msg->k[2]; intr_.cy = msg->k[5];
    intr_.frame_id = msg->header.frame_id;
  }

  void onDetections(const fv_msgs::msg::DetectionArray::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_det_stamp_ = msg->header.stamp;
    det_map_.clear();
    for (const auto &d : msg->detections) {
      det_map_[d.id] = d;
    }
  }

  void onCloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    std::lock_guard<std::mutex> lk(mutex_);
    last_cloud_msg_ = msg;
  }

  void onCounts(const fv_msgs::msg::DetectionCloudIndices::SharedPtr msg) {
    // 入力が揃っているか確認
    sensor_msgs::msg::PointCloud2::SharedPtr cloud_msg;
    {
      std::lock_guard<std::mutex> lk(mutex_);
      cloud_msg = last_cloud_msg_;
    }
    if (!cloud_msg) {
      RCLCPP_WARN_THROTTLE(get_logger(), *this->get_clock(), 2000, "Waiting for aggregated cloud");
      return;
    }

    CloudPtr cloud(new Cloud);
    pcl::fromROSMsg(*cloud_msg, *cloud);
    if (cloud->empty()) {
      return;
    }

    // オフセットで分割
    size_t offset = 0;
    fv_msgs::msg::StalkMetricsArray out;
    out.header = msg->header;
    // metricsのframe_idは可能ならカメラのframe_idに統一（TF不整合の回避）
    if (!intr_.frame_id.empty()) {
      out.header.frame_id = intr_.frame_id;
    }

    for (size_t i = 0; i < msg->ids.size() && i < msg->counts.size(); ++i) {
      const int32_t id = msg->ids[i];
      const uint32_t count = msg->counts[i];
      if (count == 0) {
        continue;
      }
      if (offset + count > cloud->points.size()) {
        RCLCPP_WARN(get_logger(), "Count exceeds cloud size: offset=%zu count=%u total=%zu", offset, count, cloud->points.size());
        break;
      }

      CloudPtr sub(new Cloud);
      sub->points.reserve(count);
      sub->points.insert(sub->points.end(), cloud->points.begin() + static_cast<long>(offset), cloud->points.begin() + static_cast<long>(offset + count));
      sub->width = sub->points.size(); sub->height = 1; sub->is_dense = false;
      offset += count;

      auto metrics = fluent_cloud::compute_pca_metrics(sub);
      // トリム再計算（pca_trim_low_/high_が既定と異なる場合）
      if (pca_trim_low_ != 0.05 || pca_trim_high_ != 0.95) {
        std::vector<float> ts; ts.reserve(sub->points.size());
        auto isFinite = [](const pcl::PointXYZRGB &p){ return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z) && p.z > 0.0f; };
        for (const auto &p : sub->points) if (isFinite(p)) {
          Eigen::Vector3f v(p.x,p.y,p.z); ts.push_back(metrics.axis.dot(v - metrics.center));
        }
        if (!ts.empty()) {
          size_t n = ts.size();
          size_t il = static_cast<size_t>(std::floor(n * std::max(0.0, std::min(1.0, pca_trim_low_))));
          size_t ih = static_cast<size_t>(std::floor(n * std::max(0.0, std::min(1.0, pca_trim_high_))));
          if (il >= n) il = n-1; if (ih >= n) ih = n-1;
          std::nth_element(ts.begin(), ts.begin()+il, ts.end());
          float tmin = ts[il];
          std::nth_element(ts.begin(), ts.begin()+ih, ts.end());
          float tmax = ts[ih];
          if (tmax < tmin) std::swap(tmax, tmin);
          metrics.tmin = tmin; metrics.tmax = tmax;
          metrics.length_m = static_cast<double>(tmax - tmin);
        }
      }
      // エンドポイント（カメラ座標）
      Eigen::Vector3f p_root, p_tip;
      Eigen::Vector3f pmin = metrics.center + metrics.axis * metrics.tmin;
      Eigen::Vector3f pmax = metrics.center + metrics.axis * metrics.tmax;

      // 2D投影の下側(=vが大)を根本とみなす。PCA軸の符号反転に対して安定。
      double vmin = projectV(pmin);
      double vmax = projectV(pmax);
      bool use_y_fallback = !intr_.valid();
      double a = use_y_fallback ? static_cast<double>(pmin.y()) : vmin;
      double b = use_y_fallback ? static_cast<double>(pmax.y()) : vmax;
      if (b >= a) { p_root = pmax; p_tip = pmin; } else { p_root = pmin; p_tip = pmax; }
      if (invert_vertical_) std::swap(p_root, p_tip);

      // 時系列スムージング（IDごと）
      if (smooth_enable_) {
        auto it = last_rt_.find(id);
        if (it == last_rt_.end()) {
          last_rt_[id] = {p_root, p_tip};
        } else {
          auto prev = it->second;
          auto dist = [](const Eigen::Vector3f &a, const Eigen::Vector3f &b){ return (a-b).norm(); };
          bool reset = (dist(prev.first, p_root) > static_cast<float>(smooth_max_jump_m_)) ||
                       (dist(prev.second, p_tip) > static_cast<float>(smooth_max_jump_m_));
          if (reset) {
            it->second = {p_root, p_tip};
          } else {
            float ap = static_cast<float>(std::clamp(smooth_alpha_, 0.0, 1.0));
            Eigen::Vector3f nr = ap * prev.first  + (1.0f - ap) * p_root;
            Eigen::Vector3f nt = ap * prev.second + (1.0f - ap) * p_tip;
            it->second = {nr, nt};
            p_root = nr; p_tip = nt;
          }
        }
      }

      fv_msgs::msg::StalkMetrics m;
      m.header = out.header;
      m.id = id;
      m.root_camera = toPointMsg(p_root);
      m.tip_camera = toPointMsg(p_tip);
      m.axis_camera = toVecMsg(metrics.axis);
      m.length_m = static_cast<float>(metrics.length_m);
      m.curvature = static_cast<float>(metrics.curvature_ratio);
      m.thickness_m = static_cast<float>(metrics.diameter_m);
      m.root_radius_m = 0.0f;
      m.grade = m.GRADE_UNKNOWN;
      m.root_world = m.root_camera; // TODO: TFがあれば変換
      m.tip_world = m.tip_camera;
      m.updated_at = msg->header.stamp;
      // 検出信頼度
      {
        std::lock_guard<std::mutex> lk(mutex_);
        auto it = det_map_.find(id);
        if (it != det_map_.end()) {
          m.confidence = std::max(0.f, std::min(1.f, it->second.conf_fused));
        } else {
          m.confidence = 0.0f;
        }
      }
      out.stalks.push_back(std::move(m));
    }

    if (!out.stalks.empty()) {
      pub_->publish(out);
      if (publish_markers_ && marker_pub_) {
        publishMarkers(out);
      }
    }
  }

  geometry_msgs::msg::Point toPointMsg(const Eigen::Vector3f &p) const {
    geometry_msgs::msg::Point q; q.x = p.x(); q.y = p.y(); q.z = p.z(); return q;
  }
  geometry_msgs::msg::Vector3 toVecMsg(const Eigen::Vector3f &v) const {
    geometry_msgs::msg::Vector3 q; q.x = v.x(); q.y = v.y(); q.z = v.z(); return q;
  }

  double projectV(const Eigen::Vector3f &p) const {
    // v = cy + fy * y / z （z>0前提）
    double z = static_cast<double>(p.z());
    if (z <= 0.0 || !intr_.valid()) return 0.0;
    return intr_.cy + intr_.fy * static_cast<double>(p.y()) / z;
  }

  void publishMarkers(const fv_msgs::msg::StalkMetricsArray &arr) {
    visualization_msgs::msg::MarkerArray ma;
    // Clear previous markers
    visualization_msgs::msg::Marker clear;
    clear.action = visualization_msgs::msg::Marker::DELETEALL;
    ma.markers.push_back(clear);

    std::string frame = intr_.frame_id.empty() ? arr.header.frame_id : intr_.frame_id;
    rclcpp::Time now = this->now();

    for (const auto &m : arr.stalks) {
      if (m.root_camera.z <= 0.0f) continue;
      visualization_msgs::msg::Marker mk;
      mk.header.frame_id = frame;
      mk.header.stamp = now;
      mk.ns = "stalk_root";
      mk.id = static_cast<int>(m.id);
      mk.type = visualization_msgs::msg::Marker::SPHERE;
      mk.action = visualization_msgs::msg::Marker::ADD;
      mk.pose.position.x = m.root_camera.x;
      mk.pose.position.y = m.root_camera.y;
      mk.pose.position.z = m.root_camera.z;
      mk.pose.orientation.w = 1.0;
      mk.scale.x = marker_scale_m_;
      mk.scale.y = marker_scale_m_;
      mk.scale.z = marker_scale_m_;
      mk.color.r = 1.0f; mk.color.g = 0.0f; mk.color.b = 0.0f; mk.color.a = 1.0f;
      mk.lifetime = rclcpp::Duration(0, 0); // persistent until overwritten
      ma.markers.push_back(std::move(mk));
    }
    if (!ma.markers.empty() && marker_pub_) marker_pub_->publish(ma);
  }

  // params
  std::string cloud_topic_;
  std::string counts_topic_;
  std::string detections_topic_;
  std::string camera_info_topic_;
  std::string output_topic_;
  bool publish_markers_{true};
  std::string marker_topic_;
  double marker_scale_m_{0.015};
  double pca_trim_low_{0.05};
  double pca_trim_high_{0.95};
  bool invert_vertical_{false};
  bool root_nearest_bottom_{true};
  double root_z_margin_m_{0.01};
  bool smooth_enable_{true};
  double smooth_alpha_{0.6};
  double smooth_max_jump_m_{0.08};

  // subs/pubs
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_sub_;
  rclcpp::Subscription<fv_msgs::msg::DetectionCloudIndices>::SharedPtr counts_sub_;
  rclcpp::Subscription<fv_msgs::msg::DetectionArray>::SharedPtr det_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<fv_msgs::msg::StalkMetricsArray>::SharedPtr pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

  // state
  std::mutex mutex_;
  sensor_msgs::msg::PointCloud2::SharedPtr last_cloud_msg_;
  std::unordered_map<int32_t, fv_msgs::msg::Detection2D> det_map_;
  builtin_interfaces::msg::Time last_det_stamp_;
  Intrinsics intr_;
  std::unordered_map<int32_t, std::pair<Eigen::Vector3f, Eigen::Vector3f>> last_rt_;
};

} // namespace fv_stalk_estimator

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fv_stalk_estimator::StalkEstimatorNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
