// Minimal C++ Asparagus UI overlay node
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc.hpp>
#include "fluent_text.hpp"
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "fv_msgs/msg/detection_array.hpp"
#include "fv_msgs/msg/stalk_metrics_array.hpp"

namespace fv_aspara_ui_cpp {

class AsparaUiCppNode : public rclcpp::Node {
public:
  explicit AsparaUiCppNode(const rclcpp::NodeOptions &options)
    : rclcpp::Node("fv_aspara_ui_cpp", options)
  {
    declare_parameter<std::string>("image_topic", "/fv/selected/d405/color/image_raw");
    declare_parameter<std::string>("detections_topic", "/fv/d405/detection_fusion/rois");
    declare_parameter<std::string>("metrics_topic", "/fv/d405/stalk/metrics");
    declare_parameter<std::string>("camera_info_topic", "/fv/d405/color/camera_info");
    declare_parameter<std::string>("annotated_topic", "/fv/d405/ui/annotated_image");
    // Optional mask overlay
    declare_parameter<std::string>("mask_topic", "");
    declare_parameter<int>("mask_threshold", 128);
    declare_parameter<double>("mask_overlay_alpha", 0.30);
    declare_parameter<int>("mask_overlay_color_bgr[0]", 0);
    declare_parameter<int>("mask_overlay_color_bgr[1]", 255);
    declare_parameter<int>("mask_overlay_color_bgr[2]", 0);
    // IDごとの色（B,G,Rの順で3つ1組）
    declare_parameter<std::vector<int>>("id_color_palette_bgr",
      std::vector<int>{
        255,0,0,  0,255,0,  0,0,255,  255,255,0,  255,0,255,  0,255,255,
        0,165,255, 128,0,128, 0,255,128, 128,128,0, 128,128,128
      });
    declare_parameter<bool>("draw_labels", true);
    declare_parameter<bool>("draw_confidence", true);
    declare_parameter<bool>("draw_metrics", true);
    declare_parameter<bool>("draw_root_tip", true);
    // 画面左上タイトルは使わず、各矩形にタイトル（アスパラ）を描画する方針
    // 後方互換のため draw_title パラメータは宣言のみ行い無視する
    declare_parameter<bool>("draw_title", false);
    // Invalid area overlay (same schema as detection_fusion)
    declare_parameter<bool>("draw_invalid_area", true);
    declare_parameter<bool>("invalid_area.enabled", false);
    declare_parameter<bool>("invalid_area.any_overlap", true);
    declare_parameter<bool>("invalid_area.touch_inclusive", true);
    declare_parameter<double>("invalid_area.box.center_x_ratio", 0.5);
    declare_parameter<double>("invalid_area.box.y_top_ratio", 0.5);
    declare_parameter<double>("invalid_area.box.width_ratio", 0.5);
    declare_parameter<double>("invalid_area.box.height_ratio", 0.5);
    // Selection drawing
    declare_parameter<bool>("draw_selection_box", true);
    declare_parameter<std::string>("selection.mode", "auto"); // auto|fixed
    declare_parameter<std::string>("selection.auto_strategy", "highest_conf"); // highest_conf|largest_height
    declare_parameter<double>("selection.fixed.box.x_center_ratio", 0.5);
    declare_parameter<int>("selection.fixed.box.y_top_px", 5);
    declare_parameter<double>("selection.fixed.box.width_ratio", 0.20);
    declare_parameter<double>("selection.fixed.box.height_ratio", 0.50);
    // Display filter (UI only)
    declare_parameter<bool>("display_filter.only_vertical", false);
    declare_parameter<double>("display_filter.min_aspect_ratio", 0.05);
    declare_parameter<double>("display_filter.max_aspect_ratio", 2.0);
    declare_parameter<int>("display_filter.min_w_px", 1);
    declare_parameter<int>("display_filter.min_h_px", 1);

    image_topic_ = get_parameter("image_topic").as_string();
    detections_topic_ = get_parameter("detections_topic").as_string();
    metrics_topic_ = get_parameter("metrics_topic").as_string();
    camera_info_topic_ = get_parameter("camera_info_topic").as_string();
    annotated_topic_ = get_parameter("annotated_topic").as_string();
    mask_topic_ = get_parameter("mask_topic").as_string();
    mask_threshold_ = get_parameter("mask_threshold").as_int();
    mask_alpha_ = get_parameter("mask_overlay_alpha").as_double();
    mask_color_bgr_[0] = get_parameter("mask_overlay_color_bgr[0]").as_int();
    mask_color_bgr_[1] = get_parameter("mask_overlay_color_bgr[1]").as_int();
    mask_color_bgr_[2] = get_parameter("mask_overlay_color_bgr[2]").as_int();
    {
      auto pal = get_parameter("id_color_palette_bgr").as_integer_array();
      id_palette_bgr_.assign(pal.begin(), pal.end());
      rebuildPaletteColors();
    }
    draw_labels_ = get_parameter("draw_labels").as_bool();
    draw_confidence_ = get_parameter("draw_confidence").as_bool();
    draw_metrics_ = get_parameter("draw_metrics").as_bool();
    draw_root_tip_ = get_parameter("draw_root_tip").as_bool();
    draw_title_ = get_parameter("draw_title").as_bool();
    draw_invalid_area_ = get_parameter("draw_invalid_area").as_bool();
    invalid_enabled_ = get_parameter("invalid_area.enabled").as_bool();
    invalid_any_overlap_ = get_parameter("invalid_area.any_overlap").as_bool();
    invalid_touch_inclusive_ = get_parameter("invalid_area.touch_inclusive").as_bool();
    ia_cx_ = get_parameter("invalid_area.box.center_x_ratio").as_double();
    ia_yt_ = get_parameter("invalid_area.box.y_top_ratio").as_double();
    ia_wr_ = get_parameter("invalid_area.box.width_ratio").as_double();
    ia_hr_ = get_parameter("invalid_area.box.height_ratio").as_double();
    draw_selection_box_ = get_parameter("draw_selection_box").as_bool();
    sel_mode_ = get_parameter("selection.mode").as_string();
    sel_auto_strategy_ = get_parameter("selection.auto_strategy").as_string();
    sel_fix_xc_ = get_parameter("selection.fixed.box.x_center_ratio").as_double();
    sel_fix_yt_px_ = get_parameter("selection.fixed.box.y_top_px").as_int();
    sel_fix_wr_ = get_parameter("selection.fixed.box.width_ratio").as_double();
    sel_fix_hr_ = get_parameter("selection.fixed.box.height_ratio").as_double();
    disp_only_vertical_ = get_parameter("display_filter.only_vertical").as_bool();
    disp_min_ar_ = get_parameter("display_filter.min_aspect_ratio").as_double();
    disp_max_ar_ = get_parameter("display_filter.max_aspect_ratio").as_double();
    disp_min_w_ = get_parameter("display_filter.min_w_px").as_int();
    disp_min_h_ = get_parameter("display_filter.min_h_px").as_int();

    rclcpp::QoS qos_sensor(5);
    qos_sensor.best_effort();

    sub_img_ = create_subscription<sensor_msgs::msg::Image>(image_topic_, qos_sensor,
      std::bind(&AsparaUiCppNode::onImage, this, std::placeholders::_1));
    sub_det_ = create_subscription<fv_msgs::msg::DetectionArray>(detections_topic_, 10,
      std::bind(&AsparaUiCppNode::onDetections, this, std::placeholders::_1));
    if (!metrics_topic_.empty()) {
      sub_metrics_ = create_subscription<fv_msgs::msg::StalkMetricsArray>(metrics_topic_, 10,
        std::bind(&AsparaUiCppNode::onMetrics, this, std::placeholders::_1));
    }
    if (!mask_topic_.empty()) {
      // detection_fusion の roi_mask はデフォルトで reliable 発行
      // QoSをreliableに合わせて互換性を確保
      rclcpp::QoS qos_mask(rclcpp::KeepLast(5));
      qos_mask.reliable();
      sub_mask_ = create_subscription<sensor_msgs::msg::Image>(mask_topic_, qos_mask,
        std::bind(&AsparaUiCppNode::onMask, this, std::placeholders::_1));
    }
    if (!camera_info_topic_.empty()) {
      rclcpp::QoS qos_info(5);
      qos_info.best_effort();
      sub_cinfo_ = create_subscription<sensor_msgs::msg::CameraInfo>(camera_info_topic_, qos_info,
        std::bind(&AsparaUiCppNode::onCameraInfo, this, std::placeholders::_1));
    }

    pub_anno_ = create_publisher<sensor_msgs::msg::Image>(annotated_topic_, qos_sensor);

    RCLCPP_INFO(get_logger(), "Aspara UI CPP started: img=%s det=%s met=%s out=%s",
                image_topic_.c_str(), detections_topic_.c_str(), metrics_topic_.c_str(), annotated_topic_.c_str());
  }

private:
  void onDetections(const fv_msgs::msg::DetectionArray::SharedPtr msg) {
    last_dets_ = msg;
  }

  void onMetrics(const fv_msgs::msg::StalkMetricsArray::SharedPtr msg) {
    last_metrics_ = msg;
  }

  void onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
    fx_ = msg->k[0]; fy_ = msg->k[4]; cx_ = msg->k[2]; cy_ = msg->k[5];
  }

  void onImage(const sensor_msgs::msg::Image::SharedPtr msg) {
    cv_bridge::CvImageConstPtr bridge;
    try {
      bridge = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception &ex) {
      RCLCPP_WARN(get_logger(), "cv_bridge failed: %s", ex.what());
      return;
    }
    cv::Mat view = bridge->image.clone();

    // compute invalid area rect in pixels if enabled
    cv::Rect invalid_rect;
    if (invalid_enabled_) {
      invalid_rect = makeInvalidRect(view.cols, view.rows);
    }

    int selected_index = -1;
    if (sel_mode_ == "auto" && last_dets_ && !last_dets_->detections.empty()) {
      selected_index = pickSelectedIndex(*last_dets_, sel_auto_strategy_, view.cols, view.rows);
    }

    int selected_id = -1;
    if (last_dets_ && selected_index >= 0 && selected_index < static_cast<int>(last_dets_->detections.size())) {
      selected_id = last_dets_->detections[static_cast<size_t>(selected_index)].id;
    }

    if (last_dets_) {
      for (size_t idx = 0; idx < last_dets_->detections.size(); ++idx) {
        const auto &d = last_dets_->detections[idx];
        int x1 = static_cast<int>(std::floor(d.bbox_min.x));
        int y1 = static_cast<int>(std::floor(d.bbox_min.y));
        int x2 = static_cast<int>(std::ceil(d.bbox_max.x));
        int y2 = static_cast<int>(std::ceil(d.bbox_max.y));
        x1 = std::clamp(x1, 0, std::max(0, view.cols-1));
        y1 = std::clamp(y1, 0, std::max(0, view.rows-1));
        x2 = std::clamp(x2, 0, std::max(0, view.cols-1));
        y2 = std::clamp(y2, 0, std::max(0, view.rows-1));
        cv::Rect bb(cv::Point(x1,y1), cv::Point(x2,y2));
        if (!passDisplayFilter(bb)) {
          continue; // UI表示フィルタ
        }
        // draw bbox: non-selected gray, selected green
        bool is_selected = (static_cast<int>(idx) == selected_index);
        cv::Scalar color = is_selected ? cv::Scalar(0, 255, 0) : cv::Scalar(128, 128, 128);
        int thickness = is_selected && draw_selection_box_ ? 2 : 1; // thinner selection box
        cv::rectangle(view, bb, color, thickness);
        if (draw_labels_ || draw_confidence_) {
          std::string label;
          if (draw_labels_) label = "ID " + std::to_string(d.id);
          if (draw_confidence_) {
            double conf = static_cast<double>(d.conf_fused);
            char buf[64]; std::snprintf(buf, sizeof(buf), " (%.2f)", conf);
            label += buf;
          }
          if (!label.empty()) {
            cv::putText(view, label, cv::Point(x1, std::max(0, y1-5)), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
          }
        }
      }
    }

    // mask overlay per asparagus ID (instance id -> detection id -> palette color)
    if (last_mask_ && !mask_topic_.empty()) {
      auto sp = last_mask_;
      if (sp) {
        try {
          cv_bridge::CvImageConstPtr mbridge = cv_bridge::toCvShare(sp);
          cv::Mat mask;
          if (mbridge->image.channels() == 1) {
            mask = mbridge->image;
          } else {
            cv::cvtColor(mbridge->image, mask, cv::COLOR_BGR2GRAY);
          }
          if (mask.size() != view.size()) {
            cv::resize(mask, mask, view.size(), 0, 0, cv::INTER_NEAREST);
          }
          // build LUT for instance id -> BGR based on current detections
          cv::Vec3b lut[256];
          for (int i=0;i<256;++i) lut[i] = cv::Vec3b((uchar)mask_color_bgr_[0], (uchar)mask_color_bgr_[1], (uchar)mask_color_bgr_[2]);
          if (last_dets_) {
            for (const auto &d : last_dets_->detections) {
              uint32_t inst = d.mask_instance_id;
              if (inst == 0) continue;
              cv::Vec3b c = colorForDetId(d.id);
              lut[inst & 0xFF] = c;
            }
          }
          cv::Mat overlay = view.clone();
          if (mask.type() == CV_8UC1) {
            for (int y = 0; y < mask.rows; ++y) {
              const uint8_t* mr = mask.ptr<uint8_t>(y);
              cv::Vec3b* ov = overlay.ptr<cv::Vec3b>(y);
              for (int x = 0; x < mask.cols; ++x) {
                uint8_t m = mr[x];
                // インスタンスID（0=背景）。非ゼロを着色対象とする
                if (m != 0) { ov[x] = lut[m]; }
              }
            }
            // ROIマスク（二値, detection_fusionのroi_maskなど）の場合でも
            // 検出ごとに矩形領域を自色で上書きしてインスタンス別の色分けを担保する
            if (last_dets_) {
              for (const auto &d : last_dets_->detections) {
                int x1 = std::max(0, (int)std::floor(d.bbox_min.x));
                int y1 = std::max(0, (int)std::floor(d.bbox_min.y));
                int x2 = std::min(view.cols, (int)std::ceil(d.bbox_max.x));
                int y2 = std::min(view.rows, (int)std::ceil(d.bbox_max.y));
                if (x2 <= x1 || y2 <= y1) continue;
                cv::Rect roi(x1,y1,x2-x1,y2-y1);
                cv::Vec3b ic = colorForDetId(d.id);
                for (int y = roi.y; y < roi.y + roi.height; ++y) {
                  const uint8_t* mr = mask.ptr<uint8_t>(y);
                  cv::Vec3b* ov = overlay.ptr<cv::Vec3b>(y);
                  for (int x = roi.x; x < roi.x + roi.width; ++x) {
                    if (mr[x] != 0) ov[x] = ic;
                  }
                }
              }
            }
          } else {
            // fallback for other types: threshold and uniform palette[0]
            for (int y = 0; y < mask.rows; ++y) {
              const uint16_t* mr = mask.ptr<uint16_t>(y);
              cv::Vec3b* ov = overlay.ptr<cv::Vec3b>(y);
              for (int x = 0; x < mask.cols; ++x) {
                uint16_t m = mr[x];
                if ((m & 0xFF) != 0) { ov[x] = lut[(m & 0xFF)]; }
              }
            }
          }
          double a = std::clamp(mask_alpha_, 0.0, 1.0);
          cv::addWeighted(overlay, a, view, 1.0 - a, 0.0, view);
        } catch (...) {}
      }
    }

    // Draw per-bbox JP title after mask overlay (so it stays on top)
    if (last_dets_) {
      for (size_t idx = 0; idx < last_dets_->detections.size(); ++idx) {
        const auto &d = last_dets_->detections[idx];
        int x1 = static_cast<int>(std::floor(d.bbox_min.x));
        int y1 = static_cast<int>(std::floor(d.bbox_min.y));
        int x2 = static_cast<int>(std::ceil(d.bbox_max.x));
        int y2 = static_cast<int>(std::ceil(d.bbox_max.y));
        x1 = std::clamp(x1, 0, std::max(0, view.cols-1));
        y1 = std::clamp(y1, 0, std::max(0, view.rows-1));
        x2 = std::clamp(x2, 0, std::max(0, view.cols-1));
        y2 = std::clamp(y2, 0, std::max(0, view.rows-1));
        cv::Rect bb(cv::Point(x1,y1), cv::Point(x2,y2));
        if (!passDisplayFilter(bb)) continue;
        int title_y = std::max(0, y1 - 6);
        int perc = 0;
        if (idx < last_dets_->detections.size()) {
          const auto &dd = last_dets_->detections[idx];
          double p = std::max(0.0, std::min(1.0, static_cast<double>(dd.conf_fused)));
          perc = static_cast<int>(std::round(p * 100.0));
        }
        char title[64];
        std::snprintf(title, sizeof(title), "アスパラ#%d(%d%%)", d.id, perc);
        fluent::text::drawShadow(view, std::string(title), cv::Point(x1 + 2, title_y),
                                 cv::Scalar(255,255,255), cv::Scalar(0,0,0), 0.6, 2, 0);
      }
    }

    // draw fixed selection box if requested
    if (draw_selection_box_ && sel_mode_ == "fixed") {
      cv::Rect fixed = makeFixedRect(view.cols, view.rows);
      cv::rectangle(view, fixed, cv::Scalar(0, 255, 255), 2);
    }

    // draw invalid area overlay last
    if (draw_invalid_area_ && invalid_enabled_) {
      cv::Mat overlay = view.clone();
      cv::rectangle(overlay, invalid_rect, cv::Scalar(0, 0, 255), cv::FILLED);
      double alpha = 0.25; // 25% red tint
      cv::addWeighted(overlay, alpha, view, 1.0 - alpha, 0.0, view);
      // border
      cv::rectangle(view, invalid_rect, cv::Scalar(0, 0, 255), 2);
    }

    // metrics: 表示対象の検出枠(id)が存在するものだけ描画
    if (draw_metrics_ && last_metrics_ && last_dets_ && !last_dets_->detections.empty()) {
      std::unordered_set<int> valid_ids;
      for (const auto &d : last_dets_->detections) {
        int x1 = static_cast<int>(std::floor(d.bbox_min.x));
        int y1 = static_cast<int>(std::floor(d.bbox_min.y));
        int x2 = static_cast<int>(std::ceil(d.bbox_max.x));
        int y2 = static_cast<int>(std::ceil(d.bbox_max.y));
        x1 = std::clamp(x1, 0, std::max(0, view.cols-1));
        y1 = std::clamp(y1, 0, std::max(0, view.rows-1));
        x2 = std::clamp(x2, 0, std::max(0, view.cols-1));
        y2 = std::clamp(y2, 0, std::max(0, view.rows-1));
        cv::Rect bb(cv::Point(x1,y1), cv::Point(x2,y2));
        if (passDisplayFilter(bb)) valid_ids.insert(d.id);
      }
      if (!valid_ids.empty()) {
        for (const auto &m : last_metrics_->stalks) {
          if (valid_ids.find(m.id) == valid_ids.end()) continue;

          cv::Point mid_pt(10, 20);
          if (validIntrinsics()) {
            if (m.root_camera.z > 0.0 && m.tip_camera.z > 0.0) {
              auto uv0 = project(m.root_camera.x, m.root_camera.y, m.root_camera.z);
              auto uv1 = project(m.tip_camera.x, m.tip_camera.y, m.tip_camera.z);
              cv::Point p0((int)uv0.first,(int)uv0.second);
              cv::Point p1((int)uv1.first,(int)uv1.second);
              mid_pt = cv::Point((p0.x + p1.x)/2, (p0.y + p1.y)/2);
            } else if (m.tip_camera.z > 0.0) {
              auto uv = project(m.tip_camera.x, m.tip_camera.y, m.tip_camera.z);
              mid_pt = cv::Point((int)uv.first,(int)uv.second);
            }
          }

          char lenjp[64];
          std::snprintf(lenjp, sizeof(lenjp), "長さ %.1f cm", m.length_m * 100.0);
          fluent::text::drawShadow(view, std::string(lenjp), mid_pt + cv::Point(8,-8),
                                   cv::Scalar(255,255,255), cv::Scalar(0,0,0), 0.7, 2, 0);
          char diajp[64];
          std::snprintf(diajp, sizeof(diajp), "直径 %.1f mm", m.thickness_m * 1000.0);
          fluent::text::drawShadow(view, std::string(diajp), mid_pt + cv::Point(8,14),
                                   cv::Scalar(200,255,200), cv::Scalar(0,0,0), 0.6, 1, 0);

          if (draw_root_tip_ && validIntrinsics()) {
            if (m.root_camera.z > 0.0) {
              auto uv0 = project(m.root_camera.x, m.root_camera.y, m.root_camera.z);
              cv::Point p0((int)uv0.first,(int)uv0.second);
              cv::circle(view, p0, 4, cv::Scalar(0,0,255), -1);
              if (selected_id == m.id) cv::circle(view, p0, 7, cv::Scalar(0,255,255), 2);
              char xyz[128];
              std::snprintf(xyz, sizeof(xyz), "x=%.1f y=%.1f z=%.1f cm",
                            (double)m.root_camera.x*100.0, (double)m.root_camera.y*100.0, (double)m.root_camera.z*100.0);
              cv::Point torg = p0 + cv::Point(8, -6);
              fluent::text::drawShadow(view, std::string(xyz), torg,
                                       cv::Scalar(255,255,255), cv::Scalar(0,0,0), 0.5, 1, 0);
            }
            if (m.tip_camera.z > 0.0) {
              auto uv1 = project(m.tip_camera.x, m.tip_camera.y, m.tip_camera.z);
              cv::circle(view, cv::Point((int)uv1.first,(int)uv1.second), 4, cv::Scalar(255,0,0), -1);
              if (m.root_camera.z > 0.0) {
                auto uv0 = project(m.root_camera.x, m.root_camera.y, m.root_camera.z);
                cv::line(view, cv::Point((int)uv0.first,(int)uv0.second), cv::Point((int)uv1.first,(int)uv1.second), cv::Scalar(0,200,255), 2);
              }
            }
          }
        }
      }
    }

    // no global title; each bbox has its own JP title

    auto out = cv_bridge::CvImage(msg->header, sensor_msgs::image_encodings::BGR8, view).toImageMsg();
    pub_anno_->publish(*out);
  }

  bool validIntrinsics() const { return fx_ > 0.0 && fy_ > 0.0; }
  std::pair<double,double> project(double x, double y, double z) const {
    if (z <= 0.0 || !validIntrinsics()) return {0.0, 0.0};
    double u = cx_ + fx_ * (x / z);
    double v = cy_ + fy_ * (y / z);
    return {u, v};
  }

  cv::Rect makeInvalidRect(int width, int height) const {
    int w = std::max(1, static_cast<int>(std::round(width * ia_wr_)));
    int h = std::max(1, static_cast<int>(std::round(height * ia_hr_)));
    int cx = static_cast<int>(std::round(width * ia_cx_));
    int x0 = std::clamp(cx - w/2, 0, std::max(0, width - 1));
    int y0 = std::clamp(static_cast<int>(std::round(height * ia_yt_)), 0, std::max(0, height - 1));
    int x1 = std::clamp(x0 + w, 0, width);
    int y1 = std::clamp(y0 + h, 0, height);
    return cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
  }

  cv::Rect makeFixedRect(int width, int height) const {
    int w = std::max(1, static_cast<int>(std::round(width * sel_fix_wr_)));
    int h = std::max(1, static_cast<int>(std::round(height * sel_fix_hr_)));
    int cx = static_cast<int>(std::round(width * sel_fix_xc_));
    int x0 = std::clamp(cx - w/2, 0, std::max(0, width - 1));
    int y0 = std::clamp(sel_fix_yt_px_, 0, std::max(0, height - 1));
    int x1 = std::clamp(x0 + w, 0, width);
    int y1 = std::clamp(y0 + h, 0, height);
    return cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
  }

  static double aspectRatio(const cv::Rect &r) {
    if (r.height <= 0) return 0.0; return static_cast<double>(r.width) / static_cast<double>(r.height);
  }

  bool passDisplayFilter(const cv::Rect &bb) const {
    if (bb.width < disp_min_w_ || bb.height < disp_min_h_) return false;
    double ar = aspectRatio(bb);
    if (ar < disp_min_ar_ || ar > disp_max_ar_) return false;
    if (disp_only_vertical_ && bb.width > bb.height) return false;
    return true;
  }

  int pickSelectedIndex(const fv_msgs::msg::DetectionArray &arr, const std::string &strategy, int width, int height) const {
    if (arr.detections.empty()) return -1;
    int best = 0; double best_val = -1e9;
    for (size_t i = 0; i < arr.detections.size(); ++i) {
      const auto &d = arr.detections[i];
      int x1 = static_cast<int>(std::floor(d.bbox_min.x));
      int y1 = static_cast<int>(std::floor(d.bbox_min.y));
      int x2 = static_cast<int>(std::ceil(d.bbox_max.x));
      int y2 = static_cast<int>(std::ceil(d.bbox_max.y));
      x1 = std::clamp(x1, 0, std::max(0, width-1));
      y1 = std::clamp(y1, 0, std::max(0, height-1));
      x2 = std::clamp(x2, 0, std::max(0, width-1));
      y2 = std::clamp(y2, 0, std::max(0, height-1));
      cv::Rect bb(cv::Point(x1,y1), cv::Point(x2,y2));
      if (!passDisplayFilter(bb)) continue;
      double val = 0.0;
      if (strategy == "largest_height") {
        val = static_cast<double>(bb.height);
      } else { // highest_conf (default)
        val = static_cast<double>(d.conf_fused);
      }
      if (val > best_val) { best_val = val; best = static_cast<int>(i); }
    }
    return best;
  }

  // params
  std::string image_topic_;
  std::string detections_topic_;
  std::string metrics_topic_;
  std::string camera_info_topic_;
  std::string annotated_topic_;
  bool draw_labels_{true};
  bool draw_confidence_{true};
  bool draw_metrics_{true};
  bool draw_root_tip_{true};
  bool draw_invalid_area_{true};
  bool invalid_enabled_{false};
  bool invalid_any_overlap_{true};
  bool invalid_touch_inclusive_{true};
  double ia_cx_{0.5}, ia_yt_{0.5}, ia_wr_{0.5}, ia_hr_{0.5};
  bool draw_selection_box_{true};
  std::string sel_mode_{"auto"};
  std::string sel_auto_strategy_{"highest_conf"};
  double sel_fix_xc_{0.5};
  int sel_fix_yt_px_{5};
  double sel_fix_wr_{0.20}, sel_fix_hr_{0.50};
  bool disp_only_vertical_{false};
  double disp_min_ar_{0.05}, disp_max_ar_{2.0};
  int disp_min_w_{1}, disp_min_h_{1};

  // subs/pubs
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
  rclcpp::Subscription<fv_msgs::msg::DetectionArray>::SharedPtr sub_det_;
  rclcpp::Subscription<fv_msgs::msg::StalkMetricsArray>::SharedPtr sub_metrics_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_mask_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_cinfo_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_anno_;

  // cache
  fv_msgs::msg::DetectionArray::SharedPtr last_dets_;
  fv_msgs::msg::StalkMetricsArray::SharedPtr last_metrics_;
  sensor_msgs::msg::Image::SharedPtr last_mask_;
  double fx_{0.0}, fy_{0.0}, cx_{0.0}, cy_{0.0};

  // mask params
  std::string mask_topic_;
  int mask_threshold_{128};
  double mask_alpha_{0.3};
  int mask_color_bgr_[3]{0,255,0};
  std::vector<int> id_palette_bgr_;
  std::vector<cv::Vec3b> id_palette_colors_;

  bool draw_title_{true};

  void onMask(const sensor_msgs::msg::Image::SharedPtr msg) {
    // 強参照で保持し、onImage側で確実に利用できるようにする
    last_mask_ = msg;
  }

  void rebuildPaletteColors() {
    id_palette_colors_.clear();
    for (size_t i = 0; i + 2 < id_palette_bgr_.size(); i += 3) {
      int b = std::clamp(id_palette_bgr_[i+0], 0, 255);
      int g = std::clamp(id_palette_bgr_[i+1], 0, 255);
      int r = std::clamp(id_palette_bgr_[i+2], 0, 255);
      id_palette_colors_.emplace_back((uchar)b,(uchar)g,(uchar)r);
    }
    if (id_palette_colors_.empty()) {
      id_palette_colors_.push_back(cv::Vec3b(0,255,0));
    }
  }

  cv::Vec3b colorForDetId(int id) const {
    if (id_palette_colors_.empty()) return cv::Vec3b(0,255,0);
    size_t idx = static_cast<size_t>(std::abs(id)) % id_palette_colors_.size();
    return id_palette_colors_[idx];
  }
};

// 保守: まれな括弧不整合対策として名前空間終端直前に閉じ括弧を明示
// （クラス/関数の括弧は上で閉じられている）

} // namespace fv_aspara_ui_cpp

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fv_aspara_ui_cpp::AsparaUiCppNode>(rclcpp::NodeOptions());
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
