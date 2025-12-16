#pragma once

#include "fv_instance_seg/backends/inferencer.hpp"

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/infer_request.hpp>

#include <algorithm>
#include <vector>

namespace fv_instance_seg {

class OvYoloSegInferencer : public Inferencer {
 public:
  OvYoloSegInferencer();
  ~OvYoloSegInferencer() override;

  bool load(const std::string& model_path, const std::string& device) override;
  bool infer(const cv::Mat& bgr, float conf_thres, float iou_thres, InferResult* out) override;
  void configure(bool nms_class_agnostic, int max_detections, bool debug_shapes) override;
  void set_timeout_ms(int timeout_ms) override;

 private:
  struct LetterboxInfo {
    float scale;
    int pad_w;
    int pad_h;
  };

  LetterboxInfo letterbox(const cv::Mat& src, cv::Mat& dst) const;
  void nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float iou_th, std::vector<int>& keep) const;

  ov::Core core_;
  ov::Output<ov::Node> input_port_;
  ov::Output<ov::Node> det_port_;
  ov::Output<ov::Node> proto_port_;
  ov::CompiledModel compiled_;
  ov::InferRequest request_;
  bool has_request_ = false;
  int timeout_ms_ = 0;
  int net_w_ = 0;
  int net_h_ = 0;
  int proto_c_ = 0;
  int proto_h_ = 0;
  int proto_w_ = 0;
  int num_coeff_ = 0;
  int num_classes_ = 1;
  bool nms_agnostic_ = true;
  int max_det_ = 100;
  bool debug_ = false;
};

}  // namespace fv_instance_seg
