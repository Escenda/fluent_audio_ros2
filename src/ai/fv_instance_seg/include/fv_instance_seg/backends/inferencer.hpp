#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace fv_instance_seg {

struct InferResult {
  std::vector<cv::Rect> boxes;
  std::vector<int> classes;
  std::vector<float> scores;
  std::vector<cv::Mat> masks;
  cv::Size mask_proto_size;
};

class Inferencer {
 public:
  virtual ~Inferencer() = default;
  virtual bool load(const std::string& model_path, const std::string& device) = 0;
  virtual bool infer(const cv::Mat& bgr, float conf_thres, float iou_thres, InferResult* out) = 0;
  virtual void configure(bool nms_class_agnostic, int max_detections, bool debug_shapes) = 0;
};

std::unique_ptr<Inferencer> CreateInferencer(const std::string& backend);

}  // namespace fv_instance_seg

