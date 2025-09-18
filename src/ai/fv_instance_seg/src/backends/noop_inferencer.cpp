#include "fv_instance_seg/backends/inferencer.hpp"
#include "fv_instance_seg/backends/ov_yolo_seg.hpp"

#include <algorithm>
#include <cctype>
#include <memory>
#include <string>

namespace fv_instance_seg {

namespace {
class NoopInferencer : public Inferencer {
 public:
  bool load(const std::string&, const std::string&) override { return true; }
  bool infer(const cv::Mat&, float, float, InferResult* out) override {
    if (!out) {
      return false;
    }
    out->boxes.clear();
    out->classes.clear();
    out->scores.clear();
    out->masks.clear();
    out->mask_proto_size = cv::Size();
    return true;
  }
  void configure(bool, int, bool) override {}
};
}  // namespace

std::unique_ptr<Inferencer> CreateInferencer(const std::string& backend) {
  std::string key = backend;
  std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });

  if (key.empty() || key == "openvino" || key == "ov") {
    return std::make_unique<OvYoloSegInferencer>();
  }

  return std::make_unique<NoopInferencer>();
}

}  // namespace fv_instance_seg
