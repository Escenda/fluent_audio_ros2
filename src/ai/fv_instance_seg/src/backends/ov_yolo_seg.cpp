#include "fv_instance_seg/backends/ov_yolo_seg.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <chrono>
#include <numeric>
#include <unordered_map>

namespace fv_instance_seg {

OvYoloSegInferencer::OvYoloSegInferencer() {}
OvYoloSegInferencer::~OvYoloSegInferencer() {}

void OvYoloSegInferencer::set_timeout_ms(int timeout_ms) {
  timeout_ms_ = std::max(0, timeout_ms);
}

bool OvYoloSegInferencer::load(const std::string& model_path, const std::string& device) {
  try {
    auto model = core_.read_model(model_path);
    input_port_ = model->input();
    auto ishape = input_port_.get_shape();
    if (ishape.size() != 4) return false;
    net_h_ = static_cast<int>(ishape[2]);
    net_w_ = static_cast<int>(ishape[3]);

    auto outs = model->outputs();
    if (outs.size() < 2) return false;
    det_port_ = outs[0];
    proto_port_ = outs[1];
    auto dshape = det_port_.get_shape();
    auto pshape = proto_port_.get_shape();
    if (dshape.size() != 3 || pshape.size() != 4) return false;
    proto_c_ = static_cast<int>(pshape[1]);
    proto_h_ = static_cast<int>(pshape[2]);
    proto_w_ = static_cast<int>(pshape[3]);
    num_coeff_ = proto_c_;
    int det_c = static_cast<int>(dshape[1]);
    num_classes_ = std::max(1, det_c - 4 - num_coeff_);

    compiled_ = core_.compile_model(model, device);
    request_ = compiled_.create_infer_request();
    has_request_ = true;
  } catch (...) {
    return false;
  }
  return true;
}

OvYoloSegInferencer::LetterboxInfo OvYoloSegInferencer::letterbox(const cv::Mat& src, cv::Mat& dst) const {
  LetterboxInfo info{1.f,0,0};
  if (src.empty() || net_w_<=0 || net_h_<=0){ dst=src.clone(); return info; }
  float r = std::min(net_w_/static_cast<float>(src.cols), net_h_/static_cast<float>(src.rows));
  int nw = static_cast<int>(std::round(src.cols*r));
  int nh = static_cast<int>(std::round(src.rows*r));
  cv::Mat rs; cv::resize(src, rs, cv::Size(nw,nh));
  int pw = (net_w_-nw)/2; int ph=(net_h_-nh)/2;
  cv::copyMakeBorder(rs, dst, ph, net_h_-nh-ph, pw, net_w_-nw-pw, cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
  info.scale=r; info.pad_w=pw; info.pad_h=ph; return info;
}

void OvYoloSegInferencer::nms(const std::vector<cv::Rect2f>& boxes, const std::vector<float>& scores, float iou_th, std::vector<int>& keep) const {
  keep.clear(); std::vector<int> idx(boxes.size()); std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(), [&](int a,int b){return scores[a]>scores[b];});
  std::vector<char> sup(boxes.size(),0);
  for(size_t _i=0; _i<idx.size(); ++_i){int i=idx[_i]; if(sup[i]) continue; keep.push_back(i);
    for(size_t _j=_i+1; _j<idx.size(); ++_j){int j=idx[_j]; if(sup[j]) continue;
      float xx1=std::max(boxes[i].x,boxes[j].x), yy1=std::max(boxes[i].y,boxes[j].y);
      float xx2=std::min(boxes[i].x+boxes[i].width,boxes[j].x+boxes[j].width);
      float yy2=std::min(boxes[i].y+boxes[i].height,boxes[j].y+boxes[j].height);
      float w=std::max(0.f,xx2-xx1), h=std::max(0.f,yy2-yy1);
      float inter=w*h; float ai=boxes[i].width*boxes[i].height; float aj=boxes[j].width*boxes[j].height;
      float ovr=inter/(ai+aj-inter+1e-6f); if(ovr>iou_th) sup[j]=1; }
  }
}

void OvYoloSegInferencer::configure(bool nms_class_agnostic, int max_detections, bool debug_shapes){
  nms_agnostic_=nms_class_agnostic; max_det_=std::max(1,max_detections); debug_=debug_shapes;
}

bool OvYoloSegInferencer::infer(const cv::Mat& bgr, float conf_thres, float iou_thres, InferResult* out){
  if(!out || bgr.empty()) return false; 
  out->boxes.clear(); out->classes.clear(); out->scores.clear(); out->masks.clear(); out->mask_proto_size=cv::Size(proto_w_, proto_h_);
  cv::Mat resized; auto info=letterbox(bgr,resized);
  cv::Mat rgb; cv::cvtColor(resized,rgb,cv::COLOR_BGR2RGB);
  cv::Mat f32; rgb.convertTo(f32, CV_32F, 1.0/255.0);

  auto ishape=input_port_.get_shape();
  ov::Tensor in(input_port_.get_element_type(), ov::Shape(ishape.begin(), ishape.end()));
  std::vector<cv::Mat> ch(3); for(int i=0;i<3;++i) ch[i]=cv::Mat(net_h_,net_w_,CV_32F, in.data<float>()+ i*net_h_*net_w_);
  std::vector<cv::Mat> sp; cv::split(f32, sp); for(int i=0;i<3;++i) sp[i].copyTo(ch[i]);

  try {
    if (!has_request_) {
      request_ = compiled_.create_infer_request();
      has_request_ = true;
    }

    request_.set_tensor(input_port_, in);

    if (timeout_ms_ > 0) {
      request_.start_async();
      bool ready = request_.wait_for(std::chrono::milliseconds(timeout_ms_));
      if (!ready) {
        try {
          request_.cancel();
        } catch (...) {
        }
        has_request_ = false;
        return false;
      }
      // Propagate async exceptions (if any)
      request_.wait();
    } else {
      request_.infer();
    }
  } catch (...) {
    has_request_ = false;
    return false;
  }

  ov::Tensor det_t=request_.get_tensor(det_port_), proto_t=request_.get_tensor(proto_port_);
  const float* det=det_t.data<const float>(); const float* proto=proto_t.data<const float>();
  auto dshape=det_port_.get_shape(); int C=static_cast<int>(dshape[1]); int N=static_cast<int>(dshape[2]); (void)C;
  int P=proto_c_, Hm=proto_h_, Wm=proto_w_; int ncls=num_classes_, ncoef=num_coeff_;

  std::vector<cv::Rect2f> boxes; std::vector<float> scores; std::vector<int> classes; std::vector<std::vector<float>> coeffs;
  boxes.reserve(128); scores.reserve(128); classes.reserve(128); coeffs.reserve(128);

  for(int i=0;i<N;++i){
    float x=det[0*N+i], y=det[1*N+i], w=det[2*N+i], h=det[3*N+i];
    int cb=0; float cs=0.f; for(int c=0;c<ncls;++c){ float s=det[(4+c)*N+i]; if(s>cs){cs=s; cb=c;} }
    if(cs<conf_thres) continue;
    float x1=x-w*0.5f, y1=y-h*0.5f, x2=x+w*0.5f, y2=y+h*0.5f;
    x1-=info.pad_w; x2-=info.pad_w; y1-=info.pad_h; y2-=info.pad_h;
    if(info.scale>0){ x1/=info.scale; x2/=info.scale; y1/=info.scale; y2/=info.scale; }
    x1=std::clamp(x1,0.f,(float)bgr.cols-1); y1=std::clamp(y1,0.f,(float)bgr.rows-1);
    x2=std::clamp(x2,0.f,(float)bgr.cols-1); y2=std::clamp(y2,0.f,(float)bgr.rows-1);
    cv::Rect2f r(x1,y1,std::max(0.f,x2-x1), std::max(0.f,y2-y1)); if(r.width<=1||r.height<=1) continue;
    boxes.push_back(r); scores.push_back(cs); classes.push_back(cb);
    std::vector<float> cv(ncoef); for(int k=0;k<ncoef;++k) cv[k]=det[(4+ncls+k)*N+i]; coeffs.emplace_back(std::move(cv));
  }

  std::vector<int> keep;
  if(nms_agnostic_) nms(boxes, scores, iou_thres, keep);
  else {
    std::unordered_map<int,std::vector<int>> byc; for(size_t i=0;i<classes.size();++i) byc[classes[i]].push_back((int)i);
    for(auto& kv: byc){ std::vector<cv::Rect2f> b2; std::vector<float> s2; for(int idx: kv.second){ b2.push_back(boxes[idx]); s2.push_back(scores[idx]); }
      std::vector<int> k2; nms(b2,s2,iou_thres,k2); for(int ki: k2) keep.push_back(kv.second[ki]); }
  }
  if((int)keep.size()>max_det_){ std::sort(keep.begin(), keep.end(), [&](int a,int b){ return scores[a]>scores[b];}); keep.resize(max_det_); }

  std::vector<cv::Mat> proto_planes(P);
  for(int p=0;p<P;++p){ const float* pp= proto + p*Hm*Wm; proto_planes[p]=cv::Mat(Hm,Wm,CV_32F, const_cast<float*>(pp)); }

  for(int idx: keep){ const auto& r=boxes[idx]; const auto& cvf=coeffs[idx];
    cv::Mat ms(Hm,Wm,CV_32F, cv::Scalar(0)); for(int p=0;p<P;++p) ms += proto_planes[p] * cvf[p];
    cv::Mat sig; cv::exp(-ms, sig); sig = 1.0 / (1.0 + sig);
    cv::Mat mnet; cv::resize(sig, mnet, cv::Size(net_w_,net_h_),0,0,cv::INTER_LINEAR);
    cv::Rect roi(info.pad_w, info.pad_h, net_w_-2*info.pad_w, net_h_-2*info.pad_h); roi &= cv::Rect(0,0,net_w_,net_h_);
    cv::Mat munp = (roi.width>0 && roi.height>0)? mnet(roi).clone(): mnet;

    cv::Mat mb= cv::Mat::zeros(bgr.rows,bgr.cols,CV_8UC1);
    cv::Rect ri((int)r.x,(int)r.y,(int)r.width,(int)r.height); ri &= cv::Rect(0,0,bgr.cols,bgr.rows);
    if (ri.width > 0 && ri.height > 0) {
      // 元画像座標(ri) -> munp(レターボックスのパディング除去済み) 座標へ写像して、
      // 必要なROIだけをリサイズする（全画面へのリサイズは高コストでフリーズ要因になり得る）。
      const float s = (info.scale > 0.f) ? info.scale : 1.f;
      int x1u = static_cast<int>(std::floor(r.x * s));
      int y1u = static_cast<int>(std::floor(r.y * s));
      int x2u = static_cast<int>(std::ceil((r.x + r.width) * s));
      int y2u = static_cast<int>(std::ceil((r.y + r.height) * s));
      cv::Rect ri_u(x1u, y1u, std::max(0, x2u - x1u), std::max(0, y2u - y1u));
      ri_u &= cv::Rect(0, 0, munp.cols, munp.rows);
      if (ri_u.width > 0 && ri_u.height > 0) {
        cv::Mat roi_unp = munp(ri_u);
        cv::Mat roi_rs;
        cv::resize(roi_unp, roi_rs, ri.size(), 0, 0, cv::INTER_LINEAR);
        cv::Mat m8;
        roi_rs.convertTo(m8, CV_8UC1, 255.0);
        cv::threshold(m8, m8, 128, 255, cv::THRESH_BINARY);
        m8.copyTo(mb(ri));
      }
    }
    out->boxes.emplace_back((int)r.x,(int)r.y,(int)r.width,(int)r.height);
    out->classes.emplace_back(classes[idx]); out->scores.emplace_back(scores[idx]); out->masks.emplace_back(std::move(mb));
  }
  return true;
}

} // namespace fv_instance_seg
