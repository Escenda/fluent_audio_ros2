/**
 * @file object_tracker.cpp
 * @brief オブジェクトトラッカーの実装ファイル
 * @details フレーム間でのオブジェクトID追跡機能を提供
 * @author Takashi Otsuka
 * @date 2024
 * @version 1.0
 */

#include "fv_object_detector/object_tracker.hpp"
#include <algorithm>
#include <limits>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace fv_object_detector
{

/**
 * @brief コンストラクタ
 * @details オブジェクトトラッカーの初期化
 */
ObjectTracker::ObjectTracker() : next_object_id_(0), selected_object_id_(-1) {}

/**
 * @brief オブジェクトIDの割り当て
 * @param detections 検出結果の配列
 * @details 前フレームのオブジェクトと現在フレームのオブジェクトを対応付け、一意のIDを割り当て
 * 
 * 処理内容：
 * - 前フレームのオブジェクトとの距離計算
 * - 最も近いオブジェクトとのマッチング
 * - 新しいオブジェクトへのID割り当て
 * - 現在の検出結果の更新
 */
void ObjectTracker::assignObjectIds(std::vector<DetectionData> &detections) {
  // 初期化: 全検出を未割り当てに
  for (auto &det : detections) {
    det.object_id = -1;
  }

  // 既存トラックの更新前インクリメント（未更新時間, age）
  for (auto &kv : tracks_) {
    kv.second.age += 1;
    kv.second.time_since_update += 1;
  }

  // 検出とトラックのマッチング（単純貪欲: IoU優先, 次に距離）
  std::vector<bool> det_assigned(detections.size(), false);
  for (auto &kv : tracks_) {
    auto &trk = kv.second;

    float best_iou = -1.0f;
    float best_dist = std::numeric_limits<float>::max();
    int best_idx = -1;

    for (size_t i = 0; i < detections.size(); ++i) {
      if (det_assigned[i]) continue;

      const auto &det = detections[i];
      if (params_.require_same_class && det.class_id != trk.class_id) {
        continue;
      }

      float iou = calculateIoU(trk.bbox, det.bbox);
      cv::Point2f c1(trk.bbox.x + trk.bbox.width * 0.5f, trk.bbox.y + trk.bbox.height * 0.5f);
      cv::Point2f c2(det.bbox.x + det.bbox.width * 0.5f, det.bbox.y + det.bbox.height * 0.5f);
      float dist = calculateDistance(c1, c2);

      bool pass_gate = (iou >= params_.iou_threshold) || (dist <= params_.max_distance_px);
      if (!pass_gate) continue;

      // IoU優先、同IoUなら距離が近い方
      if (iou > best_iou || (std::abs(iou - best_iou) < 1e-6 && dist < best_dist)) {
        best_iou = iou;
        best_dist = dist;
        best_idx = static_cast<int>(i);
      }
    }

    if (best_idx >= 0) {
      // マッチ: IDを引き継ぎ、トラック更新
      detections[best_idx].object_id = trk.id;
      det_assigned[best_idx] = true;
      // 平滑化更新
      const auto &nb = detections[best_idx].bbox;
      float a = std::clamp(params_.smooth_alpha, 0.0f, 1.0f);
      trk.bbox.x = a * nb.x + (1.0f - a) * trk.bbox.x;
      trk.bbox.y = a * nb.y + (1.0f - a) * trk.bbox.y;
      trk.bbox.width  = a * nb.width  + (1.0f - a) * trk.bbox.width;
      trk.bbox.height = a * nb.height + (1.0f - a) * trk.bbox.height;
      trk.confidence = a * detections[best_idx].confidence + (1.0f - a) * trk.confidence;
      trk.class_id = detections[best_idx].class_id;
      trk.class_name = detections[best_idx].class_name;
      // 履歴更新（中央値フィルタ用）
      if (params_.median_window > 1) {
        trk.bbox_hist.push_back(nb);
        if ((int)trk.bbox_hist.size() > std::max(params_.median_window, 3)) trk.bbox_hist.pop_front();
        trk.conf_hist.push_back(detections[best_idx].confidence);
        if ((int)trk.conf_hist.size() > std::max(params_.median_window, 3)) trk.conf_hist.pop_front();
      }
      trk.hits += 1;
      trk.time_since_update = 0;
    }
  }

  // 未割り当ての検出は新規トラックとして登録（min_hitsは公開側で利用可）
  for (size_t i = 0; i < detections.size(); ++i) {
    if (det_assigned[i]) continue;
    auto &det = detections[i];
    int new_id = next_object_id_++;
    det.object_id = new_id;
    Track t{new_id, det.bbox, det.class_id, det.class_name, 1, 1, 0, det.confidence};
    if (params_.median_window > 1) {
      t.bbox_hist.push_back(det.bbox);
      t.conf_hist.push_back(det.confidence);
    }
    tracks_.emplace(new_id, t);
  }

  // マッチしなかったトラックは年齢に応じて破棄
  std::vector<int> to_erase;
  for (auto &kv : tracks_) {
    auto &trk = kv.second;
    if (trk.time_since_update > params_.max_age) {
      to_erase.push_back(kv.first);
    }
  }
  for (int id : to_erase) {
    tracks_.erase(id);
  }

  // 現在の検出結果を更新
  current_detections_ = detections;
}

/**
 * @brief 現在の検出結果を取得
 * @return const std::vector<DetectionData>& 現在の検出結果の配列
 * @details 最新の検出結果とオブジェクトIDを取得
 */
const std::vector<DetectionData> &
ObjectTracker::getCurrentDetections() const {
  return current_detections_;
}

/**
 * @brief クラスIDごとの検出数を取得
 * @param class_id クラスID
 * @return int 指定したクラスIDの検出数
 * @details 現在の検出結果から特定クラスのオブジェクト数をカウント
 */
int ObjectTracker::getDetectionCountByClass(int class_id) const {
  return std::count_if(current_detections_.begin(), current_detections_.end(),
                       [class_id](const DetectionData &det) {
                         return det.class_id == class_id;
                       });
}

/**
 * @brief 全検出数を取得
 * @return int 全検出数
 * @details 現在の検出結果の総数を取得
 */
int ObjectTracker::getTotalDetectionCount() const {
  return current_detections_.size();
}

/**
 * @brief トラッカーをリセット
 * @details オブジェクトIDカウンタと検出結果を初期化
 * 
 * リセット内容：
 * - 次に割り当てるオブジェクトIDを0にリセット
 * - 現在の検出結果をクリア
 * - 選択中のオブジェクトIDを-1にリセット
 */
void ObjectTracker::reset() {
  next_object_id_ = 0;
  current_detections_.clear();
  selected_object_id_ = -1;
  tracks_.clear();
}

std::vector<DetectionData> ObjectTracker::getStabilizedDetections() const {
  std::vector<DetectionData> out;
  out.reserve(tracks_.size());
  for (const auto &kv : tracks_) {
    const auto &t = kv.second;
    // 出力条件: 未更新でもhold_frames以内 or 現在更新済み（time_since_update==0）
    if (t.time_since_update <= params_.hold_frames && t.hits >= params_.min_hits) {
      DetectionData d;
      // 中央値フィルタ
      if (params_.median_window > 1 && t.bbox_hist.size() >= 2) {
        d.bbox = medianRect(t.bbox_hist, params_.median_window);
      } else {
        d.bbox = t.bbox;
      }
      d.confidence = t.confidence;
      d.class_id = t.class_id;
      d.object_id = t.id;
      d.class_name = t.class_name;
      d.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch()).count() / 1000.0;
      out.push_back(std::move(d));
    }
  }
  return out;
}

float ObjectTracker::calculateIoU(const cv::Rect2f &a, const cv::Rect2f &b) const {
  float x1 = std::max(a.x, b.x);
  float y1 = std::max(a.y, b.y);
  float x2 = std::min(a.x + a.width, b.x + b.width);
  float y2 = std::min(a.y + a.height, b.y + b.height);
  if (x2 <= x1 || y2 <= y1) return 0.0f;
  float inter = (x2 - x1) * (y2 - y1);
  float ua = a.width * a.height + b.width * b.height - inter;
  if (ua <= 0.0f) return 0.0f;
  return inter / ua;
}

float ObjectTracker::calculateDistance(const cv::Point2f &p1, const cv::Point2f &p2) const {
  return cv::norm(p1 - p2);
}

cv::Rect2f ObjectTracker::medianRect(const std::deque<cv::Rect2f> &hist, int window) const {
  if (hist.empty()) return cv::Rect2f();
  int w = std::min(window, (int)hist.size());
  std::vector<float> vx, vy, vw, vh;
  vx.reserve(w); vy.reserve(w); vw.reserve(w); vh.reserve(w);
  for (int i = (int)hist.size() - w; i < (int)hist.size(); ++i) {
    vx.push_back(hist[i].x);
    vy.push_back(hist[i].y);
    vw.push_back(hist[i].width);
    vh.push_back(hist[i].height);
  }
  auto med = [](std::vector<float>& a){ std::nth_element(a.begin(), a.begin()+a.size()/2, a.end()); return a[a.size()/2]; };
  return cv::Rect2f(med(vx), med(vy), med(vw), med(vh));
}

float ObjectTracker::medianValue(const std::deque<float> &hist, int window) const {
  if (hist.empty()) return 0.0f;
  int w = std::min(window, (int)hist.size());
  std::vector<float> v; v.reserve(w);
  for (int i = (int)hist.size() - w; i < (int)hist.size(); ++i) v.push_back(hist[i]);
  std::nth_element(v.begin(), v.begin()+v.size()/2, v.end());
  return v[v.size()/2];
}

} // namespace fv_object_detector 
