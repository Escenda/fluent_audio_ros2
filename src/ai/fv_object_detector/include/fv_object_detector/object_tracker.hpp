#ifndef FV_OBJECT_DETECTOR_OBJECT_TRACKER_HPP_
#define FV_OBJECT_DETECTOR_OBJECT_TRACKER_HPP_

#include "fv_object_detector/detection_data.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <set>
#include <string>
#include <deque>

namespace fv_object_detector
{

/**
 * @brief オブジェクト追跡クラス
 *
 * 検出されたオブジェクトに一意のIDを割り当て、フレーム間で追跡する機能を提供します。
 * シンプルで再利用しやすい設計です。
 */
class ObjectTracker {
public:
    /**
     * @brief コンストラクタ
     */
    ObjectTracker();

    /**
     * @brief 設定パラメータ
     */
    struct Params {
        float max_distance_px = 60.0f;   // 中心距離によるゲート
        float iou_threshold   = 0.3f;    // IoUによるゲート
        int   max_age         = 5;       // 未検出で保持する最大フレーム数
        int   min_hits        = 1;       // 新規トラックを確定するまでの最小ヒット数
        bool  require_same_class = true; // クラスIDの一致を要求
        int   hold_frames     = 2;       // 未検出でも描画/出力で保持するフレーム数
        float smooth_alpha    = 0.6f;    // bbox/scoreの平滑化係数（0~1、1で即時追従）
        int   median_window   = 0;       // 中値フィルタ窓（>1で有効）
    };

    /**
     * @brief パラメータ設定
     */
    void setParams(const Params& p) { params_ = p; }

    /**
     * @brief 検出されたオブジェクトにIDを割り当てる
     *
     * 前フレームのオブジェクトと現在フレームのオブジェクトを対応付け、
     * 一致するオブジェクトには同じIDを割り当て、新しいオブジェクトには
     * 新しいIDを割り当てます。
     *
     * @param detections 現在フレームの検出結果
     */
    void assignObjectIds(std::vector<DetectionData>& detections);

    /**
     * @brief 現在の検出結果を取得する
     *
     * @return const std::vector<DetectionData>& 現在の検出結果
     */
    const std::vector<DetectionData>& getCurrentDetections() const;

    /**
     * @brief クラスIDごとの検出数を取得する
     *
     * @param class_id クラスID
     * @return int 指定したクラスIDの検出数
     */
    int getDetectionCountByClass(int class_id) const;

    /**
     * @brief 全検出数を取得する
     *
     * @return int 全検出数
     */
    int getTotalDetectionCount() const;

    /**
     * @brief トラッカーをリセットする
     */
    void reset();

    /**
     * @brief 安定化済みの検出リストを取得（未検出でもhold_frames内は維持）
     */
    std::vector<DetectionData> getStabilizedDetections() const;

private:
    struct Track {
        int id;
        cv::Rect2f bbox;
        int class_id;
        std::string class_name;
        int hits = 0;              // 検出ヒット回数
        int age = 0;               // 生存フレーム数（総）
        int time_since_update = 0; // 最終更新からのフレーム数
        float confidence = 0.0f;   // 直近信頼度（平滑化）
        std::deque<cv::Rect2f> bbox_hist; // 直近バウンディングボックス履歴
        std::deque<float> conf_hist;       // 直近スコア履歴
    };

    Params params_{};
    int next_object_id_;                           ///< 次に割り当てるオブジェクトID
    int selected_object_id_;                       ///< 選択中のオブジェクトID
    std::vector<DetectionData> current_detections_; ///< 現在の検出結果
    std::map<int, Track> tracks_;                   ///< 既存トラック

    /**
     * @brief 2つのオブジェクトが同じオブジェクトかどうかを判定
     *
     * @param det1 検出結果1
     * @param det2 検出結果2
     * @param threshold 距離閾値
     * @return true 同じオブジェクトの場合
     */
    bool isSameObject(const DetectionData& det1, const DetectionData& det2, float threshold = 50.0f) const;

    /**
     * @brief 2点間の距離を計算
     *
     * @param p1 点1
     * @param p2 点2
     * @return float 距離
     */
    float calculateDistance(const cv::Point2f& p1, const cv::Point2f& p2) const;

    // IoU計算
    float calculateIoU(const cv::Rect2f& a, const cv::Rect2f& b) const;

    // ヘルパー: 直近履歴から中央値矩形を算出
    cv::Rect2f medianRect(const std::deque<cv::Rect2f>& hist, int window) const;
    float medianValue(const std::deque<float>& hist, int window) const;
};

} // namespace fv_object_detector

#endif // FV_OBJECT_DETECTOR_OBJECT_TRACKER_HPP_ 
