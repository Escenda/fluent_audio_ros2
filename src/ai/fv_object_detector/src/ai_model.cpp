#include "fv_object_detector/ai_model.hpp"
#include "fv_object_detector/yolov10_model.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#include <rclcpp/rclcpp.hpp>

namespace fv_object_detector
{

AIModel::AIModel() : last_infer_time_ms_(0.0)
{
}

/**
 * @brief バウンディングボックス座標を別解像度へスケーリング
 * @param bbox 元のバウンディングボックス
 * @param from_size 元画像サイズ
 * @param to_size 変換後画像サイズ
 * @return スケーリング後のバウンディングボックス
 */
cv::Rect2f AIModel::scaleCoordinates(const cv::Rect2f& bbox, const cv::Size& from_size, const cv::Size& to_size) {
    float x_scale = static_cast<float>(to_size.width) / from_size.width;
    float y_scale = static_cast<float>(to_size.height) / from_size.height;
    return cv::Rect2f(
        bbox.x * x_scale,
        bbox.y * y_scale,
        bbox.width * x_scale,
        bbox.height * y_scale
    );
}

/**
 * @brief 画像をblob形式に変換
 * @param image 入力画像
 * @param width 目標幅
 * @param height 目標高さ
 * @return blob形式の画像
 */
cv::Mat AIModel::blobFromImage(const cv::Mat& image, int width, int height) {
    cv::Mat blob = cv::dnn::blobFromImage(
        image,              // 入力画像
        1.0f / 255.0f,      // スケーリングファクター
        cv::Size(width, height), // モデル入力サイズ
        cv::Scalar(),       // 平均値（デフォルト0）
        true,               // RGB<->BGR変換
        false,              // チャンネルごとに分割しない
        CV_32F              // 32ビット浮動小数点型
    );
    return blob;
}

/**
 * @brief 設定ファイルパスからAIモデルインスタンスを生成するファクトリ関数
 * @param config_path モデル設定ファイルのパス（JSON）
 * @return std::unique_ptr<AIModel> 生成されたAIModel派生クラスのインスタンス
 */
std::unique_ptr<AIModel> AIModel::create(const std::string& config_path) {
    // 設定ファイルを読み込む
    std::ifstream ifs(config_path);
    if (!ifs) {
        throw std::runtime_error("[AIModel::create] 設定ファイルが開けません: " + config_path);
    }
    nlohmann::json config_json;
    ifs >> config_json;
    return createFromConfig(config_json);
}

/**
 * @brief 設定JSONからAIモデルインスタンスを生成するファクトリ関数
 * @param config_json モデル設定JSON
 * @return std::unique_ptr<AIModel> 生成されたAIModel派生クラスのインスタンス
 */
std::unique_ptr<AIModel> AIModel::createFromConfig(const nlohmann::json& config_json) {
    const auto& model_cfg = config_json["model"];
    std::string type = model_cfg["type"];
    
    if (type == "yolov10") {
        auto model = std::make_unique<YoloV10Model>();
        model->setCommonConfig(model_cfg, config_json);
        model->initialize(model_cfg);
        return model;
    }
    
    throw std::runtime_error("Unknown model type: " + type);
}

/**
 * @brief 共通設定を設定
 * @param model_cfg モデル設定
 * @param config_json 設定JSON
 */
void AIModel::setCommonConfig(const nlohmann::json& model_cfg, const nlohmann::json& config_json) {
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] setCommonConfig");
    name_ = model_cfg["name"].get<std::string>();
    device_ = model_cfg["device"].get<std::string>();
    input_width_ = model_cfg["input_width"].get<int>();
    input_height_ = model_cfg["input_height"].get<int>();
    model_path_ = model_cfg["path"].get<std::string>();
    config_ = config_json;

    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] モデル名: %s", name_.c_str());
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] デバイス: %s", device_.c_str());
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] 入力サイズ: %dx%d", input_width_, input_height_);
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] モデルパス: %s", model_path_.c_str());
    
    // モデルパスの存在チェックとフォールバック
    try {
        namespace fs = std::filesystem;
        fs::path p(model_path_);
        if (!p.is_absolute()) {
            // 相対パスなら /models 配下を優先
            fs::path candidate = fs::path("/models") / p;
            if (fs::exists(candidate)) {
                model_path_ = candidate.string();
                RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] 相対パスを /models に解決: %s", model_path_.c_str());
            }
        }
        // それでも存在しなければ、ファイル名のみから /models を試行
        if (!fs::exists(model_path_)) {
            fs::path base = fs::path(model_path_).filename();
            // /models 直下
            fs::path candidate1 = fs::path("/models") / base;
            // 一般的なサブディレクトリ候補
            fs::path candidate2 = fs::path("/models/yolov10") / base;
            fs::path candidate3 = fs::path("/models/openvino") / base;
            if (fs::exists(candidate1)) { model_path_ = candidate1.string(); }
            else if (fs::exists(candidate2)) { model_path_ = candidate2.string(); }
            else if (fs::exists(candidate3)) { model_path_ = candidate3.string(); }
            if (fs::exists(model_path_)) {
                RCLCPP_WARN(rclcpp::get_logger("AIModel"), "[AIModel] モデルが見つからないため候補を使用: %s", model_path_.c_str());
            }
        }
        // なお存在しない場合は既定モデルにフォールバック
        if (!fs::exists(model_path_)) {
            // 複数候補を順に試す
            std::vector<fs::path> fallbacks = {
                fs::path("/models/yolov10/v2_nano_best_fp16_dynamic.xml"),
                fs::path("/models/yolov10/yolov10n_aspara_v2.1.xml"),
                fs::path("/models/yolov10/yolov10m_aspara_v2.0.xml"),
                fs::path("/models/openvino/aspara_v2.0_yolo10s.xml")
            };
            for (const auto& fb : fallbacks) {
                if (fs::exists(fb)) { model_path_ = fb.string(); break; }
            }
            if (fs::exists(model_path_)) {
                RCLCPP_WARN(rclcpp::get_logger("AIModel"), "[AIModel] 指定モデル未検出のため既定候補にフォールバック: %s", model_path_.c_str());
            } else {
                RCLCPP_WARN(rclcpp::get_logger("AIModel"), "[AIModel] モデルファイルが見つかりませんでした。後続の初期化で失敗する可能性があります: %s", model_path_.c_str());
            }
        }
    } catch (const std::exception& e) {
        RCLCPP_WARN(rclcpp::get_logger("AIModel"), "[AIModel] モデルパス検証中に例外: %s", e.what());
    }
    
    // クラス名リストを読み込む
    if (config_json.contains("classes") && config_json["classes"].is_array()) {
        class_names_.clear();
        for (const auto& c : config_json["classes"]) {
            class_names_.push_back(c.get<std::string>());
        }
        
        std::string class_list = "[AIModel] クラス名リスト: ";
        for (size_t i = 0; i < class_names_.size(); ++i) {
            class_list += "[" + std::to_string(i) + "]" + class_names_[i] + ", ";
        }
        RCLCPP_INFO(rclcpp::get_logger("AIModel"), "%s", class_list.c_str());
    }
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] クラス名リスト: %zu", class_names_.size());
}

/**
 * @brief 設定ファイルを読み込む
 * @param path 設定ファイルパス
 */
void AIModel::loadConfig(const std::string& path) {
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] loadConfig");
    std::ifstream ifs(path);
    if (!ifs) {
        RCLCPP_ERROR(rclcpp::get_logger("AIModel"), "設定ファイルが開けません: %s", path.c_str());
        return;
    }
    RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] 設定ファイル読み込み開始: %s", path.c_str());
    try {
        nlohmann::json config_json;
        ifs >> config_json;
        config_ = config_json;
        RCLCPP_INFO(rclcpp::get_logger("AIModel"), "[AIModel] 設定ファイル読み込み成功: %s", path.c_str());
    } catch (const std::exception& e) {
        RCLCPP_ERROR(rclcpp::get_logger("AIModel"), "設定ファイル読み込みエラー: %s", e.what());
    }
}

/**
 * @brief クラス名を取得
 * @param class_id クラスID
 * @return クラス名
 */
std::string AIModel::getClassName(int class_id) const {
    if (class_id >= 0 && class_id < (int)class_names_.size()) {
        return class_names_[class_id];
    }
    return std::to_string(class_id);
}

/**
 * @brief モデルの入出力情報を表示
 * @param compiled_model コンパイルされたモデル
 */
void AIModel::printModelIOInfo(const ov::CompiledModel& compiled_model) {
    // 入力の形状・型・名前を取得
    auto& inputs = compiled_model.inputs();
    for (const auto& input : inputs) {
        auto partial_shape = input.get_partial_shape();
        auto elem_type = input.get_element_type();
        auto names = input.get_names();
        std::string name = names.empty() ? "(no name)" : *names.begin();
        RCLCPP_DEBUG(rclcpp::get_logger("AIModel"), "[ModelInfo] Input name: %s, partial shape: %s, type: %s", 
                     name.c_str(), partial_shape.to_string().c_str(), elem_type.to_string().c_str());
    }
    
    // 出力の形状・型・名前を取得
    auto& outputs = compiled_model.outputs();
    for (const auto& output : outputs) {
        auto partial_shape = output.get_partial_shape();
        auto elem_type = output.get_element_type();
        auto names = output.get_names();
        std::string name = names.empty() ? "(no name)" : *names.begin();
        RCLCPP_DEBUG(rclcpp::get_logger("AIModel"), "[ModelInfo] Output name: %s, partial shape: %s, type: %s", 
                     name.c_str(), partial_shape.to_string().c_str(), elem_type.to_string().c_str());
    }
}

} // namespace fv_object_detector 
