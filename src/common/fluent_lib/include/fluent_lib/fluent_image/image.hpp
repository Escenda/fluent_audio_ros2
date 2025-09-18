#pragma once

// 互換レイヤ: fluent_image::Image
// - sensor_msgs::msg::Image / cv::Mat 相互変換
// - to_bgr8 / to_depth32f 等の基本変換

#include <opencv2/opencv.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/header.hpp>
#include <string>

namespace fluent_image {

class Image {
public:
    Image() = default;
    explicit Image(const cv::Mat &m) : mat_(m) {}
    explicit Image(const sensor_msgs::msg::Image &msg) { from_msg(msg); }

    // 暗黙変換（使い勝手重視）
    operator cv::Mat&() { return mat_; }
    operator const cv::Mat&() const { return mat_; }

    // 基本アクセス
    cv::Mat& mat() { return mat_; }
    const cv::Mat& mat() const { return mat_; }

    // 型変換
    Image to_bgr8() const {
        if (mat_.empty()) return Image();
        if (mat_.type() == CV_8UC3) return *this; // 既にBGR8
        cv::Mat out;
        if (mat_.type() == CV_8UC1) {
            cv::cvtColor(mat_, out, cv::COLOR_GRAY2BGR);
        } else if (mat_.type() == CV_16UC1) {
            cv::Mat tmp8; mat_.convertTo(tmp8, CV_8U, 1.0/256.0);
            cv::cvtColor(tmp8, out, cv::COLOR_GRAY2BGR);
        } else if (mat_.type() == CV_32FC1) {
            cv::Mat tmp8; mat_.convertTo(tmp8, CV_8U, 255.0);
            cv::cvtColor(tmp8, out, cv::COLOR_GRAY2BGR);
        } else {
            out = mat_.clone();
        }
        return Image(out);
    }

    Image to_depth32f(float unit_m = 1.0f) const {
        if (mat_.empty()) return Image();
        if (mat_.type() == CV_32FC1) return *this;
        cv::Mat out;
        if (mat_.type() == CV_16UC1) {
            mat_.convertTo(out, CV_32F, unit_m);
        } else if (mat_.type() == CV_8UC1) {
            mat_.convertTo(out, CV_32F, unit_m);
        } else {
            out = mat_.clone();
        }
        return Image(out);
    }

    // ROS メッセージ変換
    void from_msg(const sensor_msgs::msg::Image &msg) {
        const auto &enc = msg.encoding;
        if (enc == "bgr8") {
            mat_ = cv::Mat(msg.height, msg.width, CV_8UC3, const_cast<unsigned char*>(msg.data.data())).clone();
        } else if (enc == "mono8") {
            mat_ = cv::Mat(msg.height, msg.width, CV_8UC1, const_cast<unsigned char*>(msg.data.data())).clone();
        } else if (enc == "16UC1") {
            mat_ = cv::Mat(msg.height, msg.width, CV_16UC1, const_cast<unsigned char*>(msg.data.data())).clone();
        } else if (enc == "32FC1") {
            mat_ = cv::Mat(msg.height, msg.width, CV_32FC1, const_cast<unsigned char*>(msg.data.data())).clone();
        } else {
            // 未知エンコーディングはバイト列のまま複製
            mat_ = cv::Mat(1, static_cast<int>(msg.data.size()), CV_8UC1, const_cast<unsigned char*>(msg.data.data())).clone();
        }
    }

    static sensor_msgs::msg::Image to_msg(const Image &img, const std_msgs::msg::Header &hdr) {
        sensor_msgs::msg::Image out;
        out.header = hdr;
        if (img.mat_.type() == CV_8UC3) {
            out.height = img.mat_.rows; out.width = img.mat_.cols;
            out.encoding = "bgr8"; out.step = static_cast<uint32_t>(img.mat_.cols * 3);
            out.data.assign(img.mat_.datastart, img.mat_.dataend);
        } else if (img.mat_.type() == CV_8UC1) {
            out.height = img.mat_.rows; out.width = img.mat_.cols;
            out.encoding = "mono8"; out.step = static_cast<uint32_t>(img.mat_.cols);
            out.data.assign(img.mat_.datastart, img.mat_.dataend);
        } else if (img.mat_.type() == CV_16UC1) {
            out.height = img.mat_.rows; out.width = img.mat_.cols;
            out.encoding = "16UC1"; out.step = static_cast<uint32_t>(img.mat_.cols * 2);
            const auto bytes = img.mat_.total() * img.mat_.elemSize();
            out.data.resize(bytes);
            std::memcpy(out.data.data(), img.mat_.data, bytes);
        } else if (img.mat_.type() == CV_32FC1) {
            out.height = img.mat_.rows; out.width = img.mat_.cols;
            out.encoding = "32FC1"; out.step = static_cast<uint32_t>(img.mat_.cols * 4);
            const auto bytes = img.mat_.total() * img.mat_.elemSize();
            out.data.resize(bytes);
            std::memcpy(out.data.data(), img.mat_.data, bytes);
        } else {
            // フォールバック（バイト配列）
            out.height = 1; out.width = static_cast<uint32_t>(img.mat_.total());
            out.encoding = "mono8"; out.step = out.width;
            out.data.assign(img.mat_.datastart, img.mat_.dataend);
        }
        return out;
    }

private:
    cv::Mat mat_;
};

inline sensor_msgs::msg::Image to_msg(const Image &img, const std_msgs::msg::Header &hdr) {
    return Image::to_msg(img, hdr);
}

} // namespace fluent_image

