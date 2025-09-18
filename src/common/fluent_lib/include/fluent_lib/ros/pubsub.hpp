#pragma once

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <opencv2/imgcodecs.hpp>
#include "fluent_lib/fluent_image/image.hpp"

namespace fluent_lib::ros {

// FluentImage から publish する簡易ヘルパ
template <class PubT>
inline void publish(PubT &pub, const fluent_image::Image &img, const std_msgs::msg::Header &hdr) {
    auto msg = fluent_image::to_msg(img, hdr);
    pub->publish(msg);
}

} // namespace fluent_lib::ros

// 圧縮画像（jpeg/png）をパブリッシュ
namespace fluent_lib::ros {

template <class PubT>
inline void publish_compressed(PubT &pub,
                               const fluent_image::Image &img,
                               const std_msgs::msg::Header &hdr,
                               int quality = 85,
                               const std::string &format = "jpeg")
{
    sensor_msgs::msg::CompressedImage out;
    out.header = hdr;
    out.format = format;
    std::vector<uchar> buf;
    std::vector<int> params;
    if (format == "jpeg" || format == "jpg" || format == "JPG" || format == "JPEG") {
        params = {cv::IMWRITE_JPEG_QUALITY, std::max(1, std::min(100, quality))};
        cv::imencode(".jpg", static_cast<const cv::Mat&>(img), buf, params);
    } else {
        params = {cv::IMWRITE_PNG_COMPRESSION, 3};
        cv::imencode(".png", static_cast<const cv::Mat&>(img), buf, params);
    }
    out.data = std::move(buf);
    pub->publish(out);
}

} // namespace fluent_lib::ros
