#pragma once

#include <functional>
#include <rclcpp/rclcpp.hpp>

// 小さなDSLユーティリティ（互換レイヤ）
namespace fluent_lib::ros {

inline rclcpp::QoS qos_keep_last(int depth) { return rclcpp::QoS(depth > 0 ? depth : 1); }

template <class NodeT, class MsgT, class Fn>
auto sub(NodeT &node, const std::string &topic, const rclcpp::QoS &q, Fn &&fn) {
    return node.template create_subscription<MsgT>(topic, q, std::forward<Fn>(fn));
}

} // namespace fluent_lib::ros
