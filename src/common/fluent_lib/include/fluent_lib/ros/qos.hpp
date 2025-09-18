#pragma once

#include <rclcpp/rclcpp.hpp>

namespace fluent_lib::ros {

inline rclcpp::QoS reliable(int depth) { return rclcpp::QoS(depth).reliable(); }
inline rclcpp::QoS best_effort(int depth) { return rclcpp::QoS(depth).best_effort(); }

} // namespace fluent_lib::ros

