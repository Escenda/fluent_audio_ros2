#pragma once

#include <rclcpp/rclcpp.hpp>

namespace fluent_lib::ros {

inline bool has_parameter(rclcpp::Node &node, const std::string &name) {
    const auto list = node.list_parameters({name}, 1);
    for (const auto &n : list.names) if (n == name) return true;
    return false;
}

} // namespace fluent_lib::ros

