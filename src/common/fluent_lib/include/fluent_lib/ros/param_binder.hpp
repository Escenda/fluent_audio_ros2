#pragma once

#include <rclcpp/rclcpp.hpp>
#include <string>

namespace fluent_lib::ros {

template <typename T>
inline T get_param(rclcpp::Node &node, const std::string &name, const T &def_val) {
    node.declare_parameter<T>(name, def_val);
    return node.get_parameter(name).template get_value<T>();
}

} // namespace fluent_lib::ros

