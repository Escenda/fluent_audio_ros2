#pragma once

#include <rclcpp/rclcpp.hpp>

namespace fluent_lib::ros {

class FluentNode : public rclcpp::Node {
public:
    explicit FluentNode(const std::string &name, const rclcpp::NodeOptions &opts=rclcpp::NodeOptions())
        : rclcpp::Node(name, opts) {}
};

} // namespace fluent_lib::ros

