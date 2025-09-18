#pragma once

#include <rclcpp/rclcpp.hpp>
#include <chrono>

namespace fluent_lib::ros {

template <class NodeT, class Fn>
auto wall_timer(NodeT &node, std::chrono::milliseconds period, Fn &&fn) {
    return node.create_wall_timer(period, std::forward<Fn>(fn));
}

} // namespace fluent_lib::ros

