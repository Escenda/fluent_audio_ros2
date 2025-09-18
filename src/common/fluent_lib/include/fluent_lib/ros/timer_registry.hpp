#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>

namespace fluent_lib::ros {

struct TimerRegistry {
    std::vector<rclcpp::TimerBase::SharedPtr> timers;
    void add(const rclcpp::TimerBase::SharedPtr &t) { timers.push_back(t); }
};

} // namespace fluent_lib::ros

