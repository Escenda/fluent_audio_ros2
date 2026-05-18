#include <cstdlib>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "fa_monitor_mix/fa_monitor_mix_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_monitor_mix::FaMonitorMixNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_monitor_mix"), "Exception: %s", e.what());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return EXIT_FAILURE;
  }
  if (rclcpp::ok()) {
    rclcpp::shutdown();
  }
  return EXIT_SUCCESS;
}
