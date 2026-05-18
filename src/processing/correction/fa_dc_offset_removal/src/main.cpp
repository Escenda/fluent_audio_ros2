#include <cstdlib>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "fa_dc_offset_removal/fa_dc_offset_removal_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_dc_offset_removal::FaDcOffsetRemovalNode>();
    rclcpp::spin(node);
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_dc_offset_removal"), "Exception: %s", e.what());
    if (rclcpp::ok()) {
      rclcpp::shutdown();
    }
    return EXIT_FAILURE;
  }
}
