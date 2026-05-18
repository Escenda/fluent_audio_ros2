#include <cstdlib>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "fa_aec_linear/fa_aec_linear_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_aec_linear::FaAecLinearNode>();
    rclcpp::spin(node);
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_aec_linear"), "Exception: %s", e.what());
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
