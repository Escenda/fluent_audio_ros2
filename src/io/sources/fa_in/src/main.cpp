#include "fa_in/fa_in_node.hpp"

#include <cstdlib>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_in::FaInNode>();
    rclcpp::spin(node);
    const bool fatal_error = node->hasFatalError();
    rclcpp::shutdown();
    return fatal_error ? EXIT_FAILURE : EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_in"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}
