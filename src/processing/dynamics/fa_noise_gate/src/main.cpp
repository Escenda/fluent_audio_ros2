#include <cstdlib>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "fa_noise_gate/fa_noise_gate_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_noise_gate::FaNoiseGateNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(rclcpp::get_logger("fa_noise_gate"), "Exception: %s", e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}
