#include "fa_packet_loss_concealment/fa_packet_loss_concealment_node.hpp"

#include <cstdlib>
#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<fa_packet_loss_concealment::FaPacketLossConcealmentNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return EXIT_SUCCESS;
  } catch (const std::exception & e) {
    RCLCPP_FATAL(
      rclcpp::get_logger("fa_packet_loss_concealment_node"),
      "Exception: %s",
      e.what());
    rclcpp::shutdown();
    return EXIT_FAILURE;
  }
}
