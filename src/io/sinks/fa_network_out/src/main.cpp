#include "fa_network_out/fa_network_out_node.hpp"

#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fa_network_out::FaNetworkOutNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
