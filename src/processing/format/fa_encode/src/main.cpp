#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "fa_encode/fa_encode_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fa_encode::FaEncodeNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
