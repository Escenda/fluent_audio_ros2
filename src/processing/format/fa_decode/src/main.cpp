#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "fa_decode/fa_decode_node.hpp"

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fa_decode::FaDecodeNode>());
  rclcpp::shutdown();
  return 0;
}
