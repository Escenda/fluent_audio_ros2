#include "fa_file_out/fa_file_out_node.hpp"

#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fa_file_out::FaFileOutNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
