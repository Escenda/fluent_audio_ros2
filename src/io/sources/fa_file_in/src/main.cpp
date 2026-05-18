#include "fa_file_in/fa_file_in_node.hpp"

#include <memory>

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<fa_file_in::FaFileInNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
