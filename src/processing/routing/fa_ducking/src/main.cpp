#include "fa_ducking/fa_ducking_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fa_ducking::FaDuckingNode>());
  rclcpp::shutdown();
  return 0;
}
