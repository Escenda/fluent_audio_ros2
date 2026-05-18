#include "fa_crossfade/fa_crossfade_node.hpp"

#include <rclcpp/rclcpp.hpp>

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<fa_crossfade::FaCrossfadeNode>());
  rclcpp::shutdown();
  return 0;
}
