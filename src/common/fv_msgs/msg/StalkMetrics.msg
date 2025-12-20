uint8 GRADE_UNKNOWN=0
uint8 GRADE_A=1
uint8 GRADE_B=2
uint8 GRADE_REJECT=3

std_msgs/Header header

int32 id
geometry_msgs/Point root_camera
geometry_msgs/Point tip_camera
geometry_msgs/Vector3 axis_camera

float32 length_m
float32 curvature
float32 confidence
float32 root_radius_m
float32 thickness_m
uint8 grade

geometry_msgs/Point root_world
geometry_msgs/Point tip_world
builtin_interfaces/Time updated_at

# 骨格点列（PCA軸に沿ってサンプリングした実際の点群、root→tip順）
geometry_msgs/Point[] skeleton_points
