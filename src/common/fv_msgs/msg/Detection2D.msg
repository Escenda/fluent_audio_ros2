std_msgs/Header header

int32 id
uint8 SOURCE_OBJECT=1
uint8 SOURCE_INSTANCE=2
uint8 SOURCE_SEMANTIC=4
uint8 source_mask

int32 class_id
string label
float32 conf_fused
float32 conf_object
float32 conf_instance
float32 conf_semantic

geometry_msgs/Point32 bbox_min
geometry_msgs/Point32 bbox_max

uint32 mask_instance_id
uint32 mask_semantic_id

float32 depth_hint_m
builtin_interfaces/Time observed_at
