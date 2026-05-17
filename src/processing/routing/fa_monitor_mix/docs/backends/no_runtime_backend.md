# no_runtime_backend

`fa_monitor_mix` は ROS2 message routing と sample addition のみを行うため、外部 backend を持たない。

device、network、file、resample、limiter、normalize は別 package の責務であり、この package には
runtime backend adapter を追加しない。
