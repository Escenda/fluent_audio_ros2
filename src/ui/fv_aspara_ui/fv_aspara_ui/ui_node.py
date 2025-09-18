import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from fv_msgs.msg import DetectionArray
import cv2
from cv_bridge import CvBridge


class AsparaUiNode(Node):
    def __init__(self):
        super().__init__('fv_aspara_ui')

        # Parameters
        self.declare_parameter('image_topic', '/fv/selected/d405/color/image_raw')
        self.declare_parameter('detections_topic', '/fv/d405/detection_fusion/rois')
        self.declare_parameter('annotated_topic', '/fv/d405/ui/annotated_image')
        self.declare_parameter('draw_labels', True)
        self.declare_parameter('draw_confidence', True)

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        det_topic = self.get_parameter('detections_topic').get_parameter_value().string_value
        self._annotated_topic = self.get_parameter('annotated_topic').get_parameter_value().string_value
        self._draw_labels = self.get_parameter('draw_labels').get_parameter_value().bool_value
        self._draw_conf = self.get_parameter('draw_confidence').get_parameter_value().bool_value

        qos_sensor = QoSProfile(depth=5)
        qos_sensor.reliability = ReliabilityPolicy.BEST_EFFORT
        qos_sensor.history = HistoryPolicy.KEEP_LAST

        self._bridge = CvBridge()
        self._latest_dets = None

        self._sub_img = self.create_subscription(Image, image_topic, self._on_image, qos_sensor)
        self._sub_det = self.create_subscription(DetectionArray, det_topic, self._on_dets, 10)
        self._pub_anno = self.create_publisher(Image, self._annotated_topic, qos_sensor)

        self.get_logger().info(f'Aspara UI started: img={image_topic}, det={det_topic}, out={self._annotated_topic}')

    def _on_dets(self, msg: DetectionArray):
        self._latest_dets = msg

    def _on_image(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f'cv_bridge convert failed: {e}')
            return

        dets = self._latest_dets
        if dets is not None and dets.detections:
            for d in dets.detections:
                x1 = int(d.bbox_min.x)
                y1 = int(d.bbox_min.y)
                x2 = int(d.bbox_max.x)
                y2 = int(d.bbox_max.y)
                x1 = max(0, min(cv_img.shape[1]-1, x1))
                y1 = max(0, min(cv_img.shape[0]-1, y1))
                x2 = max(0, min(cv_img.shape[1]-1, x2))
                y2 = max(0, min(cv_img.shape[0]-1, y2))
                cv2.rectangle(cv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'ID {d.id}' if self._draw_labels else ''
                if self._draw_conf and d.confidence > 0:
                    if label:
                        label += f' ({d.confidence:.2f})'
                    else:
                        label = f'{d.confidence:.2f}'
                if label:
                    cv2.putText(cv_img, label, (x1, max(0, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)

        try:
            out = self._bridge.cv2_to_imgmsg(cv_img, encoding='bgr8')
            out.header = msg.header  # keep original timestamp/frame
            self._pub_anno.publish(out)
        except Exception as e:
            self.get_logger().warn(f'publish failed: {e}')


def main():
    rclpy.init()
    node = AsparaUiNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

