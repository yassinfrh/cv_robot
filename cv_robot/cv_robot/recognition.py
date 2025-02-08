import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory

def get_model_path():
    package_name = "cv_robot"
    file_name = "models/yolo11n.pt"

    package_share_directory = get_package_share_directory(package_name)
    file_path = os.path.join(package_share_directory, file_name)

    return file_path

class Recognition(Node):
    def __init__(self):
        super().__init__('recognition')
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10)
        self.subscription  # prevent unused variable warning
        self.cv_bridge = CvBridge()
        self.model = YOLO(get_model_path())
        self.get_logger().info('Recognition node has been started')

    def image_callback(self, msg):
        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        result = self.model(cv_image)[0]
        img = result.plot()
        self.get_logger().info('Image has been processed')
        cv2.imshow('Detection Results', img)
        cv2.waitKey(1)

        
def main(args=None):
    rclpy.init(args=args)
    recognition = Recognition()
    rclpy.spin(recognition)
    recognition.destroy_node()
    rclpy.shutdown()
    
if __name__ == '__main__':
    main()