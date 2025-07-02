import rclpy  # ROS 2 Python client library, used for creating ROS nodes and handling subscriptions/publishers.
from rclpy.node import Node 
from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import OccupancyGrid
from cv_bridge import CvBridge
from sensor_msgs_py import point_cloud2 as pcl2

import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F

import numpy as np
import cv2
import time


class SituationalAwareness(Node):
    def __init__(self):
        super().__init__('situational_awareness_node')

        self.bridge = CvBridge()
        self.lidar_points = []
        self.image = None
        self.last_time = time.time()
        self.fps_target = 5.0  # 5 frames per second

        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

        self.create_subscription(Image, '/camera/color/image_raw', self.image_callback, 10)
        self.create_subscription(PointCloud2, '/lidar_points', self.lidar_callback, 10)
        self.grid_pub = self.create_publisher(OccupancyGrid, '/situational_map', 10)

    def lidar_callback(self, msg):
        points = []
        for p in pcl2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = p
            if -0.5 < z < 0.5:  # Only keep near-ground points
                points.append((x, y))
        self.lidar_points = points

    def image_callback(self, msg):
        now = time.time()
        if now - self.last_time < 1 / self.fps_target:
            return
        self.last_time = now

        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.process_frame()

    def process_frame(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        input_tensor = F.to_tensor(img).unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(input_tensor)[0]

        grid = np.zeros((100, 100), dtype=np.int8)

        # Mark AI-detected obstacles
        for i, mask in enumerate(outputs['masks']):
            if outputs['scores'][i] < 0.8:
                continue
            m = mask[0].mul(255).byte().cpu().numpy()
            resized = cv2.resize(m, (100, 100))
            grid[resized > 128] = 100

        # Mark LiDAR-detected obstacles
        for x, y in self.lidar_points:
            gx = int((x + 5) * 10)  # map -5~5m to 0~100 grid
            gy = int((y + 5) * 10)
            if 0 <= gx < 100 and 0 <= gy < 100:
                grid[gy, gx] = 100

        self.publish_map(grid)

    def publish_map(self, grid):
        msg = OccupancyGrid()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.info.resolution = 0.1
        msg.info.width = 100
        msg.info.height = 100
        msg.info.origin.position.x = -5.0
        msg.info.origin.position.y = -5.0
        msg.info.origin.position.z = 0.0
        msg.data = grid.flatten().tolist()
        self.grid_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SituationalAwareness()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
