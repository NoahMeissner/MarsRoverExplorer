import os
import time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from message_filters import Subscriber, ApproximateTimeSynchronizer

RGB_TOPIC = '/camera/image'
DEPTH_TOPIC = '/camera/depth/image'
BASE_DIR = os.path.expanduser('~/cave_dataset')
SAVE_INTERVAL = 1.0   # Sekunden zwischen Bildern
MIN_DIFF = 12.0       # wie stark sich das RGB-Bild ändern muss
MAX_DEPTH = 10.0      # Meter, nur für die Vorschau-Bilder


class DatasetSaver(Node):
    def __init__(self):
        super().__init__('dataset_saver')
        self.dirs = {k: os.path.join(BASE_DIR, k)
                     for k in ['rgb', 'depth', 'depth_vis']}
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

        self.bridge = CvBridge()
        self.last_time = 0.0
        self.last_small = None
        self.count = len(os.listdir(self.dirs['rgb']))

        rgb_sub = Subscriber(self, Image, RGB_TOPIC)
        depth_sub = Subscriber(self, Image, DEPTH_TOPIC)
        self.sync = ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=10, slop=0.1)
        self.sync.registerCallback(self.callback)
        self.get_logger().info(f'Speichere RGB + Tiefe nach {BASE_DIR}')

    def callback(self, rgb_msg, depth_msg):
        now = time.time()
        if now - self.last_time < SAVE_INTERVAL:
            return

        rgb = self.bridge.imgmsg_to_cv2(rgb_msg, 'bgr8')

        # Fast gleiche Bilder überspringen
        small = cv2.resize(rgb, (64, 48)).astype(np.float32)
        if self.last_small is not None and \
                np.mean(np.abs(small - self.last_small)) < MIN_DIFF:
            return

        # Tiefe in Meter (float) umwandeln
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        if depth_msg.encoding == '16UC1':
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            depth_m = depth.astype(np.float32)
        depth_m[~np.isfinite(depth_m)] = 0.0   # inf/nan -> 0

        name = f'img_{self.count:05d}'
        # 1) RGB
        cv2.imwrite(os.path.join(self.dirs['rgb'], name + '.png'), rgb)
        # 2) Tiefe exakt als 16-bit PNG in Millimetern
        depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
        cv2.imwrite(os.path.join(self.dirs['depth'], name + '.png'), depth_mm)
        # 3) Vorschau zum Anschauen (nicht zum Trainieren)
        vis = np.clip(depth_m / MAX_DEPTH * 255, 0, 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(self.dirs['depth_vis'], name + '.png'), vis)

        self.get_logger().info(f'Gespeichert: {name}')
        self.count += 1
        self.last_time = now
        self.last_small = small


def main():
    rclpy.init()
    rclpy.spin(DatasetSaver())


if __name__ == '__main__':
    main()