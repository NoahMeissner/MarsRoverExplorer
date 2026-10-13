#!/usr/bin/env python3

import math
import random
from enum import Enum
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import functional as TF
import cv2  # OpenCV2
import numpy as np  # [P3]
import rclpy
import message_filters  # [P3] synchronise RGB + depth image
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point, PointStamped  # [P3] + PointStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo  # [P3] + CameraInfo
from std_msgs.msg import ColorRGBA  # [P3]
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point  # [P3]
from sklearn.cluster import DBSCAN  # [P3] clustering of repeated observations

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


# [P3] Marker colour (r, g, b) per artefact class
ARTIFACT_COLOURS = {
    'alien':    (0.0, 1.0, 0.2),
    'ball':     (0.2, 0.4, 1.0),
    'graffiti': (1.0, 0.0, 1.0),
    'mushroom': (1.0, 0.5, 0.0),
    'stone':    (0.6, 0.4, 0.2),
    'stop':     (1.0, 0.0, 0.0),
}


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

def pose2d_to_pose(pose_2d):
    """Convert a Pose2D to a full 3D Pose"""
    pose = Pose()

    pose.position.x = pose_2d.x
    pose.position.y = pose_2d.y

    pose.orientation.w = math.cos(pose_2d.theta / 2.0)
    pose.orientation.z = math.sin(pose_2d.theta / 2.0)

    return pose


class PlannerType(Enum):
    ERROR = 0
    MOVE_FORWARDS = 1
    RETURN_HOME = 2
    GO_TO_FIRST_ARTIFACT = 3
    RANDOM_WALK = 4
    RANDOM_GOAL = 5
    # Add more!


class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer_node')

        # Variables/Flags for mapping
        self.xlim_ = [0.0, 0.0]
        self.ylim_ = [0.0, 0.0]

        # Variables/Flags for perception
        self.artifact_found_ = False

        # Variables/Flags for planning
        self.planner_type_ = PlannerType.ERROR
        self.reached_first_artifact_ = False
        self.returned_home_ = False

        # Marker for artifact locations
        # See https://wiki.ros.org/rviz/DisplayTypes/Marker
        self.marker_artifacts_ = Marker()
        self.marker_artifacts_.header.frame_id = "map"
        self.marker_artifacts_.ns = "artifacts"
        self.marker_artifacts_.id = 0
        self.marker_artifacts_.type = Marker.SPHERE_LIST
        self.marker_artifacts_.action = Marker.ADD
        self.marker_artifacts_.pose.position.x = 0.0
        self.marker_artifacts_.pose.position.y = 0.0
        self.marker_artifacts_.pose.position.z = 0.0
        self.marker_artifacts_.pose.orientation.x = 0.0
        self.marker_artifacts_.pose.orientation.y = 0.0
        self.marker_artifacts_.pose.orientation.z = 0.0
        self.marker_artifacts_.pose.orientation.w = 1.0
        self.marker_artifacts_.scale.x = 1.5
        self.marker_artifacts_.scale.y = 1.5
        self.marker_artifacts_.scale.z = 1.5
        self.marker_artifacts_.color.a = 1.0
        self.marker_artifacts_.color.r = 0.0
        self.marker_artifacts_.color.g = 1.0
        self.marker_artifacts_.color.b = 0.2
        self.marker_pub_ = self.create_publisher(MarkerArray, 'marker_array_artifacts', 10)

        # Remember the artifact locations
        # Array of type geometry_msgs.Point
        self.artifact_locations_ = []
        self.artifact_labels_ = []         # [P3] class name for each entry of artifact_locations_
        self.artifact_observations_ = {}   # [P3] all observations per class: {label: [np.array([x, y, z]), ...]}

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        # [P3] TF is received by a small helper node in its own thread, so the buffer stays
        # up to date while the detector blocks the image callback (~1 s).
        # (spin_thread=True on the main node would also spin the action client -> crash)
        self.tf_node_ = rclpy.create_node(
            'cave_explorer_tf_listener',
            parameter_overrides=[rclpy.Parameter('use_sim_time', value=True)])
        self.tf_listener = TransformListener(self.tf_buffer, self.tf_node_, spin_thread=True)

        # Action client for nav2
        self.nav2_action_client_ = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.get_logger().warn('Waiting for navigate_to_pose action...')
        self.nav2_action_client_.wait_for_server()
        self.get_logger().warn('navigate_to_pose connected')
        self.ready_for_next_goal_ = True
        self.declare_parameter('print_feedback', rclpy.Parameter.Type.BOOL)

        # Publisher for the goal pose visualisation
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose', 1)

        # Subscribe to the map topic to get current bounds
        self.map_grid_ = None   # [P3] latest SLAM map as numpy array (see map_callback)
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map',  self.map_callback, 1)

        # Prepare image processing
        self.image_detections_pub_ = self.create_publisher(Image, 'detections_image', 1)
        self.declare_parameter('computer_vision_model_filename', rclpy.Parameter.Type.STRING)
        self.class_names_ = ['alien', 'ball', 'graffiti', 'mushroom', 'stone', 'stop']  # sorted, label = index + 1
        model = fasterrcnn_mobilenet_v3_large_fpn(weights=None, weights_backbone=None,
                                                num_classes=len(self.class_names_) + 1,
                                                min_size=608, max_size=912)
        model.rpn.anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),) * 3,
                                                    aspect_ratios=((0.5, 1.0, 2.0),) * 3)
        model.load_state_dict(torch.load(self.get_parameter('computer_vision_model_filename').value,
                                        map_location='cpu'))
        model.eval()
        # [P3] PyTorch uses all CPU cores by default and then starves Gazebo, SLAM and the
        # odometry (TF lags by seconds) -> limit the detector to 2 threads
        torch.set_num_threads(2)
        self.computer_vision_model_ = model

        # [P3] RGB + depth image are synchronised by timestamp, so the depth always belongs
        # to the RGB image the detection was made in (the detector takes ~1 s on the CPU)
        self.image_sub_ = message_filters.Subscriber(self, Image, 'camera/image', qos_profile=1)
        self.depth_sub_ = message_filters.Subscriber(self, Image, 'camera/depth/image', qos_profile=1)
        self.image_sync_ = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub_, self.depth_sub_], queue_size=5, slop=0.1)
        self.image_sync_.registerCallback(self.image_callback)

        # [P3] Camera intrinsics for artifact localisation (check names with "ros2 topic list")
        self.camera_info_ = None
        self.declare_parameter('camera_hfov', 2.0944)  # 120 deg, <horizontal_fov> in urdf/mars_explorer.gazebo.xacro (used if no CameraInfo arrives)
        self.camera_info_sub_ = self.create_subscription(CameraInfo, 'camera/camera_info', self.camera_info_callback, 1)

        # Timer for main loop
        self.main_loop_timer_ = self.create_timer(0.2, self.main_loop)

    def get_pose_2d(self):
        """Get the 2d pose of the robot"""

        # Lookup the latest transform
        try:
            t = self.tf_buffer.lookup_transform(
                'map',
                'base_link',
                rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().error(f'Could not transform: {ex}')
            return

        # Return a Pose2D message
        pose = Pose2D()
        pose.x = t.transform.translation.x
        pose.y = t.transform.translation.y

        qw = t.transform.rotation.w
        qz = t.transform.rotation.z

        if qz >= 0.:
            pose.theta = wrap_angle(2. * math.acos(qw))
        else:
            pose.theta = wrap_angle(-2. * math.acos(qw))

        self.get_logger().warn(f'Pose: {pose}')

        return pose

    def map_callback(self, map_msg: OccupancyGrid):
        """New map received, so update x and y limits"""

        # Extract data from message
        map_origin = [map_msg.info.origin.position.x,
                      map_msg.info.origin.position.y]
        map_resolution = map_msg.info.resolution
        map_height = map_msg.info.height
        map_width = map_msg.info.width

        # Set current limits
        self.xlim_ = [map_origin[0], map_origin[0]+map_width*map_resolution]
        self.ylim_ = [map_origin[1], map_origin[1]+map_height*map_resolution]

        # [P3] keep the map to check artifact estimates against walls / unknown space
        self.map_info_ = map_msg.info
        self.map_grid_ = np.array(map_msg.data, dtype=np.int8).reshape(map_height, map_width)

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')

    # [P3] Store the camera intrinsics
    def camera_info_callback(self, info_msg):
        self.camera_info_ = info_msg

    def image_callback(self, image_msg, depth_msg):  # [P3] + depth_msg (synchronised)
        """
        Recieve an RGB image.
        Use this method to detect artifacts of interest.

        A simple method has been provided to begin with for detecting stop signs (which is not what we're actually looking for)
        adapted from: https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
        """

        # Copy the image message to a cv image
        # see http://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
        image = self.cv_bridge_.imgmsg_to_cv2(image_msg, desired_encoding='passthrough')

        # Create a grayscale version (some simple models use this)
        # image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Retrieve the pre-trained model
        stop_sign_model = self.computer_vision_model_

        # Detect artifacts in the image
        # The minSize is used to avoid very small detections that are probably noise
        with torch.no_grad():
            out = stop_sign_model([TF.to_tensor(image)])[0]
        detections = []
        labels = []
        for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
            if score < 0.5:  # CONF from the notebook
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            labels.append(self.class_names_[int(label) - 1])

        # You can set "artifact_found_" to true to signal to "main_loop" that you have found a artifact
        # You may want to communicate more information
        # Since the "image_callback" and "main_loop" methods can run at the same time you should protect any shared variables
        # with a mutex
        # "artifact_found_" doesn't need a mutex because it's an atomic
        num_detections = len(detections)

        if num_detections > 0:
            self.artifact_found_ = True
        else:
            self.artifact_found_ = False

        # Draw a bounding box rectangle on the image for each detection
        for(x, y, width, height) in detections:
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 5)  # [P3] width/height were swapped

        # Publish the image with the detection bounding boxes
        image_detection_message = self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8")
        self.image_detections_pub_.publish(image_detection_message)

        if self.artifact_found_:
            self.get_logger().info('Artifact found!')
            # [P3] depth image in metres, taken at the same time as the RGB image
            depth = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth = depth.astype(np.float32) / (1000.0 if depth_msg.encoding == '16UC1' else 1.0)
            self.localise_artifact(detections, labels, image_msg.header, image.shape, depth)  # [P3] pass detections


    def localise_artifact(self, detections, labels, header, image_shape, depth):
        """
        [P3] Compute the location of each detected artifact:
          - direction from the pixel coordinates of the bounding box centre (pinhole model)
          - distance from the depth camera (25 % quantile of the depth in the centre of the box)
          - transform into the map frame with TF (robot pose at the time the image was taken)
          - check against the SLAM map (not behind a wall, not in unknown space)
          - all observations are clustered per class with DBSCAN (scikit-learn);
            each cluster = one artifact, its position = median of the cluster,
            observations that belong to no cluster are treated as false positives
        Save it to a list, publish rviz marker
        """

        img_h, img_w = image_shape[:2]
        sx, sy = depth.shape[1] / img_w, depth.shape[0] / img_h  # in case depth has another resolution

        # Camera intrinsics (CameraInfo, otherwise from the horizontal field of view)
        if self.camera_info_ is not None:
            k = self.camera_info_.k
            fx, fy, cx, cy = k[0], k[4], k[2], k[5]
            self.get_logger().info(f'Intrinsics from CameraInfo: fx={fx:.1f} cx={cx:.1f}', once=True)
        else:
            fx = fy = img_w / (2.0 * math.tan(self.get_parameter('camera_hfov').value / 2.0))
            cx, cy = img_w / 2.0, img_h / 2.0
            self.get_logger().warn('No CameraInfo received -> intrinsics guessed from camera_hfov', once=True)

        # Camera frame from the image header, base_link if it is not in the TF tree
        frame = header.frame_id
        if not frame or not self.tf_buffer.can_transform('map', frame, rclpy.time.Time()):
            frame = 'base_link'
        # TF in two steps, because SLAM (map -> odom) lags behind by seconds while the detector
        # keeps the CPU busy, but odometry (odom -> base_link) is published fast:
        #   odom -> camera at the time the image was taken (precise robot motion)
        #   map  -> odom latest (SLAM correction, changes only slowly)
        odom = 'odom'
        try:
            t_odom = self.tf_buffer.lookup_transform(odom, frame, rclpy.time.Time.from_msg(header.stamp))
        except TransformException:
            try:
                # latest odometry is only OK if it is close in time to the image
                t_odom = self.tf_buffer.lookup_transform(odom, frame, rclpy.time.Time())
                img_t = header.stamp.sec + header.stamp.nanosec * 1e-9
                tf_t = t_odom.header.stamp.sec + t_odom.header.stamp.nanosec * 1e-9
                if abs(img_t - tf_t) > 0.2:
                    self.get_logger().warn(f'localise_artifact: no odometry close to image time '
                                           f'(dt={img_t - tf_t:.2f} s) -> skipped', throttle_duration_sec=5.0)
                    return
            except TransformException as ex:
                self.get_logger().warn(f'localise_artifact: {ex}', throttle_duration_sec=5.0)
                return
        try:
            t_map = self.tf_buffer.lookup_transform('map', odom, rclpy.time.Time())
        except TransformException as ex:
            self.get_logger().warn(f'localise_artifact: {ex}', throttle_duration_sec=5.0)
            return

        def to_map(point_stamped):
            return do_transform_point(do_transform_point(point_stamped, t_odom), t_map).point

        # camera position in the map (start of the viewing ray for the map check)
        origin = PointStamped()
        origin.header.frame_id = frame
        cam_map = to_map(origin)
        cam = np.array([cam_map.x, cam_map.y])

        for (x, y, w, h), label in zip(detections, labels):
            u, v = x + w / 2.0, y + h / 2.0   # box centre in pixels

            # Distance: depth values in the central third of the box
            patch = depth[int((y + h / 3) * sy):int((y + 2 * h / 3) * sy) + 1,
                          int((x + w / 3) * sx):int((x + 2 * w / 3) * sx) + 1]
            valid = patch[np.isfinite(patch) & (patch > 0.3) & (patch < 8.0)]
            if valid.size < 5:
                self.get_logger().info(f'{label}: out of depth range, skipped', throttle_duration_sec=5.0)
                continue
            # 25 % quantile instead of median: the object is in front of its background,
            # so background pixels inside the box must not make the distance too long
            d = float(np.percentile(valid, 25))

            # Direction: offset of the box centre from the image centre (pinhole model)
            p_map = to_map(self.camera_point(frame, (u - cx) * d / fx, (v - cy) * d / fy, d))
            pos = np.array([p_map.x, p_map.y, p_map.z])

            # Sanity check against the SLAM map: an artifact cannot be behind a wall
            pos = self.check_against_map(cam, pos)
            if pos is None:
                self.get_logger().info(f'{label}: estimate in unknown space / outside map -> rejected')
                continue

            angle = math.degrees(math.atan2(-(u - cx), fx))
            self.get_logger().info(f'{label}: depth={d:.2f} m, angle={angle:+.0f} deg '
                                   f'-> map ({pos[0]:.2f}, {pos[1]:.2f})')

            # Store the observation, clustering is done below
            self.artifact_observations_.setdefault(label, []).append(pos)

        # Cluster all observations of each class with DBSCAN (in the ground plane x, y):
        #   eps = 1.5 m: observations closer than this are neighbours (same artifact)
        #   min_samples = 3: a cluster needs at least 3 observations, the rest is noise (false positives)
        self.artifact_locations_ = []
        self.artifact_labels_ = []
        for label, points in self.artifact_observations_.items():
            points = np.array(points)
            if len(points) < 3:
                continue
            cluster_ids = DBSCAN(eps=1.5, min_samples=3).fit_predict(points[:, :2])
            for cid in set(cluster_ids) - {-1}:     # -1 = noise
                m = np.median(points[cluster_ids == cid], axis=0)
                self.artifact_locations_.append(Point(x=float(m[0]), y=float(m[1]), z=float(m[2])))
                self.artifact_labels_.append(label)

        # Publish the markers
        self.publish_artifact_markers()

    def camera_point(self, frame, right, down, forward):
        """[P3] Point in the camera frame (optical or ROS body convention)"""
        p = PointStamped()
        p.header.frame_id = frame
        if 'optical' in frame:   # optical frame: z forward, x right, y down
            p.point.x, p.point.y, p.point.z = right, down, forward
        else:                    # body frame: x forward, y left, z up
            p.point.x, p.point.y, p.point.z = forward, -right, -down
        return p

    def check_against_map(self, cam, pos):
        """
        [P3] Walk along the viewing ray from the camera to the estimate through the SLAM map.
          - ray hits a wall first -> the depth was too long (e.g. looked past the object),
            the estimate is moved onto the wall (slightly in front of it)
          - estimate lies in unknown space or outside the map -> rejected (None)
        """
        if self.map_grid_ is None:
            return pos
        info = self.map_info_
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        h, w = self.map_grid_.shape
        ray = pos[:2] - cam
        dist = np.linalg.norm(ray)
        if dist < 1e-3:
            return pos
        n = int(dist / res) + 1
        for i in range(1, n + 1):
            p = cam + ray * i / n
            c, r = int((p[0] - ox) / res), int((p[1] - oy) / res)
            if not (0 <= r < h and 0 <= c < w):
                return None
            if self.map_grid_[r, c] >= 65:          # occupied cell -> wall
                p = p - 0.15 * ray / dist           # 15 cm in front of the wall
                return np.array([p[0], p[1], pos[2]])
        if self.map_grid_[r, c] < 0:                # unknown cell
            return None
        return pos

    def publish_artifact_markers(self):
        """ Publish the artifact location markers"""

        # Update the locations
        self.marker_artifacts_.points = self.artifact_locations_
        # [P3] one colour per class
        self.marker_artifacts_.colors = [ColorRGBA(r=c[0], g=c[1], b=c[2], a=1.0)
                                         for c in (ARTIFACT_COLOURS.get(l, (1.0, 1.0, 0.0))
                                                   for l in self.artifact_labels_)]

        # Create and publish the MarkerArray
        marker_array = MarkerArray()
        # [P3] remove old text labels first (number of clusters can shrink when clusters merge)
        clear = Marker()
        clear.header.frame_id = "map"
        clear.ns = "artifact_labels"
        clear.action = Marker.DELETEALL
        marker_array.markers = [clear, self.marker_artifacts_]

        # [P3] text label above each artifact
        for i, (point, label) in enumerate(zip(self.artifact_locations_, self.artifact_labels_)):
            text = Marker()
            text.header.frame_id = "map"
            text.ns = "artifact_labels"
            text.id = i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = Point(x=point.x, y=point.y, z=point.z + 1.5)
            text.pose.orientation.w = 1.0
            text.scale.z = 0.8
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = label
            marker_array.markers.append(text)

        self.marker_pub_.publish(marker_array)


    def planner_go_to_pose2d(self, pose2d):
        """Go to a provided 2d pose"""

        # Send a goal to navigate_to_pose with self.nav2_action_client_
        action_goal = NavigateToPose.Goal()
        action_goal.pose.header.stamp = self.get_clock().now().to_msg()
        action_goal.pose.header.frame_id = 'map'
        action_goal.pose.pose = pose2d_to_pose(pose2d)

        # Publish visualisation
        self.goal_pose_vis_.publish(action_goal.pose)

        # Decide whether to show feedback or not
        if self.get_parameter('print_feedback').value:
            feedback_method = self.feedback_callback
        else:
            feedback_method = None

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """The requested goal pose has been sent to the action server"""

        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(self.goal_reached_callback)

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future):
        """The requested goal has been reached"""

        result = future.result().result
        self.get_logger().info(f'Goal reached!')
        self.ready_for_next_goal_ = True


    def planner_move_forwards(self, distance):
        """Simply move forward by the specified distance"""

        pose_2d = self.get_pose_2d()

        pose_2d.x += distance * math.cos(pose_2d.theta)
        pose_2d.y += distance * math.sin(pose_2d.theta)

        self.planner_go_to_pose2d(pose_2d)

    def planner_go_to_first_artifact(self):
        """Go to a pre-specified artifact location"""

        goal_pose2d = Pose2D(
            x = 18.1,
            y = 6.6,
            theta = math.pi/2
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_return_home(self):
        """Return to the origin"""

        goal_pose2d = Pose2D(
            x = 0.0,
            y = 0.0,
            theta = math.pi
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_walk(self):
        """Go to a random location, which may be invalid"""

        # Select a random location
        goal_pose2d = Pose2D(
            x = random.uniform(self.xlim_[0], self.xlim_[1]),
            y = random.uniform(self.ylim_[0], self.ylim_[1]),
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def planner_random_goal(self):
        """Go to a random location out of a predefined set"""

        # Hand picked set of goal locations
        random_goals = [[15.2, 2.2],
                        [30.7, 2.2],
                        [43.0, 11.3],
                        [36.6, 21.9],
                        [33.0, 30.4],
                        [40.4, 44.3],
                        [51.5, 37.8],
                        [16.0, 24.1],
                        [3.4, 33.5],
                        [7.9, 13.8],
                        [14.2, 37.7]]

        # Select a random location
        goal_valid = False
        while not goal_valid:
            idx = random.randint(0,len(random_goals)-1)
            goal_x = random_goals[idx][0]
            goal_y = random_goals[idx][1]

            # Only accept this goal if it's within the current costmap bounds
            if goal_x > self.xlim_[0] and goal_x < self.xlim_[1] and \
               goal_y > self.ylim_[0] and goal_y < self.ylim_[1]:
                goal_valid = True
            else:
                self.get_logger().warn(f'Goal [{goal_x}, {goal_y}] out of bounds')

        goal_pose2d = Pose2D(
            x = goal_x,
            y = goal_y,
            theta = random.uniform(0, 2*math.pi)
        )
        self.planner_go_to_pose2d(goal_pose2d)

    def main_loop(self):
        """
        Set the next goal pose and send to the action server
        See https://docs.nav2.org/concepts/index.html
        """

        # Don't do anything until SLAM is launched
        if not self.tf_buffer.can_transform(
                'map',
                'base_link',
                rclpy.time.Time()):
            self.get_logger().warn('Waiting for transform... Have you launched a SLAM node?')
            return

        #######################################################
        # Update flags related to the progress of the current planner

        # Check if previous goal still running
        if not self.ready_for_next_goal_:
            # self.get_logger().info(f'Previous goal still running')
            return

        self.ready_for_next_goal_ = False

        if self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.get_logger().info('Successfully reached first artifact!')
            self.reached_first_artifact_ = True
        if self.planner_type_ == PlannerType.RETURN_HOME:
            self.get_logger().info('Successfully returned home!')
            self.returned_home_ = True

        #######################################################
        # Select the next planner to execute
        # Update this logic as you see fit!
        if not self.reached_first_artifact_:
            self.planner_type_ = PlannerType.GO_TO_FIRST_ARTIFACT
        elif not self.returned_home_:
            self.planner_type_ = PlannerType.RETURN_HOME
        else:
            self.planner_type_ = PlannerType.RANDOM_GOAL

        #######################################################
        # Execute the planner by calling the relevant method
        # Add your own planners here!
        self.get_logger().info(f'Calling planner: {self.planner_type_.name}')
        if self.planner_type_ == PlannerType.MOVE_FORWARDS:
            self.planner_move_forwards(10)
        elif self.planner_type_ == PlannerType.GO_TO_FIRST_ARTIFACT:
            self.planner_go_to_first_artifact()
        elif self.planner_type_ == PlannerType.RETURN_HOME:
            self.planner_return_home()
        elif self.planner_type_ == PlannerType.RANDOM_WALK:
            self.planner_random_walk()
        elif self.planner_type_ == PlannerType.RANDOM_GOAL:
            self.planner_random_goal()
        else:
            self.get_logger().error('No valid planner selected')
            self.destroy_node()


        #######################################################

def main():
    # Initialise
    rclpy.init()

    # Create the cave explorer
    cave_explorer = CaveExplorer()

    while rclpy.ok():
        rclpy.spin(cave_explorer)