#!/usr/bin/env python3

import math
import os
import random
import time
from collections import deque
from enum import Enum
import torch
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.transforms import functional as TF
import cv2  # OpenCV2
import numpy as np
import rclpy
import message_filters 
from cv_bridge import CvBridge
from action_msgs.msg import GoalStatus  # [P1]
from geometry_msgs.msg import Pose, Pose2D, PoseStamped, Point, PointStamped  # [P3] + PointStamped
from geometry_msgs.msg import Twist  # [P1]
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import Image, CameraInfo  # [P3] + CameraInfo
from sensor_msgs.msg import PointCloud2  # [P1]
from sensor_msgs_py import point_cloud2  # [P1]
from std_msgs.msg import Header  # [P1]
from std_msgs.msg import ColorRGBA  # [P3]
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point 
from sklearn.cluster import DBSCAN 

from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from cave_explorer.frontier_exploration import FrontierExplorer, OCCUPIED_THRESHOLD, wrap_pi  # [P1]
from cave_explorer.inspection import choose_viewpoint, line_of_sight  # [P2]
from cave_explorer.degradation import degrade, dust_alpha, dust_spots


# Marker colour (r, g, b) per artefact class
ARTIFACT_COLOURS = {
    'alien':    (0.0, 1.0, 0.2),
    'ball':     (0.2, 0.4, 1.0),
    'graffiti': (1.0, 0.0, 1.0),
    'mushroom': (1.0, 0.5, 0.0),
    'stone':    (0.6, 0.4, 0.2),
    'stop':     (1.0, 0.0, 0.0),
}

INSPECTION_STATUS_COLOURS = {
    'pending':    (1.0, 0.6, 0.0),
    'inspecting': (1.0, 1.0, 0.0),
    'inspected':  (0.0, 1.0, 0.0),
    'abandoned':  (1.0, 0.0, 0.0),
}


def wrap_angle(angle):
    """Function to wrap an angle between 0 and 2*Pi"""
    while angle < 0.0:
        angle = angle + 2 * math.pi

    while angle > 2 * math.pi:
        angle = angle - 2 * math.pi

    return angle

# Helper Class
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
    EXPLORE_FRONTIERS = 6   # [P1] frontier-based exploration
    EXPLORATION_DONE = 7    # [P1] no frontier left, returned home
    INSPECT_ARTEFACT = 8    # [P2] close-range inspection of a detected artefact
    INSPECTION_DONE = 9     # [P2] robot at the inspection viewpoint, behaviour ends here
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
        self.artifact_labels_ = []         # class name for each entry of artifact_locations_
        self.artifact_observations_ = {}   # all observations per class: {label: [np.array([x, y, z]), ...]}

        # Initialise CvBridge
        self.cv_bridge_ = CvBridge()

        # Prepare transformation to get robot pose
        self.tf_buffer = Buffer()
        # TF is received by a small helper node in its own thread, so the buffer stays
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
        # [P1] Not 'goal_pose': bt_navigator subscribes to that topic (RViz "2D Goal Pose") and sent
        # every goal a second time, which preempted our own goal -> spurious ABORTED results
        self.goal_pose_vis_ = self.create_publisher(PoseStamped, 'goal_pose_vis', 1)

        # Subscribe to the map topic to get current bounds
        self.map_grid_ = None   # latest SLAM map as numpy array (see map_callback)
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
        # PyTorch uses all CPU cores by default and then starves Gazebo, SLAM and the
        # odometry (TF lags by seconds) -> limit the detector to 2 threads
        torch.set_num_threads(2)
        self.computer_vision_model_ = model

        # RGB + depth image are synchronised by timestamp, so the depth always belongs
        # to the RGB image the detection was made in (the detector takes ~1 s on the CPU)
        self.image_sub_ = message_filters.Subscriber(self, Image, 'camera/image', qos_profile=1)
        self.depth_sub_ = message_filters.Subscriber(self, Image, 'camera/depth/image', qos_profile=1)
        self.image_sync_ = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub_, self.depth_sub_], queue_size=5, slop=0.1)
        self.image_sync_.registerCallback(self.image_callback)

        # Camera intrinsics for artifact localisation (check names with "ros2 topic list")
        self.camera_info_ = None
        self.declare_parameter('camera_hfov', 2.0944)  # 120 deg, <horizontal_fov> in urdf/mars_explorer.gazebo.xacro (used if no CameraInfo arrives)
        self.camera_info_sub_ = self.create_subscription(CameraInfo, 'camera/camera_info', self.camera_info_callback, 1)

        # [P1] Frontier-based exploration (see frontier_exploration.py)
        p = lambda name, default: self.declare_parameter('exploration.' + name, default).value
        self.explorer_ = FrontierExplorer(
            min_clearance=p('min_clearance', 0.6),          # [m] robot_radius in nav2_params.yaml is 0.5
            min_frontier_size=p('min_frontier_size', 4),    # [cells] = 1.2 m at the 0.3 m SLAM resolution
            viewpoint_radius=p('viewpoint_radius', 1.5),
            min_goal_dist=p('min_goal_dist', 1.0),          # = xy_goal_tolerance of Nav2
            w_size=p('w_size', 1.0),
            w_dist=p('w_dist', 1.0),
            w_turn=p('w_turn', 1.0),
            target_bonus=p('target_bonus', 3.0))
        self.replan_period_ = p('replan_period', 2.0)       # [s] re-evaluate frontiers while driving
        self.visited_radius_ = p('visited_radius', 1.5)     # [m] reached goals are not selected again
        self.failed_radius_ = p('failed_radius', 2.0)       # [m] goals Nav2 failed on are not selected again
        self.max_attempts_ = p('max_attempts', 2)           # Nav2 failures before a goal is given up
        self.stuck_time_ = p('stuck_time', 20.0)            # [s] no progress for this long -> rewind
        self.rewind_distance_ = p('rewind_distance', 1.5)   # [m] driven back along the trajectory when stuck
        self.rewind_speed_ = p('rewind_speed', 0.3)         # [m/s]
        self.rewind_timeout_ = p('rewind_timeout', 12.0)    # [s] then escape manoeuvres are tried
        self.obstacle_offset_ = p('obstacle_offset', 1.0)   # [m] virtual obstacle centre, ahead of the stuck robot
        self.obstacle_radius_ = p('obstacle_radius', 0.7)   # [m] virtual obstacle radius
        self.log_dir_ = os.path.expanduser(p('log_dir', '~/cave_explorer_logs'))
        planner = self.declare_parameter('planner', 'explore').value   # 'explore' or 'template'
        if planner == 'explore':
            self.planner_type_ = PlannerType.EXPLORE_FRONTIERS

        self.goal_seq_ = 0               # id of the latest Nav2 goal, older results are ignored
        self.goal_sent_time_ = 0.0
        self.goal_timeout_ = 0.0
        self.explore_target_ = None      # np.array([x, y]) of the frontier viewpoint being driven to
        self.visited_goals_ = []         # [(x, y)] reached viewpoints
        self.failed_goals_ = []          # [[x, y, n_failures]] viewpoints Nav2 could not reach / timed out
        self.last_replan_ = -math.inf
        self.progress_pose_ = None       # (x, y, t) last time the robot made progress
        self.explore_start_ = None
        self.last_pose_ = None
        self.distance_travelled_ = 0.0
        self.exploration_log_ = None
        self.exploration_marker_pub_ = self.create_publisher(MarkerArray, 'exploration_markers', 10)

        # [P1] Stuck recovery ("rewind"): the lidar is mounted 1 m high and cannot see low artefacts
        # (green crystals), so Nav2 can drive into them. When stuck, the robot drives back along its
        # own trajectory and a virtual obstacle is added to the Nav2 costmaps (/virtual_obstacles,
        # see nav2_params.yaml) so that the next path goes around it.
        self.trail_ = deque(maxlen=500)  # [(x, y)] recorded trajectory, one point every 0.2 m
        self.rewind_ = None              # dict while rewinding
        self.n_rewinds_ = 0
        self.nav_goal_handle_ = None     # to cancel the running Nav2 goal
        self.virtual_obstacles_ = []     # [(x, y)]
        self.last_obstacle_pub_ = -math.inf
        self.cmd_vel_pub_ = self.create_publisher(Twist, 'cmd_vel', 10)
        # A collision is "forward speed commanded but no movement"; Nav2 failing to plan (no command)
        # must not create a virtual obstacle
        self.cmd_vel_log_ = deque(maxlen=2000)   # [(t, linear.x)] commands received on cmd_vel
        self.cmd_vel_sub_ = self.create_subscription(
            Twist, 'cmd_vel', lambda m: self.cmd_vel_log_.append((self.now_s(), m.linear.x)), 10)
        self.virtual_obstacle_pub_ = self.create_publisher(PointCloud2, 'virtual_obstacles', 1)

        # [P2] Close-range inspection of chosen artefact types (see inspection.py). Classes chosen from the
        # detection logs of 5 exploration runs: alien 99 %, ball 99 %, mushroom 94 % of the localised
        # detections within 1.5 m of a real artefact of that type. 'stone' (green crystals) left out:
        # the robot gets stuck on them.
        q = lambda name, default: self.declare_parameter('inspection.' + name, default).value
        self.inspect_classes_ = [c for c in q('target_classes', ['alien', 'ball', 'mushroom']) if c]
        self.standoff_ = q('standoff', 2.0)                 # [m] distance from the artefact centre
        self.inspect_update_period_ = q('update_period', 1.0)   # [s] re-check the artefact estimate
        self.align_tolerance_ = q('align_tolerance', 0.15)  # [rad] final heading error
        self.inspect_ = None                 # dict while inspecting
        self.last_goal_status_ = None
        self.last_seen_ = {}                 # label -> time it was last detected in the camera image
        self.last_annotated_image_ = None    # latest image with the detection boxes (RGB)
        self.inspection_marker_pub_ = self.create_publisher(MarkerArray, 'inspection_markers', 10)

        self.inspect_visited_radius_ = q('visited_radius', 3.0)
        self.inspect_max_attempts_ = q('max_attempts', 2)
        self.lost_timeout_ = q('lost_timeout', 4.0)
        self.inspect_timeout_margin_ = q('timeout_margin', 30.0)
        self.no_viewpoint_cooldown_ = q('no_viewpoint_cooldown', 5.0)
        self.resume_exploration_ = q('resume_exploration', True)
        self.inspected_ = []
        self.abandoned_ = []
        self.inspect_cooldown_ = []
        self.last_inspect_try_ = -math.inf
        self.inspection_log_ = None

        any_type = ParameterDescriptor(dynamic_typing=True)
        self.degrade_ = self.declare_parameter('degrade', False).value
        self.degrade_type_ = self.declare_parameter('degrade_type', 'both').value
        self.degrade_severity_ = float(self.declare_parameter('degrade_severity', 1.0, any_type).value)
        self.degrade_ramp_time_ = float(self.declare_parameter('degrade_ramp_time', 300.0, any_type).value)
        self.degrade_start_ = None
        self.dust_spots_ = None
        self.dust_alpha_ = None
        self.dust_alpha_count_ = -1
        self.perception_log_ = None
        self.degraded_image_pub_ = self.create_publisher(Image, 'degraded_image', 1)
        self.get_logger().info(
            f'[A1] model {os.path.basename(self.get_parameter("computer_vision_model_filename").value)} | '
            f'degrade {self.degrade_} ({self.degrade_type_}, severity {self.degrade_severity_}, '
            f'ramp {self.degrade_ramp_time_:.0f} s)')

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

        # keep the map to check artifact estimates against walls / unknown space
        self.map_info_ = map_msg.info
        self.map_grid_ = np.array(map_msg.data, dtype=np.int8).reshape(map_height, map_width)

        # self.get_logger().warn('Map received:')
        # self.get_logger().warn(f'  xlim = [{self.xlim_[0]:.2f}, {self.xlim_[1]:.2f}]')
        # self.get_logger().warn(f'  ylim = [{self.ylim_[0]:.2f}, {self.ylim_[1]:.2f}]')

    # Store the camera intrinsics
    def camera_info_callback(self, info_msg):
        self.camera_info_ = info_msg

    def image_callback(self, image_msg, depth_msg):  # + depth_msg (synchronised)
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

        t0 = time.time()
        severity = 0.0
        if self.degrade_:
            severity = self.current_severity()
            image = self.degrade_image(image, severity)
            self.degraded_image_pub_.publish(self.cv_bridge_.cv2_to_imgmsg(image, encoding="rgb8"))
        t1 = time.time()
        detections, labels = self.run_detector(image)
        t2 = time.time()
        self.get_logger().info(f'[A1] severity {severity:.2f} | {len(detections)} detections | '
                               f'degrade {1000 * (t1 - t0):.0f} ms | detector {1000 * (t2 - t1):.0f} ms',
                               throttle_duration_sec=10.0)
        self.log_perception(severity, labels)

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

        annotated = self.draw_detections(image, detections, labels)
        self.image_detections_pub_.publish(self.cv_bridge_.cv2_to_imgmsg(annotated, encoding="rgb8"))

        # [P2] When was each class last seen in the camera, and the latest annotated image
        now = self.now_s()
        for label in labels:
            self.last_seen_[label] = now
        self.last_annotated_image_ = annotated

        if self.artifact_found_:
            self.get_logger().info('Artifact found!')
            # depth image in metres, taken at the same time as the RGB image
            depth = self.cv_bridge_.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
            depth = depth.astype(np.float32) / (1000.0 if depth_msg.encoding == '16UC1' else 1.0)
            self.localise_artifact(detections, labels, image_msg.header, image.shape, depth)  # pass detections

    # Run Classification Model
    def run_detector(self, image):
        """Run the Faster R-CNN detector once, return boxes (x, y, w, h) and class names"""
        with torch.no_grad():
            out = self.computer_vision_model_([TF.to_tensor(image)])[0]
        detections, labels = [], []
        for box, label, score in zip(out['boxes'], out['labels'], out['scores']):
            if score < 0.5:  # CONF from the notebook
                continue
            x1, y1, x2, y2 = box.tolist()
            detections.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            labels.append(self.class_names_[int(label) - 1])
        return detections, labels

    def current_severity(self):
        now = self.now_s()
        if self.degrade_start_ is None:
            self.degrade_start_ = now
        if self.degrade_ramp_time_ <= 0:
            return self.degrade_severity_
        return self.degrade_severity_ * min(1.0, (now - self.degrade_start_) / self.degrade_ramp_time_)

    def degrade_image(self, image, severity):
        if self.degrade_type_ in ('dust', 'both'):
            if self.dust_spots_ is None:
                self.dust_spots_ = dust_spots(image.shape)
            n = int(round(severity * len(self.dust_spots_)))
            if n != self.dust_alpha_count_:
                self.dust_alpha_ = dust_alpha(image.shape, self.dust_spots_[:n])
                self.dust_alpha_count_ = n
        return degrade(image, self.degrade_type_, severity, seed=None, dust_alpha_mask=self.dust_alpha_)

    def log_perception(self, severity, labels):
        try:
            if self.perception_log_ is None:
                os.makedirs(self.log_dir_, exist_ok=True)
                path = os.path.join(self.log_dir_, time.strftime('perception_%Y%m%d_%H%M%S.csv'))
                self.perception_log_ = open(path, 'w')
                self.perception_log_.write('time_s,model,degrade_type,severity,n_detections,labels\n')
                self.get_logger().info(f'Logging detections to {path}')
            model = os.path.basename(self.get_parameter('computer_vision_model_filename').value)
            kind = self.degrade_type_ if self.degrade_ else 'none'
            self.perception_log_.write(f'{self.now_s():.1f},{model},{kind},{severity:.2f},'
                                       f'{len(labels)},{" ".join(labels)}\n')
            self.perception_log_.flush()
        except OSError as ex:
            self.get_logger().warn(f'Cannot write perception log: {ex}', once=True)

    def draw_detections(self, image, detections, labels):
        """Draw boxes and class names on a copy of the image"""
        vis = image.copy()
        for (x, y, w, h), label in zip(detections, labels):
            r, g, b = ARTIFACT_COLOURS.get(label, (1.0, 1.0, 0.0))
            colour = (int(255 * r), int(255 * g), int(255 * b))   # image is RGB
            cv2.rectangle(vis, (x, y), (x + w, y + h), colour, 3)
            cv2.putText(vis, label, (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)
        return vis

    def localise_artifact(self, detections, labels, header, image_shape, depth):
        """
        Compute the location of each detected artifact:
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
        """Point in the camera frame (optical or ROS body convention)"""
        p = PointStamped()
        p.header.frame_id = frame
        if 'optical' in frame:   # optical frame: z forward, x right, y down
            p.point.x, p.point.y, p.point.z = right, down, forward
        else:                    # body frame: x forward, y left, z up
            p.point.x, p.point.y, p.point.z = forward, -right, -down
        return p

    def check_against_map(self, cam, pos):
        """
        Walk along the viewing ray from the camera to the estimate through the SLAM map.
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
        # one colour per class
        self.marker_artifacts_.colors = [ColorRGBA(r=c[0], g=c[1], b=c[2], a=1.0)
                                         for c in (ARTIFACT_COLOURS.get(l, (1.0, 1.0, 0.0))
                                                   for l in self.artifact_labels_)]

        # Create and publish the MarkerArray
        marker_array = MarkerArray()
        # remove old text labels first (number of clusters can shrink when clusters merge)
        clear = Marker()
        clear.header.frame_id = "map"
        clear.ns = "artifact_labels"
        clear.action = Marker.DELETEALL
        clear_status = Marker()
        clear_status.header.frame_id = "map"
        clear_status.ns = "artifact_status"
        clear_status.action = Marker.DELETEALL
        marker_array.markers = [clear, clear_status, self.marker_artifacts_]

        # text label above each artifact
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

            if label not in self.inspect_classes_:
                continue
            status = self.artefact_status(label, (point.x, point.y))
            text.text = f'{label} ({status})'
            ring = Marker()
            ring.header.frame_id = "map"
            ring.ns = "artifact_status"
            ring.id = i
            ring.type = Marker.CYLINDER
            ring.action = Marker.ADD
            ring.pose.position = Point(x=point.x, y=point.y, z=0.05)
            ring.pose.orientation.w = 1.0
            ring.scale.x = ring.scale.y = 2.0
            ring.scale.z = 0.1
            c = INSPECTION_STATUS_COLOURS[status]
            ring.color = ColorRGBA(r=c[0], g=c[1], b=c[2], a=0.6)
            marker_array.markers.append(ring)

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

        # [P1] Sending a new goal while one is running preempts it in Nav2 (no stop in between).
        # Every goal gets an id, so the result of a preempted goal is ignored.
        self.goal_seq_ += 1
        seq = self.goal_seq_
        self.ready_for_next_goal_ = False
        self.goal_sent_time_ = self.now_s()

        # Send goal to action server
        self.get_logger().warn(f'Sending goal [{pose2d.x:.2f}, {pose2d.y:.2f}]...')
        self.send_goal_future_ = self.nav2_action_client_.send_goal_async(
            action_goal,
            feedback_callback=feedback_method)
        self.send_goal_future_.add_done_callback(lambda f: self.goal_response_callback(f, seq))

    def goal_response_callback(self, future, seq):
        """The requested goal pose has been sent to the action server"""

        if seq != self.goal_seq_:   # [P1] already replaced by a newer goal
            return
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Goal rejected')
            self.goal_finished(GoalStatus.STATUS_ABORTED)   # [P1] template got stuck here forever
            return

        # Goal accepted: get result when it's completed
        self.get_logger().warn(f'Goal accepted')
        self.nav_goal_handle_ = goal_handle   # [P1]
        self.get_result_future_ = goal_handle.get_result_async()
        self.get_result_future_.add_done_callback(lambda f: self.goal_reached_callback(f, seq))

    def feedback_callback(self, feedback_msg):
        """Monitor the feedback from the action server"""

        feedback = feedback_msg.feedback

        self.get_logger().info(f'{feedback.distance_remaining:.2f} m remaining')

    def goal_reached_callback(self, future, seq):
        """The requested goal has been reached"""

        if seq != self.goal_seq_:   # [P1] result of a preempted goal
            return
        self.goal_finished(future.result().status)

    def goal_finished(self, status):
        """[P1] Book-keeping when the current Nav2 goal ends"""
        self.last_goal_status_ = status   # [P2]
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info('Goal reached!')
        else:
            self.get_logger().warn(f'Goal ended with status {status}')
        if self.explore_target_ is not None:
            if status == GoalStatus.STATUS_SUCCEEDED:
                self.visited_goals_.append(tuple(self.explore_target_))
            else:
                self.record_failure(self.explore_target_, 1)
            self.explore_target_ = None
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

    def now_s(self):
        """[P1] Current (simulation) time in seconds"""
        return self.get_clock().now().nanoseconds * 1e-9

    def planner_explore_frontiers(self):
        """
        [P1] Frontier-based exploration, called from main_loop at 5 Hz.
          - no goal running: pick the best frontier and send it to Nav2
          - goal running: every replan_period re-evaluate the frontiers; switch (preempt) if the
            target frontier has been mapped in the meantime or another one is now clearly better
          - goal timeout: give up the goal, it will not be selected again
          - robot not moving: rewind along the trajectory + virtual obstacle, then retry
          - no frontier left: exploration complete -> return home
        """
        pose = self.get_pose_2d()
        if pose is None or self.map_grid_ is None:
            return
        now = self.now_s()
        self.update_travel_stats(pose, now)
        self.publish_virtual_obstacles(now)
        if self.rewind_ is not None:
            self.rewind_step(pose, now)
            return

        goal_running = not self.ready_for_next_goal_
        force = False
        if goal_running and self.explore_target_ is not None:
            if now - self.goal_sent_time_ > self.goal_timeout_:
                self.get_logger().warn('Exploration goal timed out -> next frontier')
                force = True
            elif now - self.progress_pose_[2] > self.stuck_time_:
                if self.pushing_forward(self.progress_pose_[2]):
                    self.get_logger().warn(f'Driving forward but no progress for {self.stuck_time_:.0f} s '
                                           f'-> collision with an obstacle the lidar cannot see -> rewind')
                    self.record_failure(self.explore_target_, 1)   # retried once, around the obstacle
                    self.explore_target_ = None
                    self.start_rewind(pose, now)
                    return
                self.get_logger().warn(f'No progress for {self.stuck_time_:.0f} s (Nav2 not driving) '
                                       f'-> next frontier')
                force = True
                self.progress_pose_ = (pose.x, pose.y, now)
            if force:   # a whole timeout was already spent on it: give it up at once
                self.record_failure(self.explore_target_, self.max_attempts_)
                self.explore_target_ = None
        if goal_running and not force and now - self.last_replan_ < self.replan_period_:
            return
        self.last_replan_ = now

        t0 = time.time()
        excluded = [(x, y, self.visited_radius_) for x, y in self.visited_goals_] + \
                   [(x, y, self.failed_radius_) for x, y, n in self.failed_goals_ if n >= self.max_attempts_]
        info = self.map_info_
        result = self.explorer_.compute(
            self.grid_with_virtual_obstacles(), info.resolution, (info.origin.position.x, info.origin.position.y),
            (pose.x, pose.y), pose.theta, excluded=excluded, current_target=self.explore_target_)
        compute_ms = 1000.0 * (time.time() - t0)
        self.publish_exploration_markers(result, pose)
        self.log_exploration(now, result)

        best = result.best
        if best is None:
            if goal_running and self.explore_target_ is not None:
                return   # keep going: the map update may lag behind
            self.planner_type_ = PlannerType.EXPLORATION_DONE
            elapsed = now - self.explore_start_
            self.get_logger().warn(
                f'Exploration complete: {result.free_area_m2:.0f} m2 free space mapped in {elapsed:.0f} s, '
                f'{self.distance_travelled_:.0f} m travelled -> returning home')
            self.publish_exploration_markers(result, pose, done=True)
            self.planner_return_home()
            return

        # Keep the current goal as long as there is still something to explore around it: every
        # preemption makes Nav2 re-plan, and the viewpoint of a large frontier moves while it shrinks
        if goal_running and self.explore_target_ is not None and \
                self.target_still_frontier(result):
            return

        self.get_logger().info(
            f'Frontier selected: ({best.goal[0]:.1f}, {best.goal[1]:.1f}) size {best.size_m:.1f} m, '
            f'path {best.path_dist:.1f} m, score {best.score:.1f} '
            f'({len(result.frontiers)} candidates, {compute_ms:.0f} ms)')
        self.explore_target_ = best.goal
        # Generous timeout: 0.3 m/s average speed + 30 s for recoveries
        self.goal_timeout_ = best.path_dist / 0.3 + 30.0
        # Arrive facing away from the robot, i.e. looking into the unknown space behind the frontier
        theta = math.atan2(best.goal[1] - pose.y, best.goal[0] - pose.x)
        self.planner_go_to_pose2d(Pose2D(x=float(best.goal[0]), y=float(best.goal[1]), theta=theta))

    def record_failure(self, goal, count):
        """[P1] Count Nav2 failures per goal location, the goal is excluded after max_attempts"""
        for entry in self.failed_goals_:
            if math.hypot(entry[0] - goal[0], entry[1] - goal[1]) < self.failed_radius_:
                entry[2] += count
                return
        self.failed_goals_.append([float(goal[0]), float(goal[1]), count])

    def target_still_frontier(self, result):
        """[P1] Are there still frontier cells near the goal we are driving to?"""
        radius = 2.0 * self.explorer_.viewpoint_radius
        for f in result.frontiers + result.rejected:
            if np.min(np.hypot(f.cells[:, 0] - self.explore_target_[0],
                               f.cells[:, 1] - self.explore_target_[1])) < radius:
                return True
        return False

    def update_travel_stats(self, pose, now):
        """[P1] Distance travelled and progress (for stuck detection)"""
        if self.explore_start_ is None:
            self.explore_start_ = now
        if self.last_pose_ is not None:
            self.distance_travelled_ += math.hypot(pose.x - self.last_pose_.x, pose.y - self.last_pose_.y)
        self.last_pose_ = pose
        if self.progress_pose_ is None or \
                math.hypot(pose.x - self.progress_pose_[0], pose.y - self.progress_pose_[1]) > 0.5:
            self.progress_pose_ = (pose.x, pose.y, now)
        if not self.trail_ or math.hypot(pose.x - self.trail_[-1][0], pose.y - self.trail_[-1][1]) > 0.2:
            self.trail_.append((pose.x, pose.y))

    def finish_rewind(self, pose, now, outcome):
        """[P1] Stop the robot and hand back to the exploration"""
        self.cmd_vel_pub_.publish(Twist())
        if outcome is None:
            self.get_logger().error('Rewind failed: the robot could not free itself')
        else:
            self.get_logger().info(f'Rewind done ({outcome}), resuming exploration')
        self.rewind_ = None
        self.ready_for_next_goal_ = True
        self.last_replan_ = -math.inf
        self.progress_pose_ = (pose.x, pose.y, now)

    def pushing_forward(self, since):
        """[P1] Was a forward speed commanded most of the time since `since`?"""
        cmds = [v for t, v in self.cmd_vel_log_ if t >= since]
        return len(cmds) >= 10 and sum(v > 0.15 for v in cmds) > 0.5 * len(cmds)

    # [P1] Tried one after the other (linear m/s, angular rad/s) if driving back along the trail fails:
    # in tests the robot caught on a crystal by its side could not reverse straight but got free when turning
    ESCAPE_MANOEUVRES = [(-0.3, 0.6), (-0.3, -0.6), (0.0, 0.8), (0.0, -0.8), (0.3, 0.6), (0.3, -0.6)]
    ESCAPE_DURATION = 4.0   # [s] per manoeuvre
    ESCAPE_DISTANCE = 0.6   # [m] the robot is free once it moved this far

    def start_rewind(self, pose, now):
        """[P1] Stuck: virtual obstacle in front of the robot, cancel Nav2, drive back along the trail"""
        # The robot was commanded forward (pushing_forward), so the obstacle is in front of it
        heading = pose.theta
        obstacle = (pose.x + self.obstacle_offset_ * math.cos(heading),
                    pose.y + self.obstacle_offset_ * math.sin(heading))
        if all(math.hypot(obstacle[0] - x, obstacle[1] - y) > 0.5 for x, y in self.virtual_obstacles_):
            self.virtual_obstacles_.append(obstacle)
            self.last_obstacle_pub_ = -math.inf   # publish it right away

        # Cancel the Nav2 goal, its result is ignored (new goal id)
        self.goal_seq_ += 1
        if self.nav_goal_handle_ is not None:
            self.nav_goal_handle_.cancel_goal_async()
            self.nav_goal_handle_ = None

        # Waypoints: the recorded trajectory, backwards, over rewind_distance
        waypoints, travelled, prev = [], 0.0, (pose.x, pose.y)
        for x, y in reversed(self.trail_):
            travelled += math.hypot(x - prev[0], y - prev[1])
            prev = (x, y)
            waypoints.append((x, y))
            if travelled >= self.rewind_distance_:
                break
        if travelled < 0.5 * self.rewind_distance_:   # no trajectory recorded: straight back
            waypoints.append((pose.x - self.rewind_distance_ * math.cos(heading),
                              pose.y - self.rewind_distance_ * math.sin(heading)))
        self.rewind_ = {'waypoints': waypoints, 'idx': 0, 'start': now, 'start_xy': (pose.x, pose.y),
                        'phase': 'trail', 'manoeuvre': 0, 'phase_start': now}
        self.n_rewinds_ += 1
        self.get_logger().warn(f'Rewind #{self.n_rewinds_}: virtual obstacle at ({obstacle[0]:.1f}, '
                               f'{obstacle[1]:.1f}), driving {self.rewind_distance_:.1f} m back along the trajectory')

    def rewind_step(self, pose, now):
        """
        [P1] 1) follow the trail waypoints back with whichever end of the robot faces them (usually the rear)
             2) if that does not free the robot within rewind_timeout: escape manoeuvres, one after the other
        """
        rw = self.rewind_
        wps = rw['waypoints']
        moved = math.hypot(pose.x - rw['start_xy'][0], pose.y - rw['start_xy'][1])
        cmd = Twist()
        if rw['phase'] == 'escape':
            if moved >= self.ESCAPE_DISTANCE:
                self.finish_rewind(pose, now, f'free after escape manoeuvre {rw["manoeuvre"] + 1}')
                return
            if now - rw['phase_start'] > self.ESCAPE_DURATION:
                rw['manoeuvre'] += 1
                rw['phase_start'] = now
            if rw['manoeuvre'] >= len(self.ESCAPE_MANOEUVRES):
                self.finish_rewind(pose, now, None)
                return
            cmd.linear.x, cmd.angular.z = self.ESCAPE_MANOEUVRES[rw['manoeuvre']]
            self.cmd_vel_pub_.publish(cmd)
            return

        while rw['idx'] < len(wps) and \
                math.hypot(wps[rw['idx']][0] - pose.x, wps[rw['idx']][1] - pose.y) < 0.3:
            rw['idx'] += 1
        if rw['idx'] >= len(wps):
            self.finish_rewind(pose, now, f'{moved:.1f} m back along the trajectory')
            return
        if now - rw['phase_start'] > self.rewind_timeout_:
            if moved >= self.ESCAPE_DISTANCE:
                self.finish_rewind(pose, now, f'{moved:.1f} m back along the trajectory')
            else:
                self.get_logger().warn('Rewind along the trajectory did not free the robot -> escape manoeuvres')
                rw['phase'], rw['phase_start'] = 'escape', now
            return
        tx, ty = wps[rw['idx']]
        bearing = math.atan2(ty - pose.y, tx - pose.x)
        err_front = math.atan2(math.sin(bearing - pose.theta), math.cos(bearing - pose.theta))
        err_rear = math.atan2(math.sin(bearing - pose.theta - math.pi), math.cos(bearing - pose.theta - math.pi))
        direction, err = (-1.0, err_rear) if abs(err_rear) <= abs(err_front) else (1.0, err_front)
        cmd.linear.x = direction * self.rewind_speed_ * max(0.2, math.cos(err))
        cmd.angular.z = max(-0.6, min(0.6, 1.0 * err))
        self.cmd_vel_pub_.publish(cmd)

    def publish_virtual_obstacles(self, now):
        """[P1] Virtual obstacles as a point cloud (a disc of points each), re-published every second
        because Nav2's clear-costmap recoveries wipe the obstacle layer"""
        if now - self.last_obstacle_pub_ < 1.0:
            return   # (an empty cloud is published too: it removes obstacles left by a previous run)
        self.last_obstacle_pub_ = now
        step, r = 0.15, self.obstacle_radius_
        offsets = [(dx, dy) for dx in np.arange(-r, r + 1e-6, step) for dy in np.arange(-r, r + 1e-6, step)
                   if dx * dx + dy * dy <= r * r]
        points = [(float(ox + dx), float(oy + dy), 0.5) for ox, oy in self.virtual_obstacles_ for dx, dy in offsets]
        header = Header(frame_id='map', stamp=self.get_clock().now().to_msg())
        self.virtual_obstacle_pub_.publish(point_cloud2.create_cloud_xyz32(header, points))

    def grid_with_virtual_obstacles(self):
        """[P1] SLAM grid with the virtual obstacles marked occupied, for the frontier planner"""
        if not self.virtual_obstacles_:
            return self.map_grid_
        grid = self.map_grid_.copy()
        info = self.map_info_
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        rows, cols = np.ogrid[:grid.shape[0], :grid.shape[1]]
        for x, y in self.virtual_obstacles_:
            r, c = (y - oy) / res, (x - ox) / res
            grid[(rows - r) ** 2 + (cols - c) ** 2 <= (self.obstacle_radius_ / res) ** 2] = 100
        return grid

    def log_exploration(self, now, result):
        """[P1] Exploration progress as CSV (for the report: mapped area over time)"""
        try:
            if self.exploration_log_ is None:
                os.makedirs(self.log_dir_, exist_ok=True)
                path = os.path.join(self.log_dir_, time.strftime('exploration_%Y%m%d_%H%M%S.csv'))
                self.exploration_log_ = open(path, 'w')
                self.exploration_log_.write('time_s,known_area_m2,free_area_m2,n_frontiers,'
                                            'distance_travelled_m,n_visited,n_failed\n')
                self.get_logger().info(f'Logging exploration progress to {path}')
            self.exploration_log_.write(
                f'{now - self.explore_start_:.1f},{result.known_area_m2:.1f},{result.free_area_m2:.1f},'
                f'{len(result.frontiers)},{self.distance_travelled_:.1f},'
                f'{len(self.visited_goals_)},{sum(n >= self.max_attempts_ for *_, n in self.failed_goals_)}\n')
            self.exploration_log_.flush()
        except OSError as ex:
            self.get_logger().warn(f'Cannot write exploration log: {ex}', once=True)

    def publish_exploration_markers(self, result, pose, done=False):
        """
        [P1] RViz overlay of the exploration:
          frontier cells (one colour per cluster), candidate viewpoints with their score,
          chosen frontier + estimated path, visited (grey) and failed (red) goals, status text
        """
        stamp = self.get_clock().now().to_msg()

        def marker(ns, mid, mtype, scale, rgba, z=0.0):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.z = z
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            return m

        palette = [(0.0, 0.8, 1.0), (1.0, 0.8, 0.0), (0.6, 0.3, 1.0), (0.0, 1.0, 0.5),
                   (1.0, 0.4, 0.7), (0.5, 1.0, 0.0), (1.0, 0.5, 0.0), (0.3, 0.5, 1.0)]
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        markers = [clear]
        res = self.map_info_.resolution

        cells = marker('frontier_cells', 0, Marker.CUBE_LIST, res, (1.0, 1.0, 1.0, 0.9), z=0.05)
        for i, f in enumerate(result.frontiers + result.rejected):
            c = palette[i % len(palette)] if i < len(result.frontiers) else (0.5, 0.5, 0.5)
            for x, y in f.cells:
                cells.points.append(Point(x=float(x), y=float(y)))
                cells.colors.append(ColorRGBA(r=c[0], g=c[1], b=c[2], a=0.9))
        markers.append(cells)

        for i, f in enumerate(result.frontiers):
            colour = (0.0, 1.0, 0.0, 1.0) if i == 0 else (1.0, 1.0, 1.0, 0.8)
            sphere = marker('frontier_candidates', i, Marker.SPHERE, 0.8 if i == 0 else 0.4, colour, z=0.3)
            sphere.pose.position.x, sphere.pose.position.y = float(f.goal[0]), float(f.goal[1])
            text = marker('frontier_scores', i, Marker.TEXT_VIEW_FACING, 0.5, colour, z=1.2)
            text.pose.position.x, text.pose.position.y = float(f.goal[0]), float(f.goal[1])
            text.text = f'{f.score:.1f}'
            markers += [sphere, text]

        if result.path is not None:
            line = marker('frontier_path', 0, Marker.LINE_STRIP, 0.1, (0.0, 1.0, 0.0, 0.8), z=0.2)
            line.points = [Point(x=float(x), y=float(y)) for x, y in result.path]
            markers.append(line)

        failed = [(x, y) for x, y, n in self.failed_goals_ if n >= self.max_attempts_]
        for ns, goals, colour in (('visited_goals', self.visited_goals_, (0.6, 0.6, 0.6, 0.8)),
                                  ('failed_goals', failed, (1.0, 0.0, 0.0, 0.8))):
            if goals:
                m = marker(ns, 0, Marker.CUBE_LIST, 0.4, colour, z=0.2)
                m.points = [Point(x=float(x), y=float(y)) for x, y in goals]
                markers.append(m)

        for i, (x, y) in enumerate(self.virtual_obstacles_):
            m = marker('virtual_obstacles', i, Marker.CYLINDER, 2.0 * self.obstacle_radius_, (1.0, 0.2, 0.0, 0.6), z=0.5)
            m.scale.z = 1.0
            m.pose.position.x, m.pose.position.y = float(x), float(y)
            markers.append(m)

        status = marker('exploration_status', 0, Marker.TEXT_VIEW_FACING, 0.8, (1.0, 1.0, 1.0, 1.0), z=2.5)
        status.pose.position.x, status.pose.position.y = pose.x, pose.y
        status.text = (f'EXPLORATION COMPLETE  {result.free_area_m2:.0f} m2' if done else
                       f'{len(result.frontiers)} frontiers  {result.free_area_m2:.0f} m2  '
                       f'{self.distance_travelled_:.0f} m')
        markers.append(status)

        self.exploration_marker_pub_.publish(MarkerArray(markers=markers))

    # ------------------------------------------------------------------ [P2] close-range inspection
    def start_inspection(self):
        """
        [P2] Called while exploring: if Perception 3 has localised an artefact of a chosen class
        (DBSCAN cluster = at least 3 consistent observations) and a viewpoint can be computed,
        pause the exploration and switch to INSPECT_ARTEFACT
        """
        if self.map_grid_ is None:
            return False
        now = self.now_s()
        if now - self.last_inspect_try_ < self.inspect_update_period_:
            return False
        pose = self.get_pose_2d()
        if pose is None:
            return False
        self.inspect_cooldown_ = [c for c in self.inspect_cooldown_ if c[3] > now]
        targets = [(label, np.array([loc.x, loc.y]))
                   for loc, label in zip(self.artifact_locations_, self.artifact_labels_)
                   if label in self.inspect_classes_
                   and self.artefact_status(label, (loc.x, loc.y)) == 'pending'
                   and not any(l == label and math.hypot(loc.x - x, loc.y - y) < self.inspect_visited_radius_
                               for l, x, y, _ in self.inspect_cooldown_)]
        if not targets:
            return False
        self.last_inspect_try_ = now
        label, xy = min(targets, key=lambda t: math.hypot(t[1][0] - pose.x, t[1][1] - pose.y))
        self.inspect_ = {'label': label, 'xy': xy, 'phase': 'navigate', 'viewpoint': None, 'candidates': [],
                         'failed': [], 'last_update': now, 'start': now, 'attempt': 1,
                         'deadline': math.inf, 'last_ok': now,
                         'ticks': 0, 'seen_ticks': 0, 'align_start': None}
        if not self.send_viewpoint(pose):
            self.inspect_cooldown_.append((label, float(xy[0]), float(xy[1]), now + self.no_viewpoint_cooldown_))
            self.get_logger().info(f'{label} at ({xy[0]:.1f}, {xy[1]:.1f}): no inspection viewpoint yet '
                                   f'(surroundings not mapped), exploring on', throttle_duration_sec=5.0)
            return False
        self.get_logger().warn(f'{label} detected at ({xy[0]:.1f}, {xy[1]:.1f}) -> exploration paused, '
                               f'close-range inspection')
        self.planner_type_ = PlannerType.INSPECT_ARTEFACT
        self.explore_target_ = None   # the exploration goal is preempted, not failed
        self.progress_pose_ = (pose.x, pose.y, now)
        self.publish_artifact_markers()
        return True

    def send_viewpoint(self, pose):
        """[P2] Choose the best viewpoint around the artefact and send it to Nav2"""
        insp = self.inspect_
        info = self.map_info_
        best, candidates = choose_viewpoint(
            self.grid_with_virtual_obstacles(), info.resolution, (info.origin.position.x, info.origin.position.y),
            (pose.x, pose.y), insp['xy'], standoff=self.standoff_, min_clearance=self.explorer_.min_clearance,
            excluded=insp['failed'])
        insp['candidates'] = candidates
        insp['viewpoint'] = best
        self.publish_inspection_markers(pose)
        if best is None:
            return False
        if not math.isfinite(insp['deadline']):
            insp['deadline'] = self.now_s() + best.path_dist / 0.3 + self.inspect_timeout_margin_
        self.get_logger().info(f'Inspection viewpoint ({best.xy[0]:.1f}, {best.xy[1]:.1f}), '
                               f'{sum(c.valid for c in candidates)}/{len(candidates)} candidates valid, '
                               f'path {best.path_dist:.1f} m')
        self.planner_go_to_pose2d(Pose2D(x=float(best.xy[0]), y=float(best.xy[1]), theta=best.yaw))
        return True

    def closest_cluster(self, label, xy, max_dist=2.0):
        """[P2] Current Perception 3 estimate of the artefact we are inspecting (clusters are recomputed
        at every detection, so the artefact is tracked by class + proximity)"""
        best, best_d = None, max_dist
        for loc, lab in zip(self.artifact_locations_, self.artifact_labels_):
            d = math.hypot(loc.x - xy[0], loc.y - xy[1])
            if lab == label and d < best_d:
                best, best_d = np.array([loc.x, loc.y]), d
        return best

    def artefact_status(self, label, xy):
        for status, records in (('inspected', self.inspected_), ('abandoned', self.abandoned_)):
            if any(l == label and math.hypot(x - xy[0], y - xy[1]) < self.inspect_visited_radius_
                   for l, x, y in records):
                return status
        if self.planner_type_ == PlannerType.INSPECT_ARTEFACT and self.inspect_ is not None and \
                self.inspect_['label'] == label and \
                math.hypot(self.inspect_['xy'][0] - xy[0], self.inspect_['xy'][1] - xy[1]) < self.inspect_visited_radius_:
            return 'inspecting'
        return 'pending'

    def camera_half_fov(self):
        if self.camera_info_ is not None and self.camera_info_.k[0] > 0.0:
            return math.atan(self.camera_info_.width / (2.0 * self.camera_info_.k[0]))
        return self.get_parameter('camera_hfov').value / 2.0

    def artefact_in_view(self, pose, xy):
        to_art = np.asarray(xy) - np.array([pose.x, pose.y])
        dist = float(np.hypot(*to_art))
        err = wrap_pi(math.atan2(to_art[1], to_art[0]) - pose.theta)
        if dist > self.standoff_ + 2.5 or abs(err) > self.camera_half_fov() - 0.15:
            return False
        info = self.map_info_
        res, ox, oy = info.resolution, info.origin.position.x, info.origin.position.y
        cell = lambda p: (int((p[1] - oy) / res), int((p[0] - ox) / res))
        return line_of_sight(self.map_grid_ >= OCCUPIED_THRESHOLD, cell((pose.x, pose.y)), cell(xy),
                             ignore_radius=1.0 / res)

    def planner_inspect_artefact(self):
        """
        [P2] 'navigate': Nav2 drives to the viewpoint; the artefact estimate is refreshed and the viewpoint
                         re-planned if the estimate moves
             'align':    rotate on the spot until the camera faces the artefact, creep closer if Nav2
                         stopped early (goal tolerance is 1 m), done once the artefact is detected
             'rewind':   collision while approaching, rewind then retry
        """
        pose = self.get_pose_2d()
        if pose is None:
            return
        now = self.now_s()
        insp = self.inspect_
        self.update_travel_stats(pose, now)
        self.publish_virtual_obstacles(now)
        if self.rewind_ is not None:
            self.rewind_step(pose, now)
            return
        if insp['phase'] == 'rewind':
            self.retry_or_abandon(pose, now, 'collision', exclude_viewpoint=True)
            return

        to_art = insp['xy'] - np.array([pose.x, pose.y])
        dist = float(np.hypot(*to_art))
        err = wrap_pi(math.atan2(to_art[1], to_art[0]) - pose.theta)
        seen = now - self.last_seen_.get(insp['label'], -math.inf) < 1.5

        if seen or not self.artefact_in_view(pose, insp['xy']):
            insp['last_ok'] = now
        elif now - insp['last_ok'] > self.lost_timeout_:
            self.retry_or_abandon(pose, now, f'artefact in view but not detected for {self.lost_timeout_:.0f} s')
            return

        if insp['phase'] == 'navigate':
            insp['ticks'] += 1
            if seen:
                insp['seen_ticks'] += 1
            if now > insp['deadline']:
                self.retry_or_abandon(pose, now, 'timeout', exclude_viewpoint=True)
                return
            if not self.ready_for_next_goal_:   # Nav2 still driving
                if now - self.progress_pose_[2] > self.stuck_time_:
                    if self.pushing_forward(self.progress_pose_[2]):
                        self.get_logger().warn('Inspection: driving forward but no progress -> rewind')
                        insp['phase'] = 'rewind'
                        self.start_rewind(pose, now)
                    else:
                        self.retry_or_abandon(pose, now, 'no progress', exclude_viewpoint=True)
                    return
                if now - insp['last_update'] >= self.inspect_update_period_:
                    insp['last_update'] = now
                    new = self.closest_cluster(insp['label'], insp['xy'])
                    if new is not None and np.hypot(*(new - insp['xy'])) > 0.5:
                        self.get_logger().info(f'Artefact estimate moved to ({new[0]:.1f}, {new[1]:.1f}) '
                                               f'-> new viewpoint')
                        insp['xy'] = new
                        if not self.send_viewpoint(pose):
                            self.retry_or_abandon(pose, now, 'no viewpoint for the new estimate')
                    else:
                        self.publish_inspection_markers(pose)
                return
            if self.last_goal_status_ == GoalStatus.STATUS_SUCCEEDED:
                insp['phase'], insp['align_start'] = 'align', now
                self.get_logger().info(f'Viewpoint reached ({dist:.1f} m from the artefact) -> aligning')
                self.publish_inspection_markers(pose)
                return
            self.retry_or_abandon(pose, now, 'Nav2 could not reach the viewpoint', exclude_viewpoint=True)
            return

        # 'align'
        cmd = Twist()
        if abs(err) > self.align_tolerance_:
            cmd.angular.z = max(-0.4, min(0.4, 1.0 * err))
        elif dist > self.standoff_ + 0.3:
            cmd.linear.x = 0.2   # facing the artefact, free line of sight checked by the viewpoint choice
            cmd.angular.z = 0.5 * err
        elif now - self.last_seen_.get(insp['label'], -math.inf) < 2.0:
            self.finish_inspection(pose, now, success=True)
            return
        if now - insp['align_start'] > 30.0:
            self.retry_or_abandon(pose, now, 'alignment timeout')
            return
        self.cmd_vel_pub_.publish(cmd)

    def stop_navigation(self):
        self.goal_seq_ += 1
        if self.nav_goal_handle_ is not None:
            self.nav_goal_handle_.cancel_goal_async()
            self.nav_goal_handle_ = None
        self.cmd_vel_pub_.publish(Twist())
        self.ready_for_next_goal_ = True

    def retry_or_abandon(self, pose, now, reason, exclude_viewpoint=False):
        insp = self.inspect_
        self.stop_navigation()
        if insp['attempt'] >= self.inspect_max_attempts_:
            self.finish_inspection(pose, now, success=False, reason=reason)
            return
        insp['attempt'] += 1
        if exclude_viewpoint and insp['viewpoint'] is not None:
            insp['failed'].append(tuple(insp['viewpoint'].xy))
        new = self.closest_cluster(insp['label'], insp['xy'])
        if new is not None:
            insp['xy'] = new
        insp.update(phase='navigate', deadline=math.inf, last_ok=now, last_update=now, align_start=None)
        self.progress_pose_ = (pose.x, pose.y, now)
        self.get_logger().warn(f'Inspection of {insp["label"]}: {reason} -> retry '
                               f'({insp["attempt"]}/{self.inspect_max_attempts_})')
        if not self.send_viewpoint(pose):
            self.finish_inspection(pose, now, success=False, reason=f'{reason}, no reachable viewpoint left')

    def finish_inspection(self, pose, now, success, reason=''):
        """[P2] End of the inspection: report, save the close-range image, record the artefact"""
        self.stop_navigation()
        insp = self.inspect_
        insp['phase'] = 'done' if success else 'abandoned'
        to_art = insp['xy'] - np.array([pose.x, pose.y])
        dist = float(np.hypot(*to_art))
        err = wrap_pi(math.atan2(to_art[1], to_art[0]) - pose.theta)
        in_view = 100.0 * insp['seen_ticks'] / max(1, insp['ticks'])
        record = (insp['label'], float(insp['xy'][0]), float(insp['xy'][1]))
        if not success:
            self.abandoned_.append(record)
            self.get_logger().error(f'Inspection of {insp["label"]} abandoned after {insp["attempt"]} '
                                    f'attempt(s): {reason}')
        else:
            self.inspected_.append(record)
            self.get_logger().warn(
                f'Inspection of {insp["label"]} done in {now - insp["start"]:.0f} s: {dist:.1f} m from the '
                f'artefact, camera {abs(math.degrees(err)):.0f} deg off its centre, '
                f'detected during {in_view:.0f} % of the approach')
            if self.last_annotated_image_ is not None:
                try:
                    os.makedirs(self.log_dir_, exist_ok=True)
                    path = os.path.join(self.log_dir_, time.strftime(f'inspection_{insp["label"]}_%H%M%S.png'))
                    cv2.imwrite(path, cv2.cvtColor(self.last_annotated_image_, cv2.COLOR_RGB2BGR))
                    self.get_logger().info(f'Close-range image saved to {path}')
                except (OSError, cv2.error) as ex:
                    self.get_logger().warn(f'Cannot save the inspection image: {ex}')
        self.log_inspection(now, insp, success, reason, dist, err, in_view)

        if self.resume_exploration_:
            self.resume_exploration(pose, now)
            clear = Marker()
            clear.header.frame_id = 'map'
            clear.action = Marker.DELETEALL
            self.inspection_marker_pub_.publish(MarkerArray(markers=[clear]))
        else:
            self.planner_type_ = PlannerType.INSPECTION_DONE
            self.publish_inspection_markers(pose)
        self.publish_artifact_markers()
        pending = sum(lab in self.inspect_classes_ and self.artefact_status(lab, (loc.x, loc.y)) == 'pending'
                      for loc, lab in zip(self.artifact_locations_, self.artifact_labels_))
        self.get_logger().warn(f'Artefacts: {len(self.inspected_)} inspected, {len(self.abandoned_)} abandoned, '
                               f'{pending} pending')

    def resume_exploration(self, pose, now):
        self.get_logger().warn('Resuming exploration')
        self.planner_type_ = PlannerType.EXPLORE_FRONTIERS
        self.explore_target_ = None
        self.ready_for_next_goal_ = True
        self.last_replan_ = -math.inf
        self.progress_pose_ = (pose.x, pose.y, now)
        self.last_inspect_try_ = now

    def log_inspection(self, now, insp, success, reason, dist, err, in_view):
        try:
            if self.inspection_log_ is None:
                os.makedirs(self.log_dir_, exist_ok=True)
                path = os.path.join(self.log_dir_, time.strftime('inspections_%Y%m%d_%H%M%S.csv'))
                self.inspection_log_ = open(path, 'w')
                self.inspection_log_.write('time_s,label,x,y,result,reason,attempts,duration_s,'
                                           'distance_m,heading_error_deg,detected_pct\n')
                self.get_logger().info(f'Logging inspections to {path}')
            start = self.explore_start_ if self.explore_start_ is not None else now
            self.inspection_log_.write(
                f'{now - start:.1f},{insp["label"]},{insp["xy"][0]:.2f},{insp["xy"][1]:.2f},'
                f'{"inspected" if success else "abandoned"},{reason.replace(",", ";")},{insp["attempt"]},'
                f'{now - insp["start"]:.1f},{dist:.2f},{math.degrees(err):.1f},{in_view:.0f}\n')
            self.inspection_log_.flush()
        except OSError as ex:
            self.get_logger().warn(f'Cannot write inspection log: {ex}', once=True)

    def publish_inspection_markers(self, pose):
        """[P2] RViz: inspected artefact, standoff circle, candidate viewpoints (green valid / red rejected),
        chosen viewpoint (arrow = camera heading), state text"""
        insp = self.inspect_
        stamp = self.get_clock().now().to_msg()

        def marker(ns, mtype, scale, rgba, mid=0, z=0.0):
            m = Marker()
            m.header.frame_id, m.header.stamp = 'map', stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            m.pose.position.z = z
            m.scale.x = m.scale.y = m.scale.z = scale
            m.color = ColorRGBA(r=rgba[0], g=rgba[1], b=rgba[2], a=rgba[3])
            return m

        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        markers = [clear]
        ax, ay = float(insp['xy'][0]), float(insp['xy'][1])

        target = marker('inspection_target', Marker.SPHERE, 0.7, (1.0, 1.0, 0.0, 0.9), z=0.5)
        target.pose.position.x, target.pose.position.y = ax, ay
        circle = marker('inspection_standoff', Marker.LINE_STRIP, 0.05, (1.0, 1.0, 0.0, 0.6), z=0.1)
        circle.points = [Point(x=ax + self.standoff_ * math.cos(a), y=ay + self.standoff_ * math.sin(a))
                         for a in np.linspace(0.0, 2.0 * math.pi, 49)]
        cands = marker('inspection_candidates', Marker.SPHERE_LIST, 0.25, (1.0, 1.0, 1.0, 1.0), z=0.2)
        for c in insp['candidates']:
            cands.points.append(Point(x=float(c.xy[0]), y=float(c.xy[1])))
            cands.colors.append(ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.9) if c.valid else
                                ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6))
        markers += [target, circle, cands]

        vp = insp['viewpoint']
        if vp is not None:
            arrow = marker('inspection_viewpoint', Marker.ARROW, 1.0, (1.0, 0.0, 1.0, 1.0), z=0.3)
            arrow.scale.y = arrow.scale.z = 0.2
            arrow.pose.position.x, arrow.pose.position.y = float(vp.xy[0]), float(vp.xy[1])
            arrow.pose.orientation.z, arrow.pose.orientation.w = math.sin(vp.yaw / 2.0), math.cos(vp.yaw / 2.0)
            markers.append(arrow)

        text = marker('inspection_status', Marker.TEXT_VIEW_FACING, 0.6, (1.0, 1.0, 0.0, 1.0), z=2.0)
        text.pose.position.x, text.pose.position.y = ax, ay
        text.text = f'INSPECT {insp["label"]}: {insp["phase"]} ({insp["attempt"]}/{self.inspect_max_attempts_})'
        markers.append(text)
        self.inspection_marker_pub_.publish(MarkerArray(markers=markers))

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

        # [P1] Exploration runs continuously (re-plans while driving), not only when a goal is finished
        if self.planner_type_ == PlannerType.EXPLORE_FRONTIERS:
            if self.rewind_ is None and self.start_inspection():   # [P2] chosen artefact detected
                return
            self.planner_explore_frontiers()
            return
        if self.planner_type_ == PlannerType.INSPECT_ARTEFACT:      # [P2]
            self.planner_inspect_artefact()
            return
        if self.planner_type_ in (PlannerType.EXPLORATION_DONE, PlannerType.INSPECTION_DONE):
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