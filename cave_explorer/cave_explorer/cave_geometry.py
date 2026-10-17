#!/usr/bin/env python3
"""
Advanced 2: Cave geometry analysis

Separate ROS node that analyses the SLAM map online:
  1. Distance transform (DT) of the explored free space, recomputed whenever a new map arrives
  2. Ridges (medial axis) = cells where the DT gradient |grad D| collapses (the DT has slope 1
     everywhere except on the ridge, where the two walls are equally far away)
  3. Wide areas      = local maxima of D  (largest inscribed circles)
  4. Narrow passages = local minima of D *along the ridge* (saddle points of D:
     minimum along the passage, maximum across it). The crossing direction is the
     Hessian eigenvector with the most negative eigenvalue.
     A passage is a "chokepoint" if cutting it splits the accessible space (only connection).
     Bottleneck = narrowest chokepoint.
  5. Results are published as an OccupancyGrid (distance transform) + MarkerArray for RViz,
     printed to the log and saved as an image for the report.
"""

import math
import os
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# Helper Class
def non_max_suppression(candidates, min_dist_px, smallest_first):
    """candidates: list of (value, row, col). Keep the best one, drop others closer than min_dist_px."""
    candidates = sorted(candidates, key=lambda c: c[0], reverse=not smallest_first)
    kept = []
    for v, r, c in candidates:
        if all((r - kr) ** 2 + (c - kc) ** 2 >= min_dist_px ** 2 for _, kr, kc in kept):
            kept.append((v, r, c))
    return kept


def analyse_map(grid, res, robot_rc=None,
                robot_radius=0.4, narrow_max_width=3.0, wide_min_width=5.0,
                ridge_grad_thresh=0.6, nms_dist=3.0, max_features=5):
    """
    grid: int8 occupancy grid (rows = y, cols = x), -1 unknown, 0..100 occupancy
    res:  metres per cell
    robot_rc: (row, col) of the robot, used to keep only the accessible free space
    Returns a dict with the distance transform (metres) and the detected features.
    """
    free = (grid >= 0) & (grid < 25)
    occupied = grid >= 65

    # Fill tiny unobserved holes inside free space (SLAM speckle)
    free_u8 = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    free_u8[occupied] = 0

    # Distance transform of the free space: distance to nearest wall OR unknown cell (conservative)
    dt_free = cv2.distanceTransform(free_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # Distance to nearest wall only (unknown counts as open) -> tells if a boundary is a real wall
    dt_wall = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    # Accessible region: free space where the robot fits, connected to the robot
    robot_r_px = robot_radius / res
    fits = (dt_free > robot_r_px).astype(np.uint8)
    n, comp = cv2.connectedComponents(fits, connectivity=8)
    if robot_rc is not None and 0 <= robot_rc[0] < grid.shape[0] and 0 <= robot_rc[1] < grid.shape[1] \
            and comp[robot_rc] > 0:
        accessible = comp == comp[robot_rc]
    else:
        accessible = fits.astype(bool)
    # also include the clearance band around the accessible cells
    accessible_area = cv2.dilate(accessible.astype(np.uint8),
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                           (2 * int(robot_r_px) + 1,) * 2)).astype(bool) & (free_u8 > 0)

    # Gradient (Jacobian) and Hessian of the smoothed distance transform
    D = cv2.GaussianBlur(dt_free, (0, 0), 1.0)
    gy, gx = np.gradient(D)                 # rows = y, cols = x
    grad_mag = np.hypot(gx, gy)
    gyy, gyx = np.gradient(gy)
    gxy, gxx = np.gradient(gx)

    # Ridge / medial axis: gradient collapses, away from the walls
    ridge = accessible & (grad_mag < ridge_grad_thresh) & (dt_free > robot_r_px)

    result = {'dt': dt_free * res, 'accessible': accessible_area, 'ridge': ridge,
              'widest': None, 'wide': [], 'bottleneck': None, 'narrow': []}
    if not accessible.any():
        return result

    nms_px = nms_dist / res

    # Widest accessible area: global maximum of D (largest inscribed circle)
    d_acc = np.where(accessible, dt_free, 0)
    r, c = np.unravel_index(np.argmax(d_acc), d_acc.shape)
    result['widest'] = {'rc': (r, c), 'width': 2 * dt_free[r, c] * res}

    # Other wide areas: local maxima of D
    k = max(3, int(nms_px) | 1)
    local_max = (d_acc >= cv2.dilate(d_acc, np.ones((k, k), np.uint8))) & accessible
    cand = sorted(((dt_free[r, c], r, c) for r, c in zip(*np.nonzero(local_max))
                   if 2 * dt_free[r, c] * res >= wide_min_width), reverse=True)
    wide = []
    for v, r, c in cand:
        # skip maxima that lie inside the inscribed circle of an already found wider area
        if all(math.hypot(r - kr, c - kc) >= max(kv, nms_px) for kv, kr, kc in wide):
            wide.append((v, r, c))
    result['wide'] = [{'rc': (r, c), 'width': 2 * v * res} for v, r, c in wide[:max_features]]

    # Narrow passages: minima of D along the ridge (saddle points)
    # Each connected piece of ridge that is narrower than narrow_max_width is one passage.
    # Its narrowest point (middle of the narrowest stretch) is reported.
    narrow_ridge = ridge & (2 * dt_free * res <= narrow_max_width) & (dt_wall <= dt_free + 2)
    # (dt_wall <= dt_free + 2: passage is bounded by real walls, not by unexplored space)
    n, labels = cv2.connectedComponents(narrow_ridge.astype(np.uint8), connectivity=8)
    cand = []
    for i in range(1, n):
        rows, cols = np.nonzero(labels == i)
        if len(rows) < 3:
            continue
        d = dt_free[rows, cols]
        tight = d <= d.min() + 0.5
        rr, cc = rows[tight], cols[tight]
        j = np.argmin((rr - rr.mean()) ** 2 + (cc - cc.mean()) ** 2)
        cand.append((d.min(), rr[j], cc[j]))

    acc_u8 = accessible.astype(np.uint8)
    n_before = cv2.connectedComponents(acc_u8, connectivity=8)[0]
    narrow = []
    for v, r, c in non_max_suppression(cand, nms_px, smallest_first=True)[:3 * max_features]:
        H = np.array([[gxx[r, c], gxy[r, c]], [gxy[r, c], gyy[r, c]]])
        evals, evecs = np.linalg.eigh(H)
        across = evecs[:, 0]                 # most negative curvature -> across the passage (x, y)
        # Critical passage (chokepoint): cutting it splits the accessible space into more parts,
        # i.e. it is the only connection and not just a gap next to a rock inside a room
        cut = acc_u8.copy()
        half = v + 3
        cv2.line(cut, (int(c - across[0] * half), int(r - across[1] * half)),
                 (int(c + across[0] * half), int(r + across[1] * half)), 0, 3)
        critical = cv2.connectedComponents(cut, connectivity=8)[0] > n_before
        narrow.append({'rc': (r, c), 'width': 2 * v * res, 'across': across,
                       'saddle': bool(evals[0] < 0 < evals[1]), 'critical': bool(critical)})
    # Bottleneck = narrowest critical passage (fallback: narrowest passage)
    narrow.sort(key=lambda n: (not n['critical'], n['width']))
    result['narrow'] = narrow[:max_features]
    result['bottleneck'] = result['narrow'][0] if narrow else None
    return result


def draw_result(result, res):
    """Colour image of the distance transform with the features (for the report)."""
    dt = result['dt']
    vis = np.zeros(dt.shape + (3,), np.uint8)
    if dt.max() > 0:
        norm = (255 * dt / dt.max()).astype(np.uint8)
        vis = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    vis[dt == 0] = (40, 40, 40)
    vis[result['ridge']] = (0, 255, 255)
    if result['widest']:
        r, c = result['widest']['rc']
        cv2.circle(vis, (c, r), int(result['widest']['width'] / 2 / res), (0, 255, 0), 2)
    for w in result['wide']:
        r, c = w['rc']
        cv2.circle(vis, (c, r), 3, (0, 255, 0), -1)
    for i, n in enumerate(result['narrow']):
        r, c = n['rc']
        half = n['width'] / 2 / res
        dx, dy = n['across']
        cv2.line(vis, (int(c - dx * half), int(r - dy * half)), (int(c + dx * half), int(r + dy * half)),
                 (0, 0, 255) if i == 0 else (0, 140, 255), 2)
    return cv2.flip(vis, 0)   # map row 0 is at the bottom


# ----------------------------------------------------------------------------
# ROS node
# ----------------------------------------------------------------------------
class CaveGeometry(Node):
    def __init__(self):
        super().__init__('cave_geometry_node')

        self.declare_parameter('robot_radius', 0.4)          # m, minimum clearance to be accessible
        self.declare_parameter('narrow_max_width', 3.0)      # m, passages narrower than this are reported
        self.declare_parameter('wide_min_width', 5.0)        # m, open areas wider than this are reported
        self.declare_parameter('update_period', 2.0)         # s
        self.declare_parameter('save_image', os.path.expanduser('~/cave_geometry.png'))

        self.map_ = None
        self.map_sub_ = self.create_subscription(OccupancyGrid, 'map', self.map_callback, 1)

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             reliability=ReliabilityPolicy.RELIABLE)
        self.dt_pub_ = self.create_publisher(OccupancyGrid, 'distance_transform', latched)
        self.marker_pub_ = self.create_publisher(MarkerArray, 'cave_geometry_markers', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.last_report_ = None
        self.timer_ = self.create_timer(self.get_parameter('update_period').value, self.update)

    def map_callback(self, msg):
        self.map_ = msg

    def update(self):
        msg = self.map_
        if msg is None:
            return
        info = msg.info
        res = info.resolution
        ox, oy = info.origin.position.x, info.origin.position.y
        grid = np.array(msg.data, dtype=np.int8).reshape(info.height, info.width)

        # Robot cell
        robot_rc = None
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_link', rclpy.time.Time())
            robot_rc = (int((t.transform.translation.y - oy) / res),
                        int((t.transform.translation.x - ox) / res))
        except TransformException:
            pass

        result = analyse_map(grid, res, robot_rc,
                             robot_radius=self.get_parameter('robot_radius').value,
                             narrow_max_width=self.get_parameter('narrow_max_width').value,
                             wide_min_width=self.get_parameter('wide_min_width').value)

        def to_world(rc):
            return ox + (rc[1] + 0.5) * res, oy + (rc[0] + 0.5) * res

        self.publish_distance_transform(msg, result)
        self.publish_markers(result, res, to_world, msg.header.stamp)
        self.log_report(result, to_world)

        path = self.get_parameter('save_image').value
        if path:
            cv2.imwrite(path, draw_result(result, res))

    def publish_distance_transform(self, map_msg, result):
        dt = result['dt']
        out = OccupancyGrid()
        out.header = map_msg.header
        out.info = map_msg.info
        scaled = np.full(dt.shape, -1, np.int8)
        if dt.max() > 0:
            scaled[dt > 0] = np.clip(100 * dt[dt > 0] / dt.max(), 1, 100).astype(np.int8)
        out.data = scaled.flatten().tolist()
        self.dt_pub_.publish(out)

    def publish_markers(self, result, res, to_world, stamp):
        markers = MarkerArray()
        clear = Marker()
        clear.header.frame_id = 'map'
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        mid = [0]

        def marker(ns, mtype, r, g, b, a=1.0):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp = stamp
            m.ns = ns
            m.id = mid[0]
            mid[0] += 1
            m.type = mtype
            m.action = Marker.ADD
            m.pose.orientation.w = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, a
            markers.markers.append(m)
            return m

        def text(ns, x, y, s, r, g, b):
            m = marker(ns, Marker.TEXT_VIEW_FACING, r, g, b)
            m.pose.position.x, m.pose.position.y, m.pose.position.z = x, y, 1.0
            m.scale.z = 0.7
            m.text = s

        # Ridge / medial axis
        rows, cols = np.nonzero(result['ridge'])
        step = max(1, len(rows) // 5000)
        if len(rows):
            m = marker('ridge', Marker.POINTS, 1.0, 1.0, 0.0, 0.8)
            m.scale.x = m.scale.y = res * 1.5
            m.points = [Point(x=x, y=y, z=0.05) for x, y in
                        (to_world((r, c)) for r, c in zip(rows[::step], cols[::step]))]

        # Widest accessible area
        if result['widest']:
            w = result['widest']
            x, y = to_world(w['rc'])
            m = marker('widest_area', Marker.CYLINDER, 0.0, 1.0, 0.0, 0.3)
            m.pose.position.x, m.pose.position.y = x, y
            m.scale.x = m.scale.y = w['width']
            m.scale.z = 0.05
            text('widest_area', x, y, f'widest area {w["width"]:.1f} m', 0.0, 1.0, 0.0)

        for w in result['wide'][1:]:
            x, y = to_world(w['rc'])
            m = marker('wide_areas', Marker.SPHERE, 0.4, 1.0, 0.4)
            m.pose.position.x, m.pose.position.y = x, y
            m.scale.x = m.scale.y = m.scale.z = 0.4
            text('wide_areas', x, y, f'wide {w["width"]:.1f} m', 0.4, 1.0, 0.4)

        # Narrow passages (first one = bottleneck)
        for i, n in enumerate(result['narrow']):
            x, y = to_world(n['rc'])
            half = n['width'] / 2
            dx, dy = n['across']
            col = (1.0, 0.0, 0.0) if i == 0 else (1.0, 0.55, 0.0)
            m = marker('narrow_passages', Marker.LINE_STRIP, *col)
            m.scale.x = 0.15
            m.points = [Point(x=x - dx * half, y=y - dy * half, z=0.1),
                        Point(x=x + dx * half, y=y + dy * half, z=0.1)]
            label = 'BOTTLENECK' if i == 0 else ('chokepoint' if n['critical'] else 'narrow')
            text('narrow_passages', x, y, f'{label} {n["width"]:.1f} m', *col)

        self.marker_pub_.publish(markers)

    def log_report(self, result, to_world):
        parts = []
        if result['widest']:
            x, y = to_world(result['widest']['rc'])
            parts.append(f'widest area: width {result["widest"]["width"]:.2f} m at ({x:.1f}, {y:.1f})')
        if result['bottleneck']:
            b = result['bottleneck']
            x, y = to_world(b['rc'])
            parts.append(f'bottleneck: width {b["width"]:.2f} m at ({x:.1f}, {y:.1f})'
                         f'{" [only connection]" if b["critical"] else ""}')
        parts.append(f'{len(result["wide"])} wide areas, {len(result["narrow"])} narrow passages')
        report = ' | '.join(parts)
        if report != self.last_report_:
            self.get_logger().info(report)
            self.last_report_ = report


def main():
    rclpy.init()
    node = CaveGeometry()
    rclpy.spin(node)