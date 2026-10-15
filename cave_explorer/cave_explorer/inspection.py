"""
[P2] Viewpoint selection for close-range inspection of an artefact.

Pure numpy / OpenCV (no ROS), so it can be tested offline on a synthetic grid.

Candidate viewpoints lie on a circle of radius `standoff` around the artefact. A candidate is valid if
  - the robot fits there (free cell, enough clearance from walls),
  - Nav2 can reach it (finite path distance from the robot, same wavefront as the frontier planner),
  - the camera has a line of sight to the artefact (no wall cell on the segment).
The best valid candidate minimises
  path distance + w_side * (angle between the candidate and the side the robot saw the artefact from)
because the side the artefact was detected from is known to be visible, while its back may be against a wall.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

from cave_explorer.frontier_exploration import OCCUPIED_THRESHOLD, path_distance_field


@dataclass
class Viewpoint:
    xy: np.ndarray          # world x, y
    yaw: float              # heading that faces the artefact
    valid: bool
    reason: str             # why it was rejected ('' if valid)
    path_dist: float = math.inf
    cost: float = math.inf


def choose_viewpoint(grid, resolution, origin, robot_xy, artefact_xy, standoff=2.0, min_clearance=0.6,
                     n_candidates=24, w_side=2.0, artefact_radius=1.0, excluded=()):
    """
    grid:        (H, W) int8 occupancy grid (row = y, col = x), -1 unknown, 0 free, 100 occupied
    origin:      (x, y) world position of cell (0, 0) corner
    artefact_radius: occupied cells this close to the artefact centre are the artefact itself (the lidar
                 may see it), they do not block the line of sight
    excluded:    viewpoints (x, y) that Nav2 already failed to reach
    Returns (best Viewpoint or None, list of all candidate Viewpoints for display)
    """
    res = resolution
    ox, oy = origin
    h, w = grid.shape
    artefact = np.asarray(artefact_xy, dtype=float)
    robot = np.asarray(robot_xy, dtype=float)

    def cell(xy):
        return int((xy[1] - oy) / res), int((xy[0] - ox) / res)

    free = (grid >= 0) & (grid < OCCUPIED_THRESHOLD)
    occupied = grid >= OCCUPIED_THRESHOLD
    clearance = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * res
    traversable = free & (clearance >= min_clearance)

    start = cell(robot)
    if not (0 <= start[0] < h and 0 <= start[1] < w) or not traversable[start]:
        # robot close to a wall: start from the closest traversable cell
        rows, cols = np.nonzero(traversable)
        if rows.size == 0:
            return None, []
        k = int(np.argmin((rows - start[0]) ** 2 + (cols - start[1]) ** 2))
        start = (rows[k], cols[k])
    dist = path_distance_field(traversable, start) * res

    side = math.atan2(robot[1] - artefact[1], robot[0] - artefact[0])   # side the artefact was seen from
    candidates = []
    for k in range(n_candidates):
        a = 2.0 * math.pi * k / n_candidates
        xy = artefact + standoff * np.array([math.cos(a), math.sin(a)])
        vp = Viewpoint(xy=xy, yaw=math.atan2(artefact[1] - xy[1], artefact[0] - xy[0]), valid=False, reason='')
        candidates.append(vp)
        r, c = cell(xy)
        if not (0 <= r < h and 0 <= c < w):
            vp.reason = 'outside map'
        elif not free[r, c]:
            vp.reason = 'not free'
        elif not traversable[r, c]:
            vp.reason = 'too close to a wall'
        elif not np.isfinite(dist[r, c]):
            vp.reason = 'unreachable'
        elif not line_of_sight(occupied, cell(xy), cell(artefact), ignore_radius=artefact_radius / res):
            vp.reason = 'no line of sight'
        elif any(math.hypot(xy[0] - ex, xy[1] - ey) < 0.5 for ex, ey in excluded):
            vp.reason = 'Nav2 failed before'
        else:
            vp.valid = True
            vp.path_dist = float(dist[r, c])
            side_change = abs(math.atan2(math.sin(a - side), math.cos(a - side)))
            vp.cost = vp.path_dist + w_side * side_change

    valid = [vp for vp in candidates if vp.valid]
    best = min(valid, key=lambda vp: vp.cost) if valid else None
    return best, candidates


def line_of_sight(occupied, start, goal, ignore_radius=0.0):
    """True if no occupied cell lies on the grid segment start -> goal. Cells closer than
    `ignore_radius` (in cells) to the goal are ignored: they are the artefact itself."""
    (r0, c0), (r1, c1) = start, goal
    n = max(abs(r1 - r0), abs(c1 - c0))
    h, w = occupied.shape
    for i in range(n + 1):
        t = i / n if n else 0.0
        r, c = int(round(r0 + t * (r1 - r0))), int(round(c0 + t * (c1 - c0)))
        if math.hypot(r - r1, c - c1) < ignore_radius:
            break
        if not (0 <= r < h and 0 <= c < w) or occupied[r, c]:
            return False
    return True
