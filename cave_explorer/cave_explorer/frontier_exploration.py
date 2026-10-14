"""
[P1] Frontier-based exploration on the SLAM occupancy grid.

Pure numpy / OpenCV (no ROS), so it can be tested offline on a synthetic grid.

Pipeline (see 49274 Decision Making slides 18-21):
  1. frontier cells  = free cells with at least one unknown 8-neighbour
  2. frontier clusters = 8-connected components of frontier cells, small ones are noise
  3. traversable space = free cells with enough clearance from walls (robot radius)
  4. path distance from the robot to every traversable cell (wavefront / Dijkstra on the grid),
     so frontiers behind a wall are ranked by the real travel distance, not the straight line
  5. one viewpoint per cluster = reachable cell close to the cluster centre
  6. score = w_size * frontier length - w_dist * path distance - w_turn * heading change
     (+ a bonus for the current target, so the robot does not oscillate between two frontiers)
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

OCCUPIED_THRESHOLD = 50  # SLAM grid: -1 unknown, 0 free, 100 occupied

# 8-connected neighbourhood: (d_row, d_col, step length in cells)
NEIGHBOURS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, math.sqrt(2.0)), (-1, 1, math.sqrt(2.0)),
              (1, -1, math.sqrt(2.0)), (1, 1, math.sqrt(2.0))]


def wrap_pi(angle):
    """Wrap an angle to [-pi, pi]"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Frontier:
    cells: np.ndarray       # (N, 2) world x, y of the frontier cells
    size_m: float           # frontier length (number of cells * resolution)
    centroid: np.ndarray    # world x, y
    goal: np.ndarray        # world x, y of the viewpoint the robot drives to
    path_dist: float        # travel distance from the robot to the goal [m]
    heading_change: float   # |angle| between robot heading and goal direction [rad]
    score: float = 0.0


@dataclass
class ExplorationResult:
    frontiers: list         # scored, reachable, not excluded frontiers (best first)
    rejected: list          # frontiers that are unreachable / excluded / too close (for display)
    best: Frontier          # None if nothing left to explore
    path: np.ndarray        # (M, 2) world x, y from the robot to best.goal (None if no best)
    known_area_m2: float
    free_area_m2: float


class FrontierExplorer:

    def __init__(self, min_clearance=0.6, min_frontier_size=4, viewpoint_radius=1.5,
                 min_goal_dist=1.0, w_size=1.0, w_dist=1.0, w_turn=1.0, target_bonus=3.0):
        self.min_clearance = min_clearance          # [m] keep the robot centre this far from walls
        self.min_frontier_size = min_frontier_size  # [cells] smaller clusters are noise
        self.viewpoint_radius = viewpoint_radius    # [m] search radius around a cluster for its viewpoint
        self.min_goal_dist = min_goal_dist          # [m] goals closer than this are already "reached" by Nav2
        self.w_size = w_size                        # reward per metre of frontier
        self.w_dist = w_dist                        # cost per metre of travel
        self.w_turn = w_turn                        # cost per radian of heading change ("momentum")
        self.target_bonus = target_bonus            # hysteresis bonus for the frontier we already drive to

    # ------------------------------------------------------------------ main entry
    def compute(self, grid, resolution, origin, robot_xy, robot_yaw,
                excluded=(), current_target=None):
        """
        grid:           (H, W) int8 occupancy grid, row = y, col = x
        origin:         (x, y) world position of cell (0, 0) corner
        robot_xy:       (x, y) robot position in the map frame
        excluded:       iterable of (x, y, radius): goals inside these discs are not selected
                        (already visited / navigation failed)
        current_target: (x, y) of the goal currently being driven to, or None
        """
        res = resolution
        ox, oy = origin
        h, w = grid.shape

        def to_world(rows, cols):
            return np.stack([ox + (cols + 0.5) * res, oy + (rows + 0.5) * res], axis=-1)

        free = (grid >= 0) & (grid < OCCUPIED_THRESHOLD)
        unknown = grid < 0
        occupied = grid >= OCCUPIED_THRESHOLD

        # Clearance to the closest wall [m]; distanceTransform measures the distance to the closest 0 pixel
        clearance = cv2.distanceTransform((~occupied).astype(np.uint8), cv2.DIST_L2, 5) * res
        traversable = free & (clearance >= self.min_clearance)

        # Travel distance from the robot to every traversable cell
        start = self._start_cell(traversable, robot_xy, origin, res)
        if start is None:
            dist = np.full(grid.shape, np.inf)
        else:
            dist = path_distance_field(traversable, start) * res
        reachable = np.isfinite(dist)

        # Frontier cells and clusters
        unknown_nb = cv2.dilate(unknown.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        frontier_mask = free & unknown_nb
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            frontier_mask.astype(np.uint8), connectivity=8)

        r_vp = max(1, int(round(self.viewpoint_radius / res)))
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_vp + 1, 2 * r_vp + 1))
        robot = np.asarray(robot_xy, dtype=float)

        frontiers, rejected = [], []
        for i in range(1, n_labels):
            size = stats[i, cv2.CC_STAT_AREA]
            if size < self.min_frontier_size:
                continue
            cluster = labels == i
            rows, cols = np.nonzero(cluster)
            cells = to_world(rows, cols)
            centroid = np.array([ox + (centroids[i][0] + 0.5) * res, oy + (centroids[i][1] + 0.5) * res])

            # Viewpoint: reachable cell near the cluster, closest to its centre
            near = (cv2.dilate(cluster.astype(np.uint8), disk) > 0) & reachable
            f = Frontier(cells=cells, size_m=size * res, centroid=centroid, goal=centroid,
                         path_dist=math.inf, heading_change=0.0, score=-math.inf)
            if not near.any():
                rejected.append(f)          # e.g. frontier in a passage too narrow for the robot
                continue
            vr, vc = np.nonzero(near)
            cand = to_world(vr, vc)
            k = int(np.argmin(np.sum((cand - centroid) ** 2, axis=1)))
            f.goal = cand[k]
            f.path_dist = float(dist[vr[k], vc[k]])
            to_goal = f.goal - robot
            f.heading_change = abs(wrap_pi(math.atan2(to_goal[1], to_goal[0]) - robot_yaw))

            if np.hypot(*to_goal) < self.min_goal_dist or \
                    any(np.hypot(f.goal[0] - ex, f.goal[1] - ey) < er for ex, ey, er in excluded):
                rejected.append(f)
                continue

            f.score = self.w_size * f.size_m - self.w_dist * f.path_dist - self.w_turn * f.heading_change
            if current_target is not None and \
                    np.hypot(*(f.goal - np.asarray(current_target))) < 2.0 * self.viewpoint_radius:
                f.score += self.target_bonus
            frontiers.append(f)

        frontiers.sort(key=lambda fr: fr.score, reverse=True)
        best = frontiers[0] if frontiers else None
        path = None
        if best is not None:
            gr, gc = self._cell(best.goal, origin, res)
            path = to_world(*np.array(extract_path(dist, (gr, gc))).T)

        return ExplorationResult(frontiers=frontiers, rejected=rejected, best=best, path=path,
                                 known_area_m2=float((~unknown).sum()) * res * res,
                                 free_area_m2=float(free.sum()) * res * res)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _cell(xy, origin, res):
        return int((xy[1] - origin[1]) / res), int((xy[0] - origin[0]) / res)

    def _start_cell(self, traversable, robot_xy, origin, res):
        """Robot cell, or the closest traversable cell if the robot is close to a wall"""
        h, w = traversable.shape
        r, c = self._cell(robot_xy, origin, res)
        if 0 <= r < h and 0 <= c < w and traversable[r, c]:
            return r, c
        rows, cols = np.nonzero(traversable)
        if rows.size == 0:
            return None
        d2 = (rows - r) ** 2 + (cols - c) ** 2
        k = int(np.argmin(d2))
        if d2[k] * res * res > 2.0 ** 2:   # nothing traversable within 2 m
            return None
        return rows[k], cols[k]


def path_distance_field(traversable, start):
    """
    Shortest 8-connected path length (in cells) from start to every traversable cell, inf elsewhere.
    Vectorised wavefront (Bellman-Ford sweeps with numpy): each sweep relaxes every cell from its
    8 neighbours, so it converges after (longest shortest path) sweeps - ~100 ms for a cave map.
    """
    full = np.full(traversable.shape, np.inf)
    if not traversable[start]:
        return full
    # Only the bounding box of the traversable cells matters
    rows, cols = np.nonzero(traversable)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    blocked = ~traversable[r0:r1, c0:c1]
    h, w = blocked.shape

    padded = np.full((h + 2, w + 2), np.inf, dtype=np.float32)   # border stays inf
    dist = padded[1:-1, 1:-1]                                     # view into the padded buffer
    dist[start[0] - r0, start[1] - c0] = 0.0
    new = np.empty((h, w), dtype=np.float32)
    tmp = np.empty((h, w), dtype=np.float32)
    for _ in range(h * w):
        new[:] = dist
        for dr, dc, cost in NEIGHBOURS:
            np.add(padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w], cost, out=tmp)
            np.minimum(new, tmp, out=new)
        new[blocked] = np.inf
        if np.array_equal(new, dist):
            break
        dist[:] = new
    full[r0:r1, c0:c1] = dist
    return full


def extract_path(dist, goal):
    """Follow the distance field downhill from goal back to the start cell (dist == 0)"""
    h, w = dist.shape
    path = [goal]
    r, c = goal
    for _ in range(h * w):
        if dist[r, c] == 0.0:
            break
        best = (r, c)
        for dr, dc, _ in NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] < dist[best]:
                best = (nr, nc)
        if best == (r, c):
            break
        r, c = best
        path.append(best)
    return path[::-1]
