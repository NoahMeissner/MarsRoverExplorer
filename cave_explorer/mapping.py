"""World model: the occupancy grid, the global costmap, and the geometric
queries built on top of them.

Pure geometry. This module answers questions ("is this cell navigable?",
"is there line of sight between A and B?") but never decides anything.

Two distinct maps, deliberately:
  /map                     SLAM output. Authoritative on what is UNKNOWN.
                           Values: -1 unknown, 0 free, 100 occupied.
  /global_costmap/costmap  What Nav2 will actually plan against, inflation
                           already applied. Values: -1 unknown, 0 free,
                           1-98 inflation gradient, 99 inscribed, 100 lethal.

Frontiers are detected on /map. Goals are validated against the costmap, so
that a candidate the planner accepts is a candidate Nav2 will accept.

Phase 2 of the implementation plan.
"""


class MapModel:
    """Subscribes to both grids and exposes the geometric predicates."""

    def __init__(self, node):
        self.node_ = node
        # TODO 21-27
