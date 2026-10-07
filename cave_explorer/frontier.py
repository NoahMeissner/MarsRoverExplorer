"""Frontier detection and scoring.

Pure numpy / OpenCV: arrays in, candidate list out. Imports no ROS. That is
deliberate - it means the algorithm can be developed and tuned offline
against a saved map, without launching Gazebo (step 28).

A frontier is a free cell adjacent to an unknown cell: the boundary of what
is known. Three properties make it the right exploration target:
  - it sits in already-mapped free space, so it is reachable by construction
  - travelling there necessarily converts unknown space into known space
  - when no frontiers remain, exploration is complete - the stopping
    criterion emerges from the method instead of being hardcoded

Phase 3 of the implementation plan.
"""


def find_frontiers(grid, resolution, origin, min_cluster_cells, clearance_map):
    """Return the list of frontier clusters as candidate goals."""
    # TODO 29-32
    raise NotImplementedError
