"""Thin wrapper around the Nav2 action servers and costmap services.

This is the ONLY place in the codebase that talks to Nav2 directly. The state
machine asks for a pose and reads back a simple status; it never touches an
action client. That separation is what makes clean preemption possible.

Phase 1 of the implementation plan.
"""

from enum import Enum


class NavState(Enum):
    """Outcome of the most recent navigation request."""
    IDLE = 0
    ACTIVE = 1
    SUCCEEDED = 2
    FAILED = 3
    CANCELED = 4


class NavClient:
    """Owns navigate_to_pose, compute_path_to_pose, spin, backup and the
    costmap clearing services.

    Every goal carries a monotonically increasing tag. Result callbacks whose
    tag is stale are discarded: without this, the result of a cancelled goal
    arrives a few hundred milliseconds later and wrongly reports success to a
    state machine that has already moved on.
    """

    def __init__(self, node):
        self.node_ = node
        self.state_ = NavState.IDLE
        self.goal_tag_ = 0
        self.goal_handle_ = None
        # TODO 11-18
