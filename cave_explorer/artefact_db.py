"""Record of every artefact the robot knows about.

This table is what makes "never inspect the same artefact twice" possible.
It is not an algorithm, it is a state field.

Perception fills in position and class. The planner owns the state field and
never writes back to the perception side.

Phase 4-5 of the implementation plan.
"""

from enum import Enum


class ArtefactState(Enum):
    PENDING = 0      # known, in a target class, not yet visited
    IN_PROGRESS = 1  # currently being approached or inspected
    INSPECTED = 2    # done
    ABANDONED = 3    # terminal. Never goes back to PENDING, or the robot
                     # loops forever on an unreachable artefact.


class ArtefactDB:
    """Merges incoming ArtefactArray messages and tracks inspection state."""

    def __init__(self, node):
        self.node_ = node
        self.artefacts_ = {}
        # TODO 40
