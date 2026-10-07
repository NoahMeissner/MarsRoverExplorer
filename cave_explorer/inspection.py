"""Viewpoint generation for close-range artefact inspection.

The problem: perception gives a POINT, navigation needs a POSE. Converting
one into the other is the whole of Planning 2.

Going straight to the artefact fails twice over - the point sits on or inside
an obstacle so Nav2 rejects it, and even if it did not, a robot pressed
against an object sees nothing. So the robot stands off at a distance, which
defines a CIRCLE of possible positions rather than a single one, and the
circle is then pruned by navigability, clearance, line of sight, and the
direction the artefact was originally observed from.

Phase 4 of the implementation plan.
"""


def candidate_viewpoints(artefact, map_model, standoff_m, n_candidates):
    """Return reachable poses that look at the artefact, best first.

    The runners-up are kept, not discarded: they are what the single retry in
    Planning 3 uses. Retrying the same viewpoint would fail the same way.
    """
    # TODO 41-42
    raise NotImplementedError
