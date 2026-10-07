# Perception → Planning interface

Owner of the producer side: <teammate>  (Perception 1-3)
Owner of the consumer side: Dorian      (Planning 1-3)

## Topic

| | |
|---|---|
| Name | `/artefacts` |
| Type | `cave_explorer_msgs/msg/ArtefactArray` |
| Rate | 2 Hz, steady |
| Semantics | **Full list, republished every time.** Not events, not deltas. |

Publishing the complete current list on every message is deliberate: it is
idempotent, the planner just diffs against its own record, and a dropped
message costs nothing. Do not publish "new artefact" events.

## Message


## Guarantees required from the producer

1. **Frame.** `position` and `observer_position` are in the `map` frame.
   Not `base_link`, not `camera_rgb_optical_frame`. Use tf2 to transform.

2. **Stable IDs.** An artefact keeps the same `id` for the entire session,
   including when it goes out of view and comes back. If IDs churn, the
   planner re-inspects the same object indefinitely and exploration never
   completes.

3. **Position accuracy.** Target better than +/- 0.8 m. The planner places
   the robot on a 2.5 m standoff circle around `position`; beyond that error
   the robot ends up either inside the costmap inflation zone (goal rejected)
   or too far away for a usable image.

4. **`observer_position` is not optional.** The planner uses it to pick which
   side of the artefact to approach from. An artefact against a cave wall has
   only half its standoff circle reachable, and the direction it was seen from
   is proof that a line of sight exists there.

5. **Camera intrinsics.** `/camera/camera_info` is not currently bridged.
   Either add it to `config/gazebo_bridge_params.yaml`, or document the
   hardcoded intrinsics taken from `urdf/mars_explorer.urdf.xacro` here.

6. **Monotonic confidence.** `confidence` should reflect accumulated evidence,
   not just the latest frame, so that a brief occlusion does not make a
   well-established artefact drop below the planner's trigger threshold.

## What the planner does with each field

| Field | Used for |
|---|---|
| `id` | Primary key of the inspection record. Prevents double inspection. |
| `class_name` | Filtered against the planner's target list. |
| `confidence` | Trigger threshold, avoids detours on false positives. |
| `num_observations` | Trigger threshold. A real artefact is seen over several consecutive frames; a false positive flickers. |
| `position` | Centre of the standoff circle for viewpoint generation. |
| `observer_position` | Restricts candidate viewpoints to the side with known line of sight. |
| `last_seen` | "Artefact lost" detection during the final alignment phase. |

## Artefact classes

Agreed target classes for Planning 2/3 (pick 2-3 the detector handles
reliably, per the brief):

- [ ] TBD - to be filled once the detector is trained
- [ ] TBD
- [ ] TBD

The planner matches on exact string. Agree the exact spelling here before
either side hardcodes it.

## Known open question

Detection range at which the detector is reliable: __ m.
This sets the planner's "artefact lost" timeout during approach.

## Testing

Until the real detector publishes, the planner is developed against
`cave_explorer/fake_artefacts.py`, which publishes this exact message type
with perfect positions and stable IDs. Any behaviour difference when
switching to the real topic is therefore a perception-side issue, not a
planning-side one.
