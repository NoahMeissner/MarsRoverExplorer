# A single artefact of interest, as estimated by the perception pipeline.
# See INTERFACE.md in the cave_explorer package for the full contract.

string                   id                 # stable for the whole session
string                   class_name
float32                  confidence         # 0.0 - 1.0
geometry_msgs/Point      position           # in the map frame
uint32                   num_observations
geometry_msgs/Point      observer_position  # robot pos at best observation, map frame
builtin_interfaces/Time  last_seen
