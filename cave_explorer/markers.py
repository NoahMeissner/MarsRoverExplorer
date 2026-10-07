"""RViz marker helpers and the namespace registry.

Namespaces are fixed here so that the saved cave_explorer.rviz config keeps
working as the code evolves. Changing one means re-saving the RViz config.
"""

NS_FRONTIER_CELLS      = 'frontier_cells'
NS_FRONTIER_CANDIDATES = 'frontier_candidates'
NS_CURRENT_GOAL        = 'current_goal'
NS_ARTEFACTS           = 'artefacts'
NS_VIEWPOINTS          = 'viewpoints'

# Artefact state colour code, as required by the Planning 3 brief.
COLOUR_PENDING     = (1.0, 0.85, 0.0, 1.0)   # yellow
COLOUR_IN_PROGRESS = (0.2, 0.5, 1.0, 1.0)    # blue
COLOUR_INSPECTED   = (0.1, 0.8, 0.3, 1.0)    # green
COLOUR_ABANDONED   = (0.9, 0.2, 0.2, 1.0)    # red

# TODO 38, 51
