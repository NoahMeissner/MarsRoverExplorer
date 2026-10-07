"""Run metrics logged to CSV, for the quantitative comparison in the report.

Logged at 1 Hz: explored area, distance travelled, artefacts inspected and
abandoned, simulated time.

Used to compare three configurations over three runs each (template random
goals / nearest-frontier / weighted score with real path cost). Same approach
as the PRM parameter sweep in Assignment 2 - report the trend, and note the
run-to-run variability rather than presenting single numbers as exact.

Phase 6 of the implementation plan.
"""


class MetricsLogger:
    """Appends one row per second to a CSV file."""

    def __init__(self, node, csv_path):
        self.node_ = node
        self.csv_path_ = csv_path
        # TODO 54
