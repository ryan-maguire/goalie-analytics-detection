"""
Shared helper for resolving the side map active at a given time.

Used by attribution.assign_goalie_colors and postprocess.apply_side_assignments,
which previously each carried a near-identical local helper.
"""

from typing import Optional


def side_map_at(
    t: int,
    initial_side_map: dict,
    period_side_maps: Optional[list[tuple[int, dict]]] = None,
) -> dict:
    """
    Return the side map active at time t, accounting for period swaps.

    Scans ALL entries and picks the one with the largest start_t that
    is <= t — does not assume period_side_maps is sorted.
    """
    if not period_side_maps:
        return initial_side_map
    active = initial_side_map
    active_start = -1
    for start_t, sm in period_side_maps:
        if start_t <= t and start_t >= active_start:
            active = sm
            active_start = start_t
    return active
