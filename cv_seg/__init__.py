"""
cv_seg — pure computer-vision goalie threat segment detector.

Public API:
    process_video — top-level orchestrator
    extract_frame_signals, detect_whistles, detect_crowd_roar_spikes
    build_candidate_windows
    assign_goalie_colors, detect_goalie_sides_cv, detect_period_side_maps
    merge_overlapping_segments, enforce_boundaries, ...

The CLI lives in cv_seg.cli and is invoked via `python -m cv_seg`.

Imports are lazy (PEP 562 __getattr__) so importing only the pure-Python
post-process helpers does NOT drag in OpenCV / librosa / google-cloud.
The first attribute access into a heavy submodule (signals, audio,
attribution, pipeline, io_utils) loads it on demand.
"""

import importlib

__version__ = "23.7"

# Map public name → (submodule, attribute name).
# Submodules cluster by dependency footprint:
#   postprocess  — pure Python (no extra deps)
#   windows      — pure Python
#   colors       — numpy only
#   signals      — opencv + numpy
#   audio        — librosa + numpy (librosa is optional at import time)
#   attribution  — opencv + numpy
#   pipeline     — pulls in everything above
_LAZY = {
    # Pipeline
    "process_video":                       ("pipeline",     "process_video"),
    # Frame signals
    "extract_frame_signals":               ("signals",      "extract_frame_signals"),
    # detect_red_light and measure_bench_activity removed from public
    # API in v23.5; underlying functions still defined in signals.py
    # but no longer called. Remove the functions next time the file
    # is touched.
    "detect_centre_faceoff":               ("signals",      "detect_centre_faceoff"),
    "compute_motion_thirds":               ("signals",      "compute_motion_thirds"),
    "detect_scene_type":                   ("signals",      "detect_scene_type"),
    "detect_celebration_clustering":       ("signals",      "detect_celebration_clustering"),
    # Audio
    "detect_whistles":                     ("audio",        "detect_whistles"),
    "detect_crowd_roar_spikes":            ("audio",        "detect_crowd_roar_spikes"),
    # Windows
    "build_candidate_windows":             ("windows",      "build_candidate_windows"),
    # Attribution
    "assign_goalie_colors":                ("attribution",  "assign_goalie_colors"),
    "detect_goalie_sides_cv":              ("attribution",  "detect_goalie_sides_cv"),
    "detect_period_side_maps":             ("attribution",  "detect_period_side_maps"),
    # Postprocess (pure Python — cheap, but kept lazy for symmetry)
    "merge_overlapping_segments":          ("postprocess",  "merge_overlapping_segments"),
    "enforce_boundaries":                  ("postprocess",  "enforce_boundaries"),
    "merge_adjacent_same_type":            ("postprocess",  "merge_adjacent_same_type"),
    "split_long_threats":                  ("postprocess",  "split_long_threats"),
    "split_segments_at_period_boundaries": ("postprocess",  "split_segments_at_period_boundaries"),
    "cap_segment_length":                  ("postprocess",  "cap_segment_length"),
    "apply_side_assignments":              ("postprocess",  "apply_side_assignments"),
    "make_no_threat":                      ("postprocess",  "make_no_threat"),
}


def __getattr__(name: str):
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = importlib.import_module(f".{mod_name}", __name__)
        value = getattr(mod, attr)
        # Cache on the package so subsequent accesses skip __getattr__.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY.keys()))


__all__ = [
    "__version__",
    *_LAZY.keys(),
]
