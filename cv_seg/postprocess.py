"""
Post-processing helpers for the segment timeline.

These are pure-Python list-of-dict transforms with no I/O, no OpenCV,
and no external dependencies. They are heavily unit-tested in
tests/test_postprocess.py.
"""

from typing import Optional

from . import constants as C
from .logger import log
from .side_map import side_map_at


def make_no_threat(start: int, end: int) -> dict:
    """Construct a no-threat segment covering [start, end)."""
    return {
        "segmentHasThreat":    False,
        "threat_goalie_color": None,
        "threat_goalie_side":  None,
        "segment_start":       start,
        "segment_end":         end,
    }


def merge_overlapping_segments(all_segments: list[dict]) -> list[dict]:
    """
    Merge overlapping or abutting segments.

    Segments with the same classification (hasThreat + color) are merged;
    conflicting overlaps are resolved by keeping the later segment's
    classification. Input need not be sorted — this function sorts
    internally.
    """
    if not all_segments:
        return []
    segments = sorted(all_segments, key=lambda s: (s["segment_start"], s["segment_end"]))
    merged = [dict(segments[0])]

    for current in segments[1:]:
        prev = merged[-1]
        if current["segment_start"] >= prev["segment_end"]:
            if current["segment_start"] > prev["segment_end"]:
                merged.append(make_no_threat(prev["segment_end"], current["segment_start"]))
            merged.append(dict(current))
            continue
        same = (
            current["segmentHasThreat"] == prev["segmentHasThreat"]
            and current["threat_goalie_color"] == prev["threat_goalie_color"]
        )
        if same:
            prev["segment_end"] = max(prev["segment_end"], current["segment_end"])
        else:
            if current["segment_start"] > prev["segment_start"]:
                prev["segment_end"] = current["segment_start"]
                merged.append(dict(current))
            else:
                merged[-1] = dict(current)
    return merged


def enforce_boundaries(segments: list[dict], total_duration: int) -> list[dict]:
    """
    Ensure the segment list covers exactly [0, total_duration] with no
    gaps and no overruns. Fills leading/trailing/internal gaps with
    no-threat segments. Sorts internally so callers don't have to.
    """
    if not segments:
        return [make_no_threat(0, total_duration)]

    segments = sorted(segments, key=lambda s: s["segment_start"])

    result = []
    if segments[0]["segment_start"] > 0:
        result.append(make_no_threat(0, segments[0]["segment_start"]))

    for seg in segments:
        start = max(0, min(seg["segment_start"], total_duration))
        end   = max(0, min(seg["segment_end"],   total_duration))
        if end <= start:
            continue
        if result and result[-1]["segment_end"] < start:
            result.append(make_no_threat(result[-1]["segment_end"], start))
        result.append({**seg, "segment_start": start, "segment_end": end})

    if result and result[-1]["segment_end"] < total_duration:
        result.append(make_no_threat(result[-1]["segment_end"], total_duration))

    # Final tail clamp. Make a copy (not an in-place edit) so the rest
    # of this function's "every appended seg is freshly built" invariant
    # holds end-to-end. This matters when callers retain references to
    # the input list — the previous in-place edit could mutate the
    # original last segment dict.
    if result and result[-1]["segment_end"] != total_duration:
        result[-1] = {**result[-1], "segment_end": total_duration}

    return result


def merge_adjacent_same_type(
    segments: list[dict],
    max_threat_duration: Optional[int] = None,
) -> list[dict]:
    """
    Merge consecutive segments that share the same hasThreat + color.
    Threat merges are capped at max_threat_duration. No-threat segments
    are merged freely.

    Defaults to C.MAX_THREAT_DUR so the cap stays consistent with the
    rest of the pipeline when callers omit the argument.
    """
    if max_threat_duration is None:
        max_threat_duration = C.MAX_THREAT_DUR
    if not segments:
        return []
    merged = [dict(segments[0])]
    for cur in segments[1:]:
        prev = merged[-1]
        same = (
            cur["segmentHasThreat"] == prev["segmentHasThreat"]
            and cur["threat_goalie_color"] == prev["threat_goalie_color"]
        )
        if same:
            if prev["segmentHasThreat"]:
                combined = cur["segment_end"] - prev["segment_start"]
                if combined > max_threat_duration:
                    merged.append(dict(cur))
                    continue
            prev["segment_end"] = max(prev["segment_end"], cur["segment_end"])
        else:
            merged.append(dict(cur))
    return merged


def split_long_threats(
    segments: list[dict],
    max_duration: Optional[int] = None,
) -> list[dict]:
    """
    Split any threat segment longer than max_duration seconds into
    max_duration-sized chunks. No-threat segments pass through.

    Defaults to C.MAX_THREAT_DUR.
    """
    if max_duration is None:
        max_duration = C.MAX_THREAT_DUR
    result = []
    for seg in segments:
        if not seg["segmentHasThreat"]:
            result.append(seg)
            continue
        duration = seg["segment_end"] - seg["segment_start"]
        if duration <= max_duration:
            result.append(seg)
            continue
        cursor = seg["segment_start"]
        while cursor < seg["segment_end"]:
            chunk_end = min(cursor + max_duration, seg["segment_end"])
            result.append({**seg, "segment_start": cursor, "segment_end": chunk_end})
            cursor = chunk_end
    return result


def split_segments_at_period_boundaries(
    segments: list[dict],
    periods_config: Optional[list],
) -> list[dict]:
    """
    Split any segment that straddles a period boundary into two
    sub-segments, and drop any time that falls outside all periods
    (intermissions).

    If periods_config is missing or empty, segments are returned
    unchanged.
    """
    if not periods_config or not isinstance(periods_config, list):
        return segments

    clean_periods: list[tuple[int, int]] = []
    for p in periods_config:
        if not isinstance(p, dict):
            continue
        p_start = p.get("start")
        p_end   = p.get("end")
        if p_start is None or p_end is None:
            continue
        s, e = int(p_start), int(p_end)
        if e > s:
            clean_periods.append((s, e))
    clean_periods.sort()

    if not clean_periods:
        return segments

    result: list[dict] = []
    for seg in segments:
        seg_s = int(seg.get("segment_start", 0))
        seg_e = int(seg.get("segment_end", 0))
        if seg_e <= seg_s:
            continue

        emitted_any = False
        for p_start, p_end in clean_periods:
            if seg_e <= p_start or seg_s >= p_end:
                continue
            sub_s = max(seg_s, p_start)
            sub_e = min(seg_e, p_end)
            if sub_e <= sub_s:
                continue
            if sub_s == seg_s and sub_e == seg_e:
                result.append(seg)
            else:
                new_seg = dict(seg)
                new_seg["segment_start"] = sub_s
                new_seg["segment_end"]   = sub_e
                result.append(new_seg)
            emitted_any = True

        if not emitted_any:
            log.debug(f"  Dropping segment {seg_s}-{seg_e}s — outside all periods")

    result.sort(key=lambda s: (s["segment_start"], s["segment_end"]))
    return result


def cap_segment_length(
    segments: list[dict],
    max_duration: int = 90,
    overlap: int = 15,
    motion_ratio_threshold: float = 1.10,
    min_duration_to_cap: int = 100,
) -> list[dict]:
    """
    Split long, low-confidence threat segments into overlapping
    sub-segments. Currently NOT called by the default pipeline (Option-1
    alone outperforms cap-split in eval), but retained for future
    investigation.
    """
    if max_duration <= overlap:
        raise ValueError(f"max_duration ({max_duration}) must exceed overlap ({overlap})")

    stride = max_duration - overlap
    result: list[dict] = []

    for seg in segments:
        if not seg.get("segmentHasThreat"):
            result.append(seg)
            continue

        seg_s = int(seg["segment_start"])
        seg_e = int(seg["segment_end"])
        duration = seg_e - seg_s

        motion_ratio = float(seg.get("_motion_ratio", 1.0))
        should_split = (
            duration > min_duration_to_cap
            and motion_ratio < motion_ratio_threshold
        )

        if not should_split:
            result.append(seg)
            continue

        cur = seg_s
        sub_idx = 0
        while cur < seg_e:
            sub_end = min(cur + max_duration, seg_e)
            sub = dict(seg)
            sub["segment_start"] = cur
            sub["segment_end"] = sub_end
            sub["_capped_parent"] = f"{seg_s}-{seg_e}"
            sub["_capped_index"] = sub_idx
            result.append(sub)
            sub_idx += 1
            if sub_end >= seg_e:
                break
            cur += stride

        log.info(
            f"  Cap-split {seg_s}-{seg_e}s ({duration}s, ratio {motion_ratio:.2f}) "
            f"→ {sub_idx} sub-segments"
        )

    result.sort(key=lambda s: (s["segment_start"], s["segment_end"]))
    return result


def apply_side_assignments(
    segments: list[dict],
    side_map: dict,
    swap_events: list,
    period_side_maps: Optional[list[tuple[int, dict]]] = None,
) -> list[dict]:
    """
    Assign threat_goalie_side to every segment from the side_map.

    Honours period_side_maps (list of (start_sec, side_map) tuples)
    when supplied. For each segment, the side map active at
    segment_start is used.
    """
    result = []
    for seg in segments:
        seg = dict(seg)
        if seg["segmentHasThreat"] and seg["threat_goalie_color"]:
            active_map = side_map_at(
                seg.get("segment_start", 0), side_map, period_side_maps
            )
            seg["threat_goalie_side"] = active_map.get(seg["threat_goalie_color"])
        else:
            seg["threat_goalie_side"] = None
        result.append(seg)
    return result
