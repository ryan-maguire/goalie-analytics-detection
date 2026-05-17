"""
Candidate window assembly.

Convert per-second signal vectors and audio events into raw threat-window
candidates. The candidates here are not yet de-duplicated or filtered;
they go through merge_overlapping_segments and friends in postprocess.py
to become the final timeline.
"""

from math import ceil
from typing import Optional

from . import constants as C
from .logger import log


def build_candidate_windows(
    signals: list[dict],
    whistles: list[float],
    crowd_spikes: list[float],
    duration: float,
) -> list[dict]:
    """
    Convert per-second signal vectors and audio events into candidate
    threat windows as raw {start, end, source} dicts.

    Signal sources (in priority order):
      faceoff        — centre-ice faceoff detected (hard trigger)
      crowd_spike    — low-frequency crowd roar burst
      celebration    — asymmetric player clustering
      motion         — sustained optical flow

    Removed / changed in earlier versions:
      goal_light     — red corner light fraction. Removed in v23.5
                       because amateur-rink broadcasts (the bulk of
                       the dataset) lack visible goal lights, while
                       the few games that DO have red imagery (ad
                       boards, scoreboard graphics) generated dozens
                       to hundreds of false events per game.
      activity_spike — bench-area brightness. v23.5 removed entirely
                       (no source, no confirmer); recall collapsed
                       (lenient mid recall 0.97 → 0.50 across 5
                       videos, one video down to 8 predictions for a
                       4080s game). v23.6 restored as confirmer-only:
                       activity-spike timestamps no longer become
                       standalone candidate windows, but they DO
                       count toward motion_auto_close confirmation.

    A single linear walk over `signals` builds the light-run state, the
    motion-window state, the activity-spike list, and the celebration-run
    state simultaneously. The previous implementation iterated `signals`
    four times.
    """
    dur_int = ceil(duration)

    # Single-pass state for each detector tracked in lockstep
    raw_windows: list[dict] = []

    # Motion-run state
    motion_run_start: Optional[int] = None
    motion_run_len = 0
    open_window_start: Optional[int] = None

    # Faceoff times
    faceoff_times: list[int] = []

    # Activity-spike timestamps (v23.6). Tracked here separately from
    # raw_windows so they're available to the motion_auto_close
    # confirmation logic below WITHOUT becoming standalone candidate
    # threat windows themselves. See history in module docstring.
    activity_spike_times: list[int] = []

    # Celebration-run state
    celeb_run_start: Optional[int] = None
    celeb_run_side:  str = "none"
    celeb_run_len:   int = 0
    celeb_windows_added = 0

    for s in signals:
        t = s["t"]

        # red_light tracking removed permanently in v23.5.

        # ── faceoff ──
        if s["faceoff"] >= C.FACEOFF_HIGH_CONFIDENCE:
            faceoff_times.append(t)

        # ── activity spike (timestamp only — not a candidate window) ──
        if s["activity"] >= C.ACTIVITY_THRESH:
            activity_spike_times.append(t)

        # ── motion runs / windows ──
        # Two-threshold hysteresis: open at MOTION_THRESH, close at
        # MOTION_STOP_THRESH. Close threshold can be SET INDEPENDENTLY
        # (it doesn't need to be ≤ open threshold) — that's the whole
        # point of the hysteresis.
        #
        # Earlier versions had the close check gated inside the
        # `motion < MOTION_THRESH` else branch, which made the close
        # threshold effectively pinned to `min(MOTION_STOP_THRESH,
        # MOTION_THRESH)`. That collapsed any close-threshold setting
        # higher than the open threshold to a no-op, which made it
        # impossible to terminate windows during sustained-but-elevated
        # motion (e.g. broadcast camera pans inflating motion baseline).
        # v23.6.1 onwards: close check is independent.
        active = (s["motion"] >= C.MOTION_THRESH)
        if active:
            if motion_run_start is None:
                motion_run_start = t
                motion_run_len = 0
            motion_run_len += 1
            if motion_run_len >= C.MIN_MOTION_RUN_SEC and open_window_start is None:
                open_window_start = motion_run_start
        else:
            motion_run_start = None
            motion_run_len   = 0

        # Independent close check. An open window terminates as soon as
        # motion drops below MOTION_STOP_THRESH, regardless of whether
        # we're still above MOTION_THRESH. The motion-run state is NOT
        # reset on close — if motion stays above MOTION_THRESH after
        # the close, a fresh run can accumulate and open a new window
        # later in the same play.
        if open_window_start is not None and s["motion"] < C.MOTION_STOP_THRESH:
            raw_windows.append({
                "start":  open_window_start,
                "end":    t,
                "source": "motion",
            })
            open_window_start = None

        # Auto-close runaway windows so a long active streak still
        # produces a window. After auto-close, reset the motion-run
        # state so the next active second begins a fresh run rather
        # than immediately re-tripping the same threshold (which would
        # produce one duplicate auto-close window per second). The
        # fresh run still has to clear MIN_MOTION_RUN_SEC before it
        # opens a new window, so genuinely sustained motion produces
        # a sequence of discrete windows aligned to MAX_OPEN_WINDOW_SEC
        # boundaries rather than a flood of overlapping duplicates.
        if open_window_start is not None and (t - open_window_start) >= C.MAX_OPEN_WINDOW_SEC:
            raw_windows.append({
                "start":  open_window_start,
                "end":    t,
                "source": "motion_auto_close",
            })
            open_window_start = None
            motion_run_start  = None
            motion_run_len    = 0

        # ── celebration runs ──
        score = s.get("celeb", 0.0)
        side  = s.get("celeb_side", "none")
        if score > 0.0 and side != "none":
            if celeb_run_start is None or side != celeb_run_side:
                celeb_run_start = t
                celeb_run_side  = side
                celeb_run_len   = 1
            else:
                celeb_run_len += 1

            if celeb_run_len >= C.CELEBRATION_MIN_RUN_SEC and celeb_run_start is not None:
                raw_windows.append({
                    "start":      max(0, celeb_run_start - 20),
                    "end":        min(dur_int, t + 15),
                    "source":     "celebration",
                    "celeb_side": celeb_run_side,
                })
                celeb_windows_added += 1
                celeb_run_start = None
                celeb_run_len   = 0
        else:
            celeb_run_start = None
            celeb_run_len   = 0
            celeb_run_side  = "none"

    # Close any motion window still open at EOF
    if open_window_start is not None:
        raw_windows.append({
            "start":  open_window_start,
            "end":    dur_int,
            "source": "motion_eof",
        })

    log.info(f"  Hard triggers: {len(faceoff_times)} centre faceoffs, "
             f"{len(whistles)} whistles")

    # ── D1: Faceoff hard-trigger windows ────────────────────────────────
    # (goal_light removed in v23.5 — see module docstring)
    for t in faceoff_times:
        raw_windows.append({
            "start":  max(0, t - 25),
            "end":    min(dur_int, t + 15),
            "source": "faceoff",
        })

    # ── D5: Crowd roar spikes ───────────────────────────────────────────
    for ct in crowd_spikes:
        ct_int = int(ct)
        raw_windows.append({
            "start":  max(0, ct_int - 25),
            "end":    min(dur_int, ct_int + 20),
            "source": "crowd_roar",
        })
    if crowd_spikes:
        log.info(f"  Crowd roar: {len(crowd_spikes)} spikes → "
                 f"{len(crowd_spikes)} crowd_roar windows added")

    if celeb_windows_added:
        log.info(f"  Celebration clustering: {celeb_windows_added} windows added")

    # ── Unconfirmed motion_auto_close suppression ──────────────────────
    # motion_auto_close windows are emitted whenever a motion run stays
    # active past MAX_OPEN_WINDOW_SEC. Without independent confirmation
    # from a discrete-event signal these are almost always camera pans
    # / neutral-zone forecheck / general game flow rather than real
    # threats. Confirming signals: faceoff, crowd_roar, celebration,
    # regular motion, motion_eof, whistles, or activity-spike events.
    #
    # History:
    #   v23.3 — first version; activity_spike treated as a confirmer
    #           (alongside being a window source). FPs ballooned.
    #   v23.4 — dropped activity_spike from the confirmer set. FPs
    #           shifted from "MAC blobs" to "standalone activity_spike
    #           windows".
    #   v23.5 — activity_spike removed entirely (no source, no
    #           confirmer). FPs dropped further but RECALL COLLAPSED:
    #           lenient mid recall fell from 0.97 → 0.50 across 5
    #           videos, with one video (n2cy8b755Tg) producing 8
    #           predictions for a 4080s game. Without activity_spike
    #           confirming long motion runs on signal-starved videos,
    #           those runs got suppressed and real threats vanished.
    #   v23.6 — activity_spike timestamps are tracked again, but only
    #           used as confirmers for motion_auto_close. They never
    #           become standalone candidate windows. This is the
    #           middle ground I argued against in v23.4 chat — eval
    #           data made it the obvious answer.
    #
    # If a real threat happens to span a suppressed window, the state
    # machine above opens a fresh motion window the next second motion
    # is active — see the `if motion_run_len >= ...` check earlier.
    other_source_intervals: list[tuple[int, int]] = [
        (w["start"], w["end"]) for w in raw_windows
        if w.get("source") != "motion_auto_close"
    ]
    # Whistles and activity spikes are zero-duration events, but they
    # genuinely correlate with real play. Widen each to a
    # CONFIRMATION_EVENT_WIDTH_SEC interval so the minimum-overlap rule
    # below treats them as meaningful confirmation rather than a 1-second
    # blip that fails the new threshold.
    event_width = max(C.MIN_CONFIRMATION_OVERLAP_SEC,
                      C.CONFIRMATION_EVENT_WIDTH_SEC)
    for wt in whistles:
        wt_int = int(wt)
        other_source_intervals.append((wt_int, wt_int + event_width))
    for at in activity_spike_times:
        other_source_intervals.append((at, at + event_width))

    def _has_co_confirmation(start: int, end: int) -> bool:
        """Return True iff at least one other-source signal overlaps
        [start, end) by at least MIN_CONFIRMATION_OVERLAP_SEC seconds.

        v23.7 and earlier: any overlap, however brief, counted. That
        let a single 1-second whistle inside a 60-second motion blob
        keep the whole blob — empirically the dominant source of FPs.

        v23.8 (this version): require the overlap to be at least
        MIN_CONFIRMATION_OVERLAP_SEC. Set to 0 to revert to the prior
        permissive behaviour.
        """
        min_overlap = C.MIN_CONFIRMATION_OVERLAP_SEC
        for o_start, o_end in other_source_intervals:
            # Half-open intervals; compute the actual overlap duration
            ov_start = max(start, o_start)
            ov_end   = min(end,   o_end)
            overlap  = ov_end - ov_start
            if overlap >= min_overlap:
                return True
        return False

    suppressed = 0
    confirmed_kept: list[dict] = []
    for w in raw_windows:
        if w.get("source") == "motion_auto_close" \
                and not _has_co_confirmation(w["start"], w["end"]):
            suppressed += 1
            continue
        confirmed_kept.append(w)
    raw_windows = confirmed_kept
    if suppressed:
        log.info(f"  Unconfirmed motion_auto_close suppression: dropped {suppressed} windows")

    # ── Whistle-based trimming ──────────────────────────────────────────
    # A whistle inside an open window strongly suggests play stopped.
    # Trim the window to (whistle_t + WHISTLE_GRACE_SEC). Sort whistles
    # ascending so the earliest matching whistle wins, and break on the
    # first hit — the original code worked by accident on sorted input.
    sorted_whistles = sorted(whistles)
    trimmed = []
    for w in raw_windows:
        end = w["end"]
        for wt in sorted_whistles:
            if w["start"] < wt < end:
                end = min(end, int(wt) + C.WHISTLE_GRACE_SEC)
                break  # earliest whistle defines the trim
        trimmed.append({**w, "end": end})

    log.info(f"  Raw candidate windows: {len(trimmed)}")
    return trimmed
