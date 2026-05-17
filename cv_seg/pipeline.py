"""
Pipeline orchestrator.

process_video() is the top-level entry point that wires together signal
extraction, audio analysis, candidate window assembly, attribution,
and post-processing into a complete CV segment-detection run.
"""

import json
import os
import time
from math import ceil
from typing import Optional, Union

from . import constants as C
from .attribution import (
    detect_goalie_sides_cv,
    detect_period_side_maps,
    assign_goalie_colors,
)
from .audio import detect_whistles, detect_crowd_roar_spikes
from .colors import is_light_jersey
from .io_utils import (
    HAS_GCS,
    extract_audio_wav,
    gcs_download_to_temp,
    gcs_write_json,
    get_video_duration,
    load_audio,
    load_audio_via_ffmpeg_pipe,
)
from .logger import log
from .postprocess import (
    apply_side_assignments,
    enforce_boundaries,
    make_no_threat,
    merge_adjacent_same_type,
    merge_overlapping_segments,
    split_long_threats,
    split_segments_at_period_boundaries,
)
from .signals import extract_frame_signals
from .windows import build_candidate_windows


def _build_signal_trace(
    final_segments: list[dict],
    raw_windows: list[dict],
) -> dict:
    """
    Map each final threat segment back to the raw-window source signals
    that contributed to it, by interval overlap.

    A final threat segment can be the merge of multiple raw windows
    from different sources (e.g. a "motion" run that overlapped a
    "celebration" cluster). We record the SET of source tags that
    overlap, not the count, so the trace remains stable under
    post-processing splits — splitting one segment into two preserves
    the source list on each piece if both still overlap the same raw
    windows.

    Returns a JSON-serialisable dict:
      {
        "version": "1",
        "n_raw_windows": <int>,
        "n_final_threats": <int>,
        "segments": [
          {
            "segment_start": float,
            "segment_end":   float,
            "threat_goalie_color": str | None,
            "source_signals":      ["motion", "celebration", ...],
            "n_overlapping_raw":   <int>,
          },
          ...
        ]
      }

    Only threat segments are included; no-threat segments aren't
    interesting for FP analysis. Order matches `final_segments`.
    """
    # Sort raw windows once; final segments may not be sorted but each
    # one independently scans the full raw list anyway. Linear scan
    # per segment is fine — final_segments and raw_windows are both
    # bounded by ~hundreds, so O(N*M) ≈ 10^4–10^5 ops at worst.
    out_segments = []
    for seg in final_segments:
        if not seg.get("segmentHasThreat"):
            continue
        s_start = seg["segment_start"]
        s_end   = seg["segment_end"]

        sources: set[str] = set()
        n_overlap = 0
        for rw in raw_windows:
            r_start = rw["start"]
            r_end   = rw["end"]
            # Half-open interval overlap. Touching boundaries don't count
            # (an end-to-start adjacency isn't really "this signal caused
            # that segment" — it's two distinct things).
            if r_end > s_start and r_start < s_end:
                src = rw.get("source")
                if src:
                    sources.add(src)
                n_overlap += 1

        out_segments.append({
            "segment_start":       s_start,
            "segment_end":         s_end,
            "threat_goalie_color": seg.get("threat_goalie_color"),
            "source_signals":      sorted(sources),
            "n_overlapping_raw":   n_overlap,
        })

    return {
        "version":          "1",
        "n_raw_windows":    len(raw_windows),
        "n_final_threats":  len(out_segments),
        "segments":         out_segments,
    }


def process_video(
    vID:          str,
    config:       Union[list, dict],
    local_video:  Optional[str] = None,
    write_gcs:    bool = True,
    output_dir:   Optional[str] = None,
    use_net_detection: bool = True,
    target_filter: bool = True,
) -> bool:
    """
    Run the full CV segment-detection pipeline for one video.

    Returns True on success, False on configuration or input failure.

    Args:
        use_net_detection: If True, use the HockeyAI YOLOv8 model
            (downloaded lazily from Hugging Face on first use) as the
            primary attribution signal, with motion asymmetry as
            fallback. Requires `ultralytics` and `huggingface_hub` to
            be installed. Defaults to False (v23.6.1 baseline behaviour).
        target_filter: If True (the default), drop segments not
            attributed to the target goalie. Both opponent-threat
            segments and no-threat segments are removed. The metadata
            block records the pre/post-filter counts so the filtering
            is auditable. Set False with --no-target-filter to keep all
            segments (useful for debugging attribution issues).
    """
    log.info("=" * 60)
    log.info(f"CV pipeline: {vID}")
    log.info("=" * 60)
    t_start = time.time()

    # ── 1. Resolve jersey colours from config ───────────────────────────
    if isinstance(config, list):
        record = next((r for r in config if str(r.get("vID")) == str(vID)), None)
        if record is None:
            log.error(f"[{vID}] No config record found — skipping.")
            return False
    else:
        record = config

    color_1 = record.get("targetGoalieColor")
    color_2 = record.get("opponentGoalieColor")
    if not color_1 or not color_2:
        log.error(f"[{vID}] Missing targetGoalieColor/opponentGoalieColor — skipping.")
        return False

    # Assign light/dark ordering (A = lighter)
    if is_light_jersey(color_1) and not is_light_jersey(color_2):
        goalie_color_a, goalie_color_b = color_1, color_2
    elif is_light_jersey(color_2) and not is_light_jersey(color_1):
        goalie_color_a, goalie_color_b = color_2, color_1
    else:
        goalie_color_a, goalie_color_b = tuple(sorted([color_1, color_2]))

    log.info(f"[{vID}] Colours: A={goalie_color_a}  B={goalie_color_b}")

    override_side_a = record.get("targetStartSide")
    override_side_b = record.get("opponentStartSide")

    # ── 2. Get local video path ─────────────────────────────────────────
    cleanup_video = False
    if local_video and os.path.exists(local_video):
        video_path = local_video
        log.info(f"[{vID}] Using local video: {video_path}")
    else:
        local_candidate = os.path.join("data", "full_video", f"full_{vID}.mp4")
        if os.path.exists(local_candidate):
            video_path = local_candidate
            log.info(f"[{vID}] Using local video: {video_path}")
        elif HAS_GCS:
            video_blob = f"{C.VIDEO_PREFIX}/full_{vID}.mp4"
            try:
                # ORDER-SENSITIVE: assign video_path FIRST, then set the
                # cleanup flag. If the download raises, video_path stays
                # unbound and cleanup_video stays False, so the finally
                # block at the bottom of process_video has no orphan to
                # touch.
                video_path   = gcs_download_to_temp(C.GCS_BUCKET, video_blob)
                cleanup_video = True
            except Exception as e:
                log.error(f"[{vID}] GCS download failed: {e} — skipping.")
                return False
        else:
            log.error(f"[{vID}] No local video found and GCS unavailable — skipping.")
            return False

    try:
        duration = get_video_duration(video_path)
        dur_int  = ceil(duration)
        log.info(f"[{vID}] Duration: {dur_int}s")

        # ── 3. Detect goalie sides ──────────────────────────────────────
        if override_side_a and override_side_b:
            tss_low = override_side_a.lower()
            oss_low = override_side_b.lower()
            if color_1 == goalie_color_a:
                initial_side_map = {
                    goalie_color_a: tss_low,
                    goalie_color_b: oss_low,
                }
            else:
                initial_side_map = {
                    goalie_color_b: tss_low,
                    goalie_color_a: oss_low,
                }
            log.info(f"[{vID}] Using config side override: {initial_side_map}")
        else:
            initial_side_map = detect_goalie_sides_cv(
                video_path, goalie_color_a, goalie_color_b
            )

        # ── 3b. Detect period side swaps ────────────────────────────────
        log.info(f"[{vID}] Detecting period side maps...")
        periods_config = record.get("periods")
        period_side_maps = detect_period_side_maps(
            video_path, goalie_color_a, goalie_color_b, duration,
            periods_config=periods_config,
            target_start_side=override_side_a,
            opponent_start_side=override_side_b,
            target_color=color_1,
            opponent_color=color_2,
        )
        if not period_side_maps:
            period_side_maps = [(0, initial_side_map)]

        # ── 4. Extract frame signals ────────────────────────────────────
        log.info(f"[{vID}] Extracting frame signals...")
        signals, _ = extract_frame_signals(video_path, sample_fps=1)

        # ── 5. Audio: load once via the ffmpeg pipe (or WAV fallback),
        #          detect whistles + crowd roar ────────────────────────
        wav_path    = None
        whistles:     list[float] = []
        crowd_spikes: list[float] = []
        try:
            log.info(f"[{vID}] Loading audio...")
            y_audio, sr_audio = load_audio_via_ffmpeg_pipe(video_path, sr=16000)
            if y_audio is None:
                # Fallback: extract a WAV to disk, then read it back.
                log.info(f"[{vID}] ffmpeg-pipe path unavailable — using WAV roundtrip")
                wav_path = extract_audio_wav(video_path)
                y_audio, sr_audio = load_audio(wav_path, sr=16000)
            whistles     = detect_whistles(y_audio, sr_audio, duration)
            crowd_spikes = detect_crowd_roar_spikes(y_audio, sr_audio, duration)
        except Exception as e:
            log.warning(f"[{vID}] Audio processing failed (non-fatal): {e}")
        finally:
            if wav_path and os.path.exists(wav_path):
                os.unlink(wav_path)

        # ── 6. Build candidate windows ─────────────────────────────────
        log.info(f"[{vID}] Building candidate windows...")
        raw_windows = build_candidate_windows(signals, whistles, crowd_spikes, duration)

        # ── 7. Assign goalie colours ───────────────────────────────────
        log.info(f"[{vID}] Assigning goalie colours to {len(raw_windows)} windows...")
        threat_segs = assign_goalie_colors(
            raw_windows,
            goalie_color_a, goalie_color_b,
            initial_side_map, signals, duration,
            period_side_maps=period_side_maps,
            target_color=color_1,
            video_path=video_path,
            use_net_detection=use_net_detection,
        )

        threat_segs = [
            s for s in threat_segs
            if (s["segment_end"] - s["segment_start"]) >= C.MIN_THREAT_DUR
        ]

        # Drop windows fully inside the first 20s unless caused by a
        # hard trigger (faceoff). "Fully inside" means segment_end <= 20
        # (a segment ending at second 20 has not yet crossed the 20s
        # mark). Goal-light hard triggers were removed in v23.5.
        hard_trigger_times = set(
            s["t"] for s in signals
            if s["faceoff"] >= C.FACEOFF_HIGH_CONFIDENCE
        )
        def _is_opening_false_positive(seg: dict) -> bool:
            if seg["segment_end"] > 20:
                return False
            return not any(
                seg["segment_start"] <= t <= seg["segment_end"]
                for t in hard_trigger_times
            )
        threat_segs = [s for s in threat_segs if not _is_opening_false_positive(s)]
        threat_segs.sort(key=lambda s: s["segment_start"])

        # ── 8. Post-processing ─────────────────────────────────────────
        log.info(f"[{vID}] Post-processing...")
        final = merge_overlapping_segments(threat_segs)
        final = enforce_boundaries(final, dur_int)
        final = merge_adjacent_same_type(final)
        final = split_long_threats(final, max_duration=C.MAX_THREAT_DUR)

        # Demote tiny threat slivers to no-threat
        cleaned = []
        for seg in final:
            if (seg["segmentHasThreat"]
                    and (seg["segment_end"] - seg["segment_start"]) < C.MIN_KEEP_DUR):
                cleaned.append(make_no_threat(seg["segment_start"], seg["segment_end"]))
            else:
                cleaned.append(seg)
        final = merge_adjacent_same_type(cleaned)

        # Clip to period boundaries (drops intermission time)
        final = split_segments_at_period_boundaries(final, periods_config)

        # NOTE: cap_segment_length() is intentionally NOT called here.
        # Option 2 (cap-split) was tried and reverted — F1=0.69 without
        # vs F1=0.56 with cap-split enabled.

        final = apply_side_assignments(
            final, initial_side_map, swap_events=[],
            period_side_maps=period_side_maps,
        )

        # ── 8b. Target-color filter ────────────────────────────────────
        # Drop segments not attributed to the target goalie. Both
        # opponent-threat segments and no-threat segments are removed —
        # the deliverable for this customer only covers their goalie's
        # threats. This is a cost-control measure: feeding only target
        # segments to metrics_seg and feedback_seg roughly halves their
        # Gemini spend per video.
        #
        # The pre-filter list is preserved in metadata for auditability.
        # Pass target_filter=False (CLI: --no-target-filter) to skip
        # this and write all segments — useful when debugging
        # attribution mistakes (e.g. "why did this NSW shot get tagged
        # as Amherst-defending?").
        target_color = color_1  # color_1 is targetGoalieColor (line 157)
        prefilter_total      = len(final)
        prefilter_threat     = sum(1 for s in final if s["segmentHasThreat"])
        prefilter_target     = sum(1 for s in final
                                   if s["segmentHasThreat"]
                                   and s["threat_goalie_color"] == target_color)
        prefilter_opponent   = prefilter_threat - prefilter_target
        prefilter_no_threat  = prefilter_total - prefilter_threat

        if target_filter:
            final = [s for s in final
                     if s["segmentHasThreat"]
                     and s["threat_goalie_color"] == target_color]
            log.info(
                f"[{vID}] Target-color filter applied "
                f"(target={target_color!r}): "
                f"{prefilter_total} → {len(final)} segments "
                f"(dropped {prefilter_opponent} opponent-threat, "
                f"{prefilter_no_threat} no-threat)"
            )
        else:
            log.info(
                f"[{vID}] Target-color filter SKIPPED "
                f"(--no-target-filter): keeping all {len(final)} segments"
            )

        # ── 9. Build metadata ──────────────────────────────────────────
        threat_count    = sum(1 for s in final if s["segmentHasThreat"])
        no_threat_count = len(final) - threat_count
        color_duration: dict[str, int] = {}
        for seg in final:
            if seg["segmentHasThreat"] and seg["threat_goalie_color"]:
                c = seg["threat_goalie_color"]
                color_duration[c] = color_duration.get(c, 0) + (
                    seg["segment_end"] - seg["segment_start"]
                )

        elapsed = time.time() - t_start
        meta = {
            "vID":                      vID,
            "method":                   "openCV",
            "processed_at":             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "video_duration_sec":       dur_int,
            "processing_time_sec":      round(elapsed, 1),
            "goalie_color_a":           goalie_color_a,
            "goalie_color_b":           goalie_color_b,
            "initial_side_map":         initial_side_map,
            "total_segments":           len(final),
            "threat_segments":          threat_count,
            "no_threat_segments":       no_threat_count,
            "threat_duration_by_color_sec": color_duration,
            "signals_summary": {
                # goal_light_events removed in v23.5 (red_light detector dropped)
                "faceoff_events":       sum(1 for s in signals if s["faceoff"] >= C.FACEOFF_HIGH_CONFIDENCE),
                "whistle_events":       len(whistles),
                "crowd_roar_events":    len(crowd_spikes),
                "celebration_events":   sum(1 for s in signals if s.get("celeb", 0) > 0),
                "raw_candidate_windows": len(raw_windows),
            },
            "thresholds": {
                # RED_LIGHT_THRESH removed in v23.5
                "FACEOFF_THRESH":       C.FACEOFF_HIGH_CONFIDENCE,
                "MOTION_THRESH":        C.MOTION_THRESH,
                "WHISTLE_ENERGY_THRESH": C.WHISTLE_ENERGY_THRESH,
                "MIN_THREAT_DUR":       C.MIN_THREAT_DUR,
                "MAX_THREAT_DUR":       C.MAX_THREAT_DUR,
                "MAX_OPEN_WINDOW_SEC":  C.MAX_OPEN_WINDOW_SEC,
            },
            "target_filter": {
                "applied":              target_filter,
                "target_color":         target_color,
                "prefilter_total":      prefilter_total,
                "prefilter_threat":     prefilter_threat,
                "prefilter_target":     prefilter_target,
                "prefilter_opponent":   prefilter_opponent,
                "prefilter_no_threat":  prefilter_no_threat,
                "postfilter_total":     len(final),
            },
        }

        log.info(f"[{vID}] Final: {len(final)} segments "
                 f"({threat_count} threat, {no_threat_count} no-threat) "
                 f"in {elapsed:.0f}s")

        # ── Diagnostic: source-signal trace ────────────────────────────
        # For each FINAL threat segment, list every raw-candidate-window
        # source (motion / faceoff / crowd_roar / celebration /
        # motion_auto_close / motion_eof) that overlaps it. This is
        # the data the eval needs to attribute FPs back to the
        # originating signal — written as a sidecar so the main
        # gt_seg_*.json output remains stable.
        # (goal_light and activity_spike sources removed in v23.5.)
        trace = _build_signal_trace(final, raw_windows)

        # v23.10: also include the raw per-second signal vectors so
        # downstream tooling (shot-centering snapshot, future window-
        # refinement code) can analyse motion intensity inside each
        # window without re-extracting from the video. Adds ~10-50 KB
        # per file at 1Hz sampling. Existing consumers see the same
        # `segments` and `version` fields — `per_second` is additive.
        trace["per_second"] = [
            {
                "t":        int(s.get("t", i)),
                "motion":   round(float(s.get("motion",   0.0)), 3),
                "faceoff":  round(float(s.get("faceoff",  0.0)), 3),
                "activity": round(float(s.get("activity", 0.0)), 3),
            }
            for i, s in enumerate(signals)
        ]

        # ── 10. Write output ───────────────────────────────────────────
        seg_blob   = f"{C.OUTPUT_PREFIX}/gt_seg_{vID}.json"
        meta_blob  = f"{C.OUTPUT_PREFIX}/gt_seg_{vID}_meta.json"
        trace_blob = f"{C.OUTPUT_PREFIX}/gt_seg_{vID}_signals.json"

        if write_gcs and HAS_GCS:
            gcs_write_json(C.GCS_BUCKET, seg_blob,   final)
            gcs_write_json(C.GCS_BUCKET, meta_blob,  meta)
            gcs_write_json(C.GCS_BUCKET, trace_blob, trace)

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            seg_path   = os.path.join(output_dir, f"gt_seg_{vID}.json")
            meta_path  = os.path.join(output_dir, f"gt_seg_{vID}_meta.json")
            trace_path = os.path.join(output_dir, f"gt_seg_{vID}_signals.json")
            with open(seg_path,   "w", encoding="utf-8") as f:
                json.dump(final, f, indent=2, ensure_ascii=False)
            with open(meta_path,  "w", encoding="utf-8") as f:
                json.dump(meta,  f, indent=2, ensure_ascii=False)
            with open(trace_path, "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
            log.info(f"[{vID}] Written to {output_dir}/")

        return True

    finally:
        if cleanup_video and os.path.exists(video_path):
            os.unlink(video_path)
