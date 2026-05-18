"""
Tunable thresholds for the CV goalie segment detector.

Every threshold has a comment explaining what it controls and how it was
calibrated. Keep all tunables here so the rest of the package never
embeds a magic number.

Many of these values are exposed via CLI flags (see cli.py) and can be
overridden at runtime.
"""

# ---------------------------------------------------------------------------
# GCS / config
# ---------------------------------------------------------------------------

GCS_BUCKET    = "goalie_video_bucket"
VIDEO_PREFIX  = "ground_truth_video/full_video"
OUTPUT_PREFIX = "analyze_video/01-segment_detection"

# ---------------------------------------------------------------------------
# Goal-light detection
# ---------------------------------------------------------------------------
# Enlarged ROI (10% vs 6%) catches feeds where the light is not in the
# extreme corner. Lowered pixel threshold (4% vs 8%) catches dimmer/smaller
# lights. A temporal spike filter (RED_LIGHT_MIN_FRAMES) is applied on top:
# a single-frame blip is ignored; the light must be visible for ≥N
# consecutive sampled seconds.
RED_LIGHT_THRESH      = 0.04    # 4% of the corner ROI must be saturated red
RED_LIGHT_ROI_FRAC    = 0.10    # ROI height/width as fraction of frame
RED_LIGHT_MIN_FRAMES  = 2       # must appear in ≥N consecutive 1fps frames

# ---------------------------------------------------------------------------
# Centre-ice faceoff detection
# ---------------------------------------------------------------------------
# Lower accumulator threshold (20 vs 40) catches the circle at broadcast
# resolutions where it is small and partially occluded by players.
# Looser density ratio (0.5 vs 0.7) still requires rough symmetry.
FACEOFF_CIRCLE_THRESH = 20      # HoughCircles accumulator threshold
FACEOFF_DENSITY_RATIO = 0.5     # left/right player density ratio
FACEOFF_HIGH_CONFIDENCE = 0.8   # threshold above which a faceoff is considered "definite"

# ---------------------------------------------------------------------------
# Optical flow / motion thresholds
# ---------------------------------------------------------------------------
MOTION_THRESH         = 3.0     # mean flow magnitude to consider "active"
MOTION_STOP_THRESH    = 1.2     # below this = play stopped
MIN_MOTION_RUN_SEC    = 8       # consecutive active seconds to open a threat window
MAX_OPEN_WINDOW_SEC   = 30      # auto-close a window if it stays open this long.
                                # Lowered from 90 → 45 in v23.3 alongside
                                # the unconfirmed-autoclose suppression.
                                # See EVAL_NOTES.md.
                                # v24.0: 45 → 30. 64.8% of fast-set FPs were
                                # exactly 45s MAC windows. Shorter cap reduces
                                # MAC window duration and improves IoU with GT.

# motion_auto_close FPs accounted for 87% of all FPs across a 14-video
# eval (482 of 553 FPs). The pre-existing "unconfirmed motion_auto_close
# suppression" already drops MAC windows with no co-confirming signal,
# but the confirmation rule is "any overlap, however brief, counts" —
# so a 1-second whistle inside a 60-second motion blob keeps the entire
# blob. Requiring the confirmation to overlap by at least
# MIN_CONFIRMATION_OVERLAP_SEC tightens this without changing which
# signals act as confirmers. Set to 0 to disable (pre-v23.8 behaviour).
#
# v23.8 ran with both values = 4. Result on 14 video eval set:
#   strict F1: 0.23 → 0.41, motion_auto_close FPs: 482 → 405 (-16%),
#   no recall regression. Net win.
# v23.9 raises both to 6 in lockstep — meaningfully stronger overlap
# requirement, but single whistle/activity_spike events still confirm
# because the event-width widening keeps pace with the overlap threshold.
#   Why not raise overlap alone? Setting overlap=6 with event_width=4
#   would silently disable single-event confirmation, which v23.5
#   history shows tanks recall (0.97 → 0.50 on the same eval set).
MIN_CONFIRMATION_OVERLAP_SEC = 6

# Whistles and activity_spikes are zero-duration events but genuinely
# correlate with real play. Widen each to this many seconds when used
# as motion_auto_close confirmation, so they survive the new minimum-
# overlap rule above. A whistle at second 500 becomes a confirmation
# interval [500, 500+CONFIRMATION_EVENT_WIDTH_SEC).
#
# Kept in lockstep with MIN_CONFIRMATION_OVERLAP_SEC — see comment above.
CONFIRMATION_EVENT_WIDTH_SEC = 6

# ---------------------------------------------------------------------------
# Whistle detection (audio)
# ---------------------------------------------------------------------------
# WHISTLE_ENERGY_THRESH was raised from 0.55 → 2.5 z-scores. At 0.55 nearly
# half the game triggers as "whistle" because skate scraping and puck impacts
# also contain 2–4kHz energy. Real referee whistles produce sharp, brief
# spikes well above the game's background noise floor. 2.5 z-scores selects
# only those sharp peaks (top ~1% of the distribution).
# WHISTLE_MIN_DUR_SEC was raised from 0.3s → 0.8s to reject brief transients.
# WHISTLE_REFRACTORY_SEC: minimum gap between two separate whistle events —
# prevents one long crowd-noise burst from generating dozens of events.
WHISTLE_FREQ_LOW        = 2000  # Hz — lower bound of whistle detection band
WHISTLE_FREQ_HIGH       = 4500  # Hz — upper bound
WHISTLE_ENERGY_THRESH   = 2.5   # z-score threshold (was 0.55 — far too low)
WHISTLE_MIN_DUR_SEC     = 0.8   # minimum sustained whistle duration
WHISTLE_REFRACTORY_SEC  = 3.0   # minimum gap between distinct whistle events
# WHISTLE_GRACE_SEC — when a whistle falls inside an open window, the
# window is trimmed to (whistle_t + WHISTLE_GRACE_SEC). 12s gives enough
# room for goal-mouth scrambles and rebounds. History: 5s → 8s → 12s.
WHISTLE_GRACE_SEC       = 12

# ---------------------------------------------------------------------------
# Bench/crowd activity (visual proxy)
# ---------------------------------------------------------------------------
ACTIVITY_THRESH       = 0.15    # relative brightness spike in bench ROI
ACTIVITY_ROI_FRAC     = 0.10    # top fraction of frame = bench/crowd area

# ---------------------------------------------------------------------------
# Crowd roar detection (audio)
# ---------------------------------------------------------------------------
# Goal celebrations produce a broad low-frequency roar distinct from
# whistles. Z-score normalised so the threshold is independent of arena
# acoustics. Small-arena feeds have lower baseline noise so goal
# celebrations only reach z≈2.0–2.4; raising the threshold to 2.5
# eliminated real goal spikes on those feeds.
CROWD_FREQ_LOW        = 20      # Hz — lower bound (exclude DC/rumble)
CROWD_FREQ_HIGH       = 500     # Hz — upper bound (goal roar band)
CROWD_ENERGY_THRESH   = 2.0     # z-score threshold
CROWD_MIN_DUR_SEC     = 3.0     # must be sustained ≥3.0s
CROWD_REFRACTORY_SEC  = 10.0    # minimum gap between separate crowd events

# ---------------------------------------------------------------------------
# Celebration clustering (visual, asymmetric player density after a goal)
# ---------------------------------------------------------------------------
CELEBRATION_DENSITY_RATIO   = 3.0   # dominant half must have ≥3.0× other half
CELEBRATION_MIN_DENSITY     = 0.03  # absolute dark-pixel fraction
CELEBRATION_MOTION_MIN      = 1.5   # require some motion
CELEBRATION_MIN_RUN_SEC     = 5     # consecutive seconds to confirm a celebration

# ---------------------------------------------------------------------------
# Motion-asymmetry attribution (v23)
# ---------------------------------------------------------------------------
# Real OZ-pressure ratios sit in the 1.15-1.25 band on broadcast hockey,
# with neutral-zone ambiguity at 1.00-1.10. 1.25 was too strict (only
# 17% "motion" decisions); 1.15 admits real asymmetry while still
# rejecting noise.
MOTION_ATTR_RATIO       = 1.15   # right must be ≥ 1.15× left (or vice versa)
MOTION_ATTR_ABS_FLOOR   = 0.5    # absolute diff must exceed this
MOTION_ATTR_PRE_ROLL_SEC = 10    # seconds before window start to include

# ---------------------------------------------------------------------------
# Post-processing limits
# ---------------------------------------------------------------------------
MIN_THREAT_DUR        = 15      # seconds — drop threat windows shorter than this
MAX_THREAT_DUR        = 60      # seconds — split windows longer than this.
                                # Tightened from 120 in v23.2 after the
                                # MAX_THREAT_DUR sweep on 3 videos showed
                                # aggregate F1 0.624 → 0.642 going from
                                # 120 → 60. See EVAL_NOTES.md for full
                                # per-video numbers and tradeoff notes.
MIN_KEEP_DUR          = 5       # demote threat segments shorter than this to no-threat

# SIDE_DETECTION_TIE_EPS — when both colours score nearly identical
# left/right fractions, treat the result as a tie and fall back to the
# brightness heuristic. 0.005 matches what was previously hard-coded.
SIDE_DETECTION_TIE_EPS = 0.005

# ---------------------------------------------------------------------------
# Goal ROI sampling (used for game-start side detection)
# ---------------------------------------------------------------------------
# Defaults assume a standard side-on broadcast angle. Using a band
# rather than bottom-only so low-angle cameras (where the ice/crease
# sits at 20-50% of frame height) are handled alongside high-angle feeds
# (crease at 70-100%).
GOAL_ROI_SIDE_FRAC    = 0.25    # left/right quarter of frame width = net zone
GOAL_ROI_TOP_FRAC     = 0.15    # top of crease sampling band
GOAL_ROI_BOTTOM_FRAC  = 1.00    # bottom of crease sampling band
