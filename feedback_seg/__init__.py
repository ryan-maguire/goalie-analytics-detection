"""feedback_seg — Stage 4 of the goalie analytics pipeline.

Reads the segment metrics output from `metrics_seg`, re-extracts each
threat window as a short clip, and sends it to Gemini with a
"professional goaltending scout/coach" prompt. Produces qualitative
coaching feedback per window plus a game-level coaches' summary.

Pipeline stage flow:
    cv_seg → metrics_seg → feedback_seg
"""

__version__ = "1.2.2"
