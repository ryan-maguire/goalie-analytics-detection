"""
Jersey colour helpers.

Mapping a jersey colour name (e.g. "White and Blue") to an HSV range
that picks out distinctive pixels of that jersey. Used for game-start
side detection. Per-window attribution in v23 is motion-based and does
not depend on these ranges.
"""

import numpy as np

_LIGHT_KEYWORDS = {"white", "light", "yellow", "gold", "silver", "cream"}
_DARK_KEYWORDS  = {"black", "dark", "navy", "red", "green", "blue", "maroon",
                   "purple", "brown", "grey", "gray"}


def is_light_jersey(color_str: str) -> bool:
    """True if the jersey colour name reads as a light colour."""
    lower = color_str.lower()
    light_score = sum(1 for k in _LIGHT_KEYWORDS if k in lower)
    dark_score  = sum(1 for k in _DARK_KEYWORDS  if k in lower)
    return light_score >= dark_score


def jersey_color_to_hsv_range(color_str: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Return a (lower_hsv, upper_hsv) tuple for the MOST DISTINCTIVE colour
    in a jersey colour name string.

    Key design principle: when a jersey name contains multiple colours,
    use the most distinctive one (highest HSV hue specificity), NOT white
    or black. White and black have very broad HSV ranges that match
    almost every pixel in a broadcast frame, causing false dominance.

    Priority order: red > blue > green > yellow/gold > navy > orange
                    > purple > white > black.
    This ensures "White and Blue" uses blue (distinctive), not white
    (broad), and "Black and Red" uses red (distinctive), not black
    (broad).

    Saturation thresholds are deliberately relaxed (≥40–60) for broadcast
    video where MPEG compression and camera angle desaturate jersey
    colours.
    """
    lower = color_str.lower()
    if "red" in lower:
        return (np.array([0,   60,  60]),  np.array([10,  255, 255]))
    if "blue" in lower:
        return (np.array([100, 40,  40]),  np.array([140, 255, 255]))
    if "green" in lower:
        # v23: inverted saturation bound for desaturated-jersey feeds.
        # On some feeds jersey green reads S≈17 while arena LEDs/walls
        # read S≈200. A sat FLOOR matches walls; a sat CEILING matches
        # jerseys. Narrow hue (45-75) + V ceiling (200) keeps us out of
        # bright ice. This range is used only as a tiebreak in period
        # side detection in v23.
        return (np.array([45, 0, 50]), np.array([75, 140, 200]))
    if "yellow" in lower or "gold" in lower:
        return (np.array([20,  60,  80]),  np.array([35,  255, 255]))
    if "navy" in lower:
        return (np.array([105, 60,  20]),  np.array([135, 255, 140]))
    if "orange" in lower:
        return (np.array([10,  80,  80]),  np.array([25,  255, 255]))
    if "purple" in lower or "violet" in lower:
        return (np.array([130, 60,  40]),  np.array([160, 255, 255]))
    if "white" in lower:
        return (np.array([0,   0,   180]), np.array([180, 40,  255]))
    if "black" in lower:
        return (np.array([0,   0,   0]),   np.array([180, 255, 55]))
    # Generic fallback
    return (np.array([0, 0, 80]), np.array([180, 255, 200]))
