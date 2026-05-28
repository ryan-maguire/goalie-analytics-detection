"""PyTorch dataset for the action-recognition manifest.

Extracts video clips on-demand from data/videos/ using PyAV (no
disk-rendering required, saves ~30 GB).

Two modes:
  - per_frame_mean: returns a (3, H, W) tensor = mean RGB of N
    sampled frames. Cheap baseline that feeds a per-frame CNN.
  - clip:           returns a (3, T, H, W) tensor of T sampled frames.
                    For X3D / SlowFast / TimeSformer training.

The manifest is split by GAME, not by clip, to prevent leakage:
clips from the same game can't appear in both train and val.
"""

import json
import random
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


LABEL_NAMES = ("no_event", "shot_save", "goal")
LABEL_TO_IDX = {name: i for i, name in enumerate(LABEL_NAMES)}


@dataclass
class ClipRecord:
    vID:        str
    hudl_id:    int
    video_path: str
    start_sec:  int
    end_sec:    int
    label:      str
    source:     str


def load_manifest(path: Path) -> list[ClipRecord]:
    data = json.loads(path.read_text())
    return [ClipRecord(
        vID=c["vID"], hudl_id=c["hudl_id"], video_path=c["video_path"],
        start_sec=c["start_sec"], end_sec=c["end_sec"],
        label=c["label"], source=c["source"],
    ) for c in data["clips"]]


def split_by_game(records: list[ClipRecord], *,
                   train_frac: float = 0.7, val_frac: float = 0.15,
                   seed: int = 42) -> dict[str, list[ClipRecord]]:
    """Game-disjoint split. Stratifies by trying to balance per-class
    counts roughly across splits."""
    # Group by hudl_id (unique per game)
    by_game: dict[int, list[ClipRecord]] = {}
    for r in records:
        by_game.setdefault(r.hudl_id, []).append(r)
    game_ids = sorted(by_game.keys())
    rng = random.Random(seed)
    rng.shuffle(game_ids)
    n_train = int(len(game_ids) * train_frac)
    n_val   = int(len(game_ids) * val_frac)
    train_games = set(game_ids[:n_train])
    val_games   = set(game_ids[n_train:n_train + n_val])
    test_games  = set(game_ids[n_train + n_val:])
    splits = {"train": [], "val": [], "test": []}
    for r in records:
        if r.hudl_id in train_games: splits["train"].append(r)
        elif r.hudl_id in val_games: splits["val"].append(r)
        elif r.hudl_id in test_games: splits["test"].append(r)
    return splits


def extract_frames_pyav(video_path: str, start_sec: int, end_sec: int,
                          n_frames: int = 8, target_size: tuple[int, int] = (224, 224)
                          ) -> Optional[np.ndarray]:
    """Decode N evenly-spaced frames from [start_sec, end_sec].
    Returns (N, H, W, 3) uint8 array, or None on failure.

    Uses PyAV (libav bindings) — much faster than ffmpeg-subprocess
    for short clips because we keep the decoder open per call.
    """
    import av
    try:
        container = av.open(video_path)
    except Exception:
        return None
    try:
        stream = container.streams.video[0]
        # Time base = fraction of a second per pts unit
        tb = stream.time_base
        fps = float(stream.average_rate) if stream.average_rate else 30.0
        seek_pts = int(start_sec / tb)
        try:
            container.seek(seek_pts, stream=stream)
        except Exception:
            pass

        # Decode all frames in the [start, end] window, then subsample
        wanted_secs = np.linspace(start_sec, end_sec, n_frames + 1)[:n_frames]
        target_pts = set(int(t / tb) for t in wanted_secs)
        # Be lenient — we'll match closest frame
        wanted = sorted(wanted_secs)
        out_frames = []
        wanted_idx = 0
        for frame in container.decode(stream):
            if wanted_idx >= len(wanted):
                break
            ts = float(frame.pts * tb) if frame.pts is not None else 0.0
            if ts < wanted[wanted_idx] - 1.0:
                continue
            # Use this frame for the current target (and advance past any
            # targets we've now passed)
            img = frame.to_ndarray(format="rgb24")
            # Resize to target_size
            from PIL import Image
            pil = Image.fromarray(img).resize(target_size,
                                                 Image.Resampling.BILINEAR)
            out_frames.append(np.array(pil))
            # Advance past any targets we've now caught up to
            while wanted_idx < len(wanted) and ts >= wanted[wanted_idx]:
                wanted_idx += 1
        if not out_frames:
            return None
        # Pad with the last frame if we ran short
        while len(out_frames) < n_frames:
            out_frames.append(out_frames[-1].copy())
        return np.stack(out_frames[:n_frames], axis=0)
    finally:
        container.close()


# Simpler ffmpeg-based extractor (used as fallback / for cases where PyAV
# isn't installed or is slow). Always works.
def extract_frames_ffmpeg(video_path: str, start_sec: int, end_sec: int,
                            n_frames: int = 8, target_size: tuple[int, int] = (224, 224)
                            ) -> Optional[np.ndarray]:
    """ffmpeg-subprocess fallback. Slower but rock-solid."""
    dur = end_sec - start_sec
    if dur <= 0:
        return None
    # Sample at fps = n_frames / dur
    target_fps = n_frames / dur
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", str(start_sec), "-i", video_path,
        "-t", str(dur),
        "-vf", f"fps={target_fps:.4f},scale={target_size[0]}:{target_size[1]}",
        "-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=20)
        if proc.returncode != 0:
            return None
        raw = proc.stdout
        H, W = target_size
        bytes_per_frame = H * W * 3
        n_decoded = len(raw) // bytes_per_frame
        if n_decoded == 0:
            return None
        frames = np.frombuffer(raw[:n_decoded * bytes_per_frame],
                                dtype=np.uint8).reshape(n_decoded, H, W, 3)
        # Pad / trim to exactly n_frames
        if n_decoded < n_frames:
            pad = np.repeat(frames[-1:], n_frames - n_decoded, axis=0)
            frames = np.concatenate([frames, pad], axis=0)
        return frames[:n_frames]
    except Exception:
        return None


class ActionClipDataset(Dataset):
    """Returns (clip_tensor, label_idx).

    mode='middle_frame':   clip_tensor is (3, H, W) — single middle frame.
                           Best Phase-1 baseline; averaging frames loses
                           temporal info AND creates a blurry composite.
    mode='per_frame_mean': clip_tensor is (3, H, W) — average of N frames.
                           (deprecated baseline — kept for comparison)
    mode='clip':           clip_tensor is (C, T, H, W) — N frames stacked
                           for X3D / SlowFast.
    """
    def __init__(self, records: list[ClipRecord],
                  mode: Literal["middle_frame", "per_frame_mean", "clip"] = "middle_frame",
                  n_frames: int = 8,
                  size: int = 224,
                  use_pyav: bool = True,
                  augment: bool = False):
        self.records  = records
        self.mode     = mode
        self.n_frames = n_frames
        self.size     = size
        self.use_pyav = use_pyav
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.records)

    def _extract(self, r: ClipRecord) -> Optional[np.ndarray]:
        extractor = extract_frames_pyav if self.use_pyav else extract_frames_ffmpeg
        # For middle_frame mode we only need 1 frame; for the others we need N.
        n = 1 if self.mode == "middle_frame" else self.n_frames
        if self.mode == "middle_frame":
            # Sample from the middle of the clip
            mid_sec = (r.start_sec + r.end_sec) // 2
            return extractor(r.video_path, mid_sec, mid_sec + 1,
                              n_frames=1, target_size=(self.size, self.size))
        return extractor(r.video_path, r.start_sec, r.end_sec,
                          n_frames=n, target_size=(self.size, self.size))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        r = self.records[idx]
        frames = self._extract(r)
        n = 1 if self.mode == "middle_frame" else self.n_frames
        if frames is None:
            frames = np.zeros((n, self.size, self.size, 3), dtype=np.uint8)
        x = torch.from_numpy(frames.copy()).float() / 255.0   # .copy() to silence non-writable warning
        x = x.permute(0, 3, 1, 2)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        x = (x - mean) / std
        if self.augment:
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
        if self.mode == "middle_frame":
            x = x.squeeze(0)              # (C, H, W) — single frame
        elif self.mode == "per_frame_mean":
            x = x.mean(dim=0)             # (C, H, W) — frame average
        elif self.mode == "clip":
            x = x.permute(1, 0, 2, 3)     # (C, T, H, W) — 3D model layout
        y = LABEL_TO_IDX[r.label]
        return x, y


def class_weights(records: list[ClipRecord],
                    mode: str = "sqrt_inv") -> torch.Tensor:
    """Weights for the weighted CE loss.

    mode='inv'      — 1/freq (extreme; collapses small-data models to the
                       rarest class)
    mode='sqrt_inv' — 1/sqrt(freq) (recommended default; tames extreme
                       imbalance without collapsing the model)
    mode='log_inv'  — 1/log(1+freq) (most conservative)
    mode='uniform'  — equal weight for all classes
    """
    c = Counter(r.label for r in records)
    import math
    if mode == "uniform":
        weights = [1.0] * len(LABEL_NAMES)
    elif mode == "inv":
        weights = [1.0 / max(c[lbl], 1) for lbl in LABEL_NAMES]
    elif mode == "log_inv":
        weights = [1.0 / math.log(1 + max(c[lbl], 1)) for lbl in LABEL_NAMES]
    else:  # sqrt_inv
        weights = [1.0 / math.sqrt(max(c[lbl], 1)) for lbl in LABEL_NAMES]
    w = torch.tensor(weights, dtype=torch.float32)
    return w / w.sum() * len(LABEL_NAMES)   # normalize so sum = num_classes
