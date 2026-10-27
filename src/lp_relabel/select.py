"""Choose frames to label from per-frame scores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_windows(path: str | Path, n_frames: int, pre: int = 0, post: int = 0) -> np.ndarray:
    """Boolean mask of frames inside the windows listed in a CSV.

    The CSV has either ``start_frame`` and ``end_frame`` columns (end exclusive),
    or a ``frame`` column of event frames, each opening ``[frame - pre, frame + post]``.
    """
    df = pd.read_csv(path)
    mask = np.zeros(n_frames, dtype=bool)
    if {"start_frame", "end_frame"} <= set(df.columns):
        pairs = zip(df["start_frame"], df["end_frame"], strict=True)
    elif "frame" in df.columns:
        if pre == 0 and post == 0:
            raise ValueError(f"{path}: a 'frame' column needs --pre and/or --post")
        pairs = ((f - pre, f + post + 1) for f in df["frame"])
    else:
        raise ValueError(f"{path}: needs start_frame/end_frame columns or a frame column")
    for start, end in pairs:
        if pd.isna(start) or pd.isna(end):
            continue
        mask[max(int(start), 0) : min(int(end), n_frames)] = True
    return mask


def pick_frames(
    scores: pd.DataFrame,
    n: int,
    min_gap: int,
    margin: int,
    random_fraction: float = 0.2,
    allowed: np.ndarray | None = None,
    exclude: set[int] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Pick up to ``n`` frames: the highest scores, plus some at random.

    ``n * random_fraction`` frames (rounded) are random among the allowed frames,
    so each round also covers ordinary frames and not only the hardest ones.
    Picks are at least ``min_gap`` frames apart and at least ``margin`` frames from
    either end of the video (so context frames exist). Frames in ``exclude`` and
    within ``min_gap`` of one are skipped.

    Returns the chosen rows of ``scores`` with a ``reason`` column
    (``"score"`` or ``"random"``), sorted by frame.
    """
    n_frames = len(scores)
    ok = np.ones(n_frames, dtype=bool)
    ok[:margin] = False
    if margin:
        ok[n_frames - margin :] = False
    if allowed is not None:
        ok &= allowed
    taken: list[int] = []
    blocked = np.zeros(n_frames, dtype=bool)

    def block(frame: int) -> None:
        blocked[max(frame - min_gap + 1, 0) : frame + min_gap] = True

    for frame in exclude or ():
        if 0 <= frame < n_frames:
            block(frame)

    n_random = int(round(n * random_fraction))
    n_score = n - n_random
    reasons: dict[int, str] = {}

    order = np.argsort(-scores["score"].to_numpy(), kind="stable")
    for frame in order:
        if len(taken) >= n_score:
            break
        if ok[frame] and not blocked[frame]:
            taken.append(int(frame))
            reasons[int(frame)] = "score"
            block(int(frame))

    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(ok & ~blocked)
    while len(taken) < n and pool.size:
        frame = int(rng.choice(pool))
        taken.append(frame)
        reasons[frame] = "random"
        block(frame)
        pool = pool[~blocked[pool]]

    out = scores.iloc[sorted(taken)].copy()
    out["reason"] = [reasons[f] for f in out["frame"]]
    return out.reset_index(drop=True)
