"""Choose frames to label from per-frame scores.

Frames are video frame numbers throughout: the ``frame`` column of the scores,
windows, ``min_gap``, and excluded frames. Scores may cover the whole video or
only windows of it (sparse predictions).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_windows(
    path: str | Path, frames: np.ndarray | int, pre: int = 0, post: int = 0
) -> np.ndarray:
    """Boolean mask over ``frames`` of those inside the windows listed in a CSV.

    ``frames`` is the video frame of each scored row (an int ``n`` means 0..n-1).
    The CSV has either ``start_frame`` and ``end_frame`` columns (end exclusive),
    or a ``frame`` column of event frames, each opening ``[frame - pre, frame + post]``.
    """
    frames = np.arange(frames) if isinstance(frames, int) else np.asarray(frames)
    df = pd.read_csv(path)
    mask = np.zeros(len(frames), dtype=bool)
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
        mask |= (frames >= int(start)) & (frames < int(end))
    return mask


def has_context(frames: np.ndarray, margin: int) -> np.ndarray:
    """True for rows with ``margin`` predicted, consecutive frames on each side."""
    n = len(frames)
    ok = np.zeros(n, dtype=bool)
    if n > 2 * margin:
        inner = np.arange(margin, n - margin)
        ok[inner] = frames[inner + margin] - frames[inner - margin] == 2 * margin
    return ok


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
    Picks are at least ``min_gap`` video frames apart, and each has ``margin``
    predicted consecutive frames on either side (so its context frames exist and
    are inside the video and the window). Frames in ``exclude`` and within
    ``min_gap`` of one are skipped.

    Returns the chosen rows of ``scores`` with a ``reason`` column
    (``"score"`` or ``"random"``), sorted by frame.
    """
    frames = scores["frame"].to_numpy(dtype=np.int64)
    ok = has_context(frames, margin)
    if allowed is not None:
        ok &= allowed
    taken: list[int] = []  # rows
    blocked = np.zeros(len(frames), dtype=bool)

    def block(frame: int) -> None:
        lo, hi = np.searchsorted(frames, [frame - min_gap + 1, frame + min_gap])
        blocked[lo:hi] = True

    for frame in exclude or ():
        block(int(frame))

    n_random = int(round(n * random_fraction))
    n_score = n - n_random
    reasons: dict[int, str] = {}

    order = np.argsort(-scores["score"].to_numpy(), kind="stable")
    for row in order:
        if len(taken) >= n_score:
            break
        if ok[row] and not blocked[row]:
            taken.append(int(row))
            reasons[int(row)] = "score"
            block(int(frames[row]))

    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(ok & ~blocked)
    while len(taken) < n and pool.size:
        row = int(rng.choice(pool))
        taken.append(row)
        reasons[row] = "random"
        block(int(frames[row]))
        pool = pool[~blocked[pool]]

    rows = sorted(taken)
    out = scores.iloc[rows].copy()
    out["reason"] = [reasons[r] for r in rows]
    return out.reset_index(drop=True)
