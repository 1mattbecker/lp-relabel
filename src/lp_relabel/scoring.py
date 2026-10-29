"""Per-frame difficulty scores from a model's own predictions.

Two signals, both available without labels:

* **Likelihood.** ``"uncertain"`` scores likelihoods near 0.5 highest. Use it when
  a keypoint is often truly absent (a tongue inside a closed mouth): the model is
  then right to say ~0, and ``"low"`` would pick those frames over and over.
  ``"low"`` scores 1 - likelihood.
* **Jump.** Distance of a keypoint from the midpoint of its two neighbours, in
  units of that keypoint's median such distance. A one-frame spike is almost
  always an error at high frame rates. Only computed where all three frames have
  likelihood >= ``jump_min_likelihood``, so appearing / disappearing points are
  not jumps, and only where the three rows are consecutive video frames, so the
  gap between two windows of a sparse file is not a jump.

Each signal is the max over keypoints, then turned into a percentile rank within
the video, and the score is their weighted mean.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lp_relabel.poses import PoseTrack

LIKELIHOOD_MODES = ("uncertain", "low")


def likelihood_score(lik: np.ndarray, mode: str = "uncertain") -> np.ndarray:
    """Per-frame likelihood score: max over keypoints."""
    lik = np.clip(np.nan_to_num(lik, nan=0.0), 0.0, 1.0)
    if mode == "uncertain":
        per_kp = 1.0 - np.abs(2.0 * lik - 1.0)
    elif mode == "low":
        per_kp = 1.0 - lik
    else:
        raise ValueError(f"likelihood mode must be one of {LIKELIHOOD_MODES}")
    return per_kp.max(axis=1)


def jump_score(
    xy: np.ndarray, lik: np.ndarray, min_likelihood: float = 0.5, frames: np.ndarray | None = None
) -> np.ndarray:
    """Per-frame spike score: max over keypoints of the normalized midpoint distance.

    ``frames`` gives each row's video frame (default consecutive rows).
    """
    n, k, _ = xy.shape
    out = np.zeros((n, k))
    if n < 3:
        return out.max(axis=1) if k else np.zeros(n)
    mid = 0.5 * (xy[:-2] + xy[2:])
    dist = np.linalg.norm(xy[1:-1] - mid, axis=-1)  # (n-2, k)
    ok = (lik[:-2] >= min_likelihood) & (lik[1:-1] >= min_likelihood) & (lik[2:] >= min_likelihood)
    if frames is not None:
        ok &= (frames[2:] - frames[:-2] == 2)[:, None]  # increasing, so both steps are 1
    dist = np.where(ok & np.isfinite(dist), dist, np.nan)
    for j in range(k):
        col = dist[:, j]
        finite = col[np.isfinite(col)]
        if finite.size == 0:
            continue
        scale = max(float(np.median(finite)), 1e-3)
        out[1:-1, j] = np.nan_to_num(col / scale, nan=0.0)
    return out.max(axis=1)


def frame_scores(
    track: PoseTrack,
    likelihood_mode: str = "uncertain",
    jump_min_likelihood: float = 0.5,
    likelihood_weight: float = 1.0,
    jump_weight: float = 1.0,
) -> pd.DataFrame:
    """Score every predicted frame. Columns: frame (video frame), lik_score, jump_score, score (0..1)."""
    lik_s = likelihood_score(track.likelihood, likelihood_mode)
    jump_s = jump_score(track.xy, track.likelihood, jump_min_likelihood, track.frames)
    total = likelihood_weight + jump_weight
    if total <= 0:
        raise ValueError("weights must sum to more than 0")
    rank = lambda a: pd.Series(a).rank(pct=True).to_numpy()  # noqa: E731
    score = (likelihood_weight * rank(lik_s) + jump_weight * rank(jump_s)) / total
    return pd.DataFrame(
        {"frame": track.frames, "lik_score": lik_s, "jump_score": jump_s, "score": score}
    )
