"""Read Lightning Pose (DLC-style) prediction CSVs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PoseTrack:
    """Predictions for frames of one video.

    Row ``i`` of every array is video frame ``frames[i]`` (presentation order).
    A whole-video file has ``frames == 0..n-1``; a sparse one (windows of the
    video, as lp-capsule writes) has increasing frame numbers with gaps.
    """

    keypoints: list[str]
    xy: np.ndarray  # (n_rows, n_keypoints, 2), float
    likelihood: np.ndarray  # (n_rows, n_keypoints), float in [0, 1]
    frames: np.ndarray | None = None  # (n_rows,) source frame of each row; default 0..n-1

    def __post_init__(self) -> None:
        if self.frames is None:
            self.frames = np.arange(self.xy.shape[0])
        self.frames = np.asarray(self.frames, dtype=np.int64)

    @property
    def n_frames(self) -> int:
        """Number of rows (predicted frames)."""
        return int(self.xy.shape[0])

    @property
    def is_dense(self) -> bool:
        """True when the rows are frames 0..n-1 (a whole-video file)."""
        return bool(np.array_equal(self.frames, np.arange(self.n_frames)))

    def row(self, frame: int) -> int:
        """Row of a predicted frame."""
        i = int(np.searchsorted(self.frames, frame))
        if i >= self.n_frames or self.frames[i] != frame:
            raise KeyError(f"frame {frame} has no prediction")
        return i

    def subset(self, keypoints: list[str]) -> PoseTrack:
        """The same track with only ``keypoints``, in that order."""
        missing = [k for k in keypoints if k not in self.keypoints]
        if missing:
            raise ValueError(f"keypoints not in predictions: {missing}; have {self.keypoints}")
        cols = [self.keypoints.index(k) for k in keypoints]
        return PoseTrack(list(keypoints), self.xy[:, cols], self.likelihood[:, cols], self.frames)


def fetch(source: str | Path, cache_dir: str | Path | None = None) -> Path:
    """Local path as is; an http(s) URL is downloaded once into ``cache_dir``."""
    source = str(source)
    if not source.startswith(("http://", "https://")):
        return Path(source)
    import hashlib
    import shutil
    import urllib.request

    cache = Path(cache_dir or Path.home() / ".cache" / "lp-relabel")
    cache.mkdir(parents=True, exist_ok=True)
    name = hashlib.sha1(source.encode()).hexdigest()[:12] + "_" + source.rsplit("/", 1)[-1]
    dest = cache / name
    if not dest.exists():
        logger.info("downloading %s", source)
        tmp = dest.with_suffix(".part")
        with urllib.request.urlopen(source) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 22)
        tmp.replace(dest)
    return dest


def read_pose_csv(path: str | Path, cache_dir: str | Path | None = None) -> PoseTrack:
    """Read a Lightning Pose predictions CSV (header rows scorer / bodyparts / coords).

    ``path`` may be an http(s) URL; it is then cached locally (see :func:`fetch`).
    Uses the ``x``, ``y`` and ``likelihood`` columns of each bodypart and ignores
    others (for example EKS variances). A file without ``likelihood`` columns gets
    likelihood 1 everywhere. The index must be the video frame number of each
    row: 0..n-1 for a whole video, or increasing frame numbers with gaps for
    windows of it (lp-capsule's predictions).
    """
    path = fetch(path, cache_dir)
    header = pd.read_csv(path, header=[0, 1, 2], index_col=0, nrows=0)
    coords = list(header.columns.droplevel(0))
    bodyparts = list(dict.fromkeys(b for b, _ in coords))
    keypoints = [b for b in bodyparts if (b, "x") in coords and (b, "y") in coords]
    if not keypoints:
        raise ValueError(f"{path}: no bodypart has both x and y columns")
    has_lik = all((k, "likelihood") in coords for k in keypoints)
    wanted = [(k, c) for k in keypoints for c in (("x", "y", "likelihood") if has_lik else ("x", "y"))]
    positions = [coords.index(w) + 1 for w in wanted]  # +1: column 0 is the index
    data = pd.read_csv(
        path, skiprows=3, header=None, usecols=[0, *positions], index_col=0,
        dtype={p: "float32" for p in positions},
    )  # fmt: skip
    index = pd.to_numeric(pd.Series(data.index), errors="coerce").to_numpy()
    if (
        np.isnan(index).any()
        or (index != np.round(index)).any()
        or (index < 0).any()
        or (np.diff(index) <= 0).any()
    ):
        raise ValueError(
            f"{path}: index is not increasing video frame numbers; "
            "rows are matched to video frames by this index"
        )
    data = data[positions].to_numpy()
    width = 3 if has_lik else 2
    arr = data.reshape(len(data), len(keypoints), width)
    xy = arr[:, :, :2].astype(float)
    if has_lik:
        lik = arr[:, :, 2].astype(float)
    else:
        logger.warning("%s: no likelihood columns; using 1.0", path)
        lik = np.ones(xy.shape[:2])

    return PoseTrack(keypoints, xy, lik, index.astype(np.int64))
