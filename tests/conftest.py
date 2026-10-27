"""Synthetic videos whose pixels encode their own frame index, and matching predictions."""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BLOCK = 16
DIGITS = 4  # base-8 digits -> up to 4096 frames


def encode_frame(i: int) -> np.ndarray:
    img = np.zeros((BLOCK, BLOCK * DIGITS), dtype=np.uint8)
    for d in range(DIGITS):
        img[:, d * BLOCK : (d + 1) * BLOCK] = ((i >> (3 * d)) & 7) * 32 + 16
    return img


def decode_image(path: Path) -> int:
    raw = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "gray", "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    img = np.frombuffer(raw, dtype=np.uint8).reshape(BLOCK, BLOCK * DIGITS)
    value = 0
    for d in range(DIGITS):
        level = img[4:-4, d * BLOCK + 4 : (d + 1) * BLOCK - 4].mean()
        value |= int(np.clip(round((level - 16) / 32), 0, 7)) << (3 * d)
    return value


def make_video(path: Path, n_frames: int, fps: int = 500) -> Path:
    """H.264 with a 50-frame GOP and B-frames, so seeks must decode forward."""
    frames = np.stack([encode_frame(i) for i in range(n_frames)])
    cmd = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{BLOCK * DIGITS}x{BLOCK}", "-r", str(fps), "-i", "-",
        "-c:v", "libx264", "-qp", "0", "-g", "50", "-bf", "2", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", str(path),
    ]  # fmt: skip
    subprocess.run(cmd, input=frames.tobytes(), check=True)
    return path


def make_pose_csv(path: Path, n_frames: int, keypoints=("jaw", "tongue_tip"), seed=0) -> Path:
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product([["heatmap_tracker"], list(keypoints), ["x", "y", "likelihood"]])
    data = np.zeros((n_frames, len(cols)))
    for k in range(len(keypoints)):
        data[:, 3 * k] = 30 + rng.normal(0, 0.2, n_frames)
        data[:, 3 * k + 1] = 8 + rng.normal(0, 0.2, n_frames)
        data[:, 3 * k + 2] = 0.98
    pd.DataFrame(data, columns=cols).to_csv(path)
    return path


@pytest.fixture(scope="session")
def video(tmp_path_factory) -> Path:
    return make_video(tmp_path_factory.mktemp("vid") / "session_a.mp4", 600)
