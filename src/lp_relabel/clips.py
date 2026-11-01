"""Clean clips for Lightning Pose: frame-addressing-safe MP4s and context frames.

LP's context model reads a labeled frame's neighbours as the files numbered
``k-2..k+2`` beside it; its unlabeled losses and video inference read MP4s
with DALI. Both go wrong on clips that were cut by stream copy: such an MP4
indexes more samples than it displays (an edit list hides the rest) and may
have timestamp gaps, so frame ``k`` of the file is not ``img<k>.png``.

- :func:`clean_project` re-encodes each clip of a labeled project with
  ``transcode_video(normalize_cfr=True)`` (lossless, temporary), checks every
  labeled PNG against ``extract_frame_by_index`` of that transcode, writes the
  neighbours as PNGs, and writes a crf-18 clean clip for the unlabeled losses.
- :func:`cut_session_clips` cuts unlabeled clips from long session videos,
  away from labeled frames, and checks that each one is clean.

A clip is clean when its frame index is frame-addressing-safe and its
timeline uniform (``is_clean``).
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from aind_video_utils import (
    EncodingProfile,
    extract_frame_by_index,
    read_mp4_frame_index,
    transcode_video,
)
from aind_video_utils import __version__ as video_utils_version
from aind_video_utils.utils import http_input_flags
from PIL import Image

from lp_relabel import __version__, project
from lp_relabel.frames import resolve_video

logger = logging.getLogger(__name__)

# LP 2.4.3 reads context frames k-2..k+2 by file name (utils/io.py
# get_context_img_paths) and DALI reads consecutive video frames, so a stride
# other than 1 would need LP changes.
OFFSETS = (-2, -1, 1, 2)
# Mean |difference| (0-255) allowed between a labeled PNG and the same frame
# of the lossless transcode; measured <= 0.004 on Pylon2Test2, a wrong frame is >= 0.9.
MAX_DIFF = 0.05
CLIP_NAME = "{prefix}_f{start:07d}.mp4"


def profile(*codec_params: str) -> EncodingProfile:
    """H.264 with no colour conversion, so pixels stay those of the source decode."""
    return EncodingProfile(
        video_filters="format=yuv420p",
        codec="libx264",
        pixel_format="yuv420p",
        container="mp4",
        codec_params=codec_params,
        output_flags=("-movflags", "+faststart"),
    )


LOSSLESS = profile("-preset", "fast", "-qp", "0")
CLEAN = profile("-preset", "slow", "-crf", "18")


def is_remote(source: str | Path) -> bool:
    return str(source).startswith(("http://", "https://", "s3://"))


def fetch(source: str | Path, path: Path, tries: int = 6) -> Path:
    """Copy a local file, or download a URL retrying network errors, to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not is_remote(source):
        shutil.copyfile(source, path)
        return path
    url = resolve_video(str(source))
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as f:
                shutil.copyfileobj(response, f)
            return path
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == tries - 1:
                raise
            time.sleep(10 * 2**attempt)
    raise AssertionError("unreachable")


def join(root: str | Path, rel: str) -> str:
    """``rel`` under a local folder or a URL prefix."""
    return f"{str(root).rstrip('/')}/{rel}" if is_remote(root) else str(Path(root) / rel)


def is_clean(index) -> bool:
    """Whether DALI and frame-index addressing see the clip's frames as they are displayed."""
    return index.is_frame_addressing_safe() and index.timeline_notches().is_uniform


def transcode(src: Path, dst: Path, codec: EncodingProfile) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        # The notched source timeline is what normalize_cfr repairs.
        warnings.simplefilter("ignore")
        transcode_video(src, dst, profile=codec, normalize_cfr=True, auto_fix_colorspace=False)
    return dst


def clean_clip(
    clip: str, images: list[str], source: str, videos: str, out: Path, done_dir: Path
) -> pd.DataFrame:
    """Clean one clip and write its labeled frames and their neighbours; see the module docstring.

    Resumable: the clip's check record ``done_dir/<clip>.csv`` is written last
    and, when present, is returned without redoing the clip.
    """
    done = done_dir / f"{clip}.csv"
    if done.is_file():
        return pd.read_csv(done)
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        src = fetch(join(videos, f"{clip}.mp4"), Path(tmp) / "src.mp4")
        lossless = transcode(src, Path(tmp) / "lossless.mp4", LOSSLESS)
        index = read_mp4_frame_index(lossless)
        if not is_clean(index):
            raise RuntimeError(f"{clip}: transcode is not a clean clip")
        source_samples = read_mp4_frame_index(src).n_samples
        folder = out / project.LABELED_DATA / clip
        for image in images:
            name = image.rsplit("/", 1)[1]
            png = fetch(join(source, image), folder / name)
            digits = re.fullmatch(r"img(\d+)\.png", name)
            if not digits:
                raise ValueError(f"{image}: expected labeled-data/<clip>/img<frame>.png")
            k, width = int(digits[1]), len(digits[1])
            labeled = np.asarray(Image.open(png).convert("RGB"), dtype=np.int16)
            decoded = extract_frame_by_index(lossless, k, index=index)
            if decoded.shape != labeled.shape:
                raise RuntimeError(f"{image}: {labeled.shape} image, clip frames are {decoded.shape}")
            diff = float(np.abs(decoded.astype(np.int16) - labeled).mean())
            if diff > MAX_DIFF:
                raise RuntimeError(f"{image}: differs from frame {k} of its clip ({diff:.3f})")
            written = 0
            for d in OFFSETS:
                j = k + d
                path = folder / f"img{str(j).zfill(width)}.png"
                # Past either end of the clip LP falls back to the labeled frame.
                if 0 <= j < index.n_samples and not path.exists():
                    Image.fromarray(extract_frame_by_index(lossless, j, index=index)).save(path)
                    written += 1
            rows.append(
                {
                    "image": image,
                    "frame": k,
                    "clip_frames": index.n_samples,
                    "source_samples": source_samples,
                    "labeled_vs_clip_mean_abs_diff": diff,
                    "neighbours_written": written,
                }
            )
        clean = transcode(src, out / "videos" / f"{clip}.mp4", CLEAN)
        if not is_clean(read_mp4_frame_index(clean)):
            raise RuntimeError(f"{clean}: not a clean clip")
    # Last: its existence marks the clip done.
    pd.DataFrame(rows).to_csv(done, index=False)
    logger.info("  %s: %d labeled, %d neighbours", clip, len(images), sum(r["neighbours_written"] for r in rows))
    return pd.DataFrame(rows)


def labeled_clips(labels_csv: str | Path, match: str | None = None) -> dict[str, list[str]]:
    """Labeled images of a label CSV by clip folder, optionally only folders matching a regex."""
    index = pd.read_csv(labels_csv, header=[0, 1, 2], index_col=0).index.astype(str)
    by_clip: dict[str, list[str]] = {}
    for image in index:
        parts = image.split("/")
        if len(parts) != 3 or parts[0] != project.LABELED_DATA:
            raise ValueError(f"{labels_csv}: {image} is not labeled-data/<clip>/<image>")
        if match is None or re.search(match, parts[1]):
            by_clip.setdefault(parts[1], []).append(image)
    return dict(sorted(by_clip.items()))


def clean_project(
    labels_csv: str | Path,
    source: str,
    out: Path,
    videos: str | None = None,
    match: str | None = None,
    n_clips: int | None = None,
    workers: int = 6,
) -> pd.DataFrame:
    """Clean every clip with labeled frames in ``labels_csv``; write ``out/checks.csv``.

    ``source`` (folder or http/S3 prefix) holds ``labeled-data/<clip>/img*.png``
    and, unless ``videos`` is given, ``videos/<clip>.mp4``. Per-clip check
    records go to ``<out>_checks/`` so ``out`` itself is the asset to upload.
    """
    by_clip = labeled_clips(labels_csv, match)
    if n_clips:
        by_clip = dict(list(by_clip.items())[:n_clips])
    if not by_clip:
        raise ValueError(f"{labels_csv}: no labeled frames" + (f" in folders matching {match!r}" if match else ""))
    videos = videos or join(source, "videos")
    logger.info("%d labeled frames in %d clips -> %s", sum(map(len, by_clip.values())), len(by_clip), out)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    done_dir = out.with_name(f"{out.name}_checks")
    done_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda c: clean_clip(c, by_clip[c], source, videos, out, done_dir), by_clip))
    checks = pd.concat(results, ignore_index=True)
    checks.to_csv(out / "checks.csv", index=False)
    logger.info(
        "done: %d frames, max labeled-vs-clip diff %.4f, %d neighbours, %d clean clips",
        len(checks), checks.labeled_vs_clip_mean_abs_diff.max(), checks.neighbours_written.sum(), len(by_clip),
    )  # fmt: skip
    return checks


def clip_starts(
    n_samples: int,
    n: int,
    clip_frames: int,
    span: tuple[float, float],
    min_gap: int,
    taken: list[int],
    rng: np.random.Generator,
    tries: int = 100_000,
) -> list[int]:
    """``n`` random clip starts within ``span`` of the video, each more than ``min_gap``
    frames from every taken frame and every other start."""
    lo, hi = (int(f * n_samples) for f in span)
    if hi - clip_frames <= lo:
        raise ValueError(f"span {span} of {n_samples} frames is shorter than a {clip_frames}-frame clip")
    starts: list[int] = []
    for _ in range(tries):
        if len(starts) == n:
            return sorted(starts)
        s = int(rng.integers(lo, hi - clip_frames))
        if all(abs(s - t) > min_gap for t in taken + starts):
            starts.append(s)
    raise ValueError(f"found {len(starts)} of {n} clip starts {min_gap} frames apart in {tries} tries")


def _cut_command(video: str, seek: float, n_frames: int, out_path: Path) -> list[str]:
    # As aind-dynamic-foraging-behavior-video-analysis's cut_clip: seek to the
    # first frame's own presentation time, take exactly n frames, restamp evenly.
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-accurate_seek", "-ss", f"{seek:.9f}",
        *http_input_flags(video),
        "-i", video,
        "-map", "0:v:0",
        "-frames:v", str(n_frames),
        "-vf", "setpts=N/FRAME_RATE/TB",
        "-fps_mode", "passthrough",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(out_path),
    ]  # fmt: skip


def cut_clip(video: str, start: int, n_frames: int, out_path: Path, index=None) -> dict:
    """Cut source frames ``[start, start + n_frames)`` into a clean MP4.

    Re-encodes the cut with ``normalize_cfr`` if it is not clean, and fails if
    the result is not either. Returns the clip's checks.
    """
    if index is None:
        index = read_mp4_frame_index(video)
    if not index.is_frame_addressing_safe():
        raise ValueError(f"{video}: edit list is not safe for addressing frames by index")
    if start < 0 or start + n_frames > index.n_samples:
        raise ValueError(f"frames [{start}, {start + n_frames}) outside the video ({index.n_samples} frames)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent, prefix=".tmp-") as tmp:
        cut = Path(tmp) / "cut.mp4"
        cmd = _cut_command(video, index.presentation_seconds(start), n_frames, cut)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {video} frames {start}+{n_frames}:\n{result.stderr}")
        clip_index = read_mp4_frame_index(cut)
        renormalized = not is_clean(clip_index)
        if renormalized:
            cut = transcode(cut, Path(tmp) / "normalized.mp4", CLEAN)
            clip_index = read_mp4_frame_index(cut)
        if not is_clean(clip_index):
            raise RuntimeError(f"{video} frames {start}+{n_frames}: cut is not a clean clip")
        if clip_index.n_samples != n_frames:
            raise RuntimeError(f"{video} frames {start}+{n_frames}: cut has {clip_index.n_samples} frames")
        cut.replace(out_path)
    return {"clip_frames": clip_index.n_samples, "renormalized": renormalized}


def cut_session_clips(
    video: str, starts: list[int], n_frames: int, out_dir: Path, prefix: str, index=None
) -> pd.DataFrame:
    """Cut one clean clip per start as ``<prefix>_f<start:07d>.mp4``, each with a JSON sidecar.

    Resumable: a clip whose sidecar names the same video and length is kept.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in starts:
        clip = out_dir / CLIP_NAME.format(prefix=prefix, start=start)
        sidecar = clip.with_suffix(".json")
        info = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
        if clip.is_file() and info.get("source_video") == video and info.get("n_frames") == n_frames:
            rows.append({"start_frame": start, "n_frames": n_frames, "clip_path": str(clip), "status": "exists",
                         **{k: info.get(k) for k in ("clip_frames", "renormalized")}})  # fmt: skip
            continue
        if index is None:
            index = read_mp4_frame_index(resolve_video(video))
        checks = cut_clip(resolve_video(video), start, n_frames, clip, index=index)
        info = {
            "source_video": video, "start_frame": start, "n_frames": n_frames, **checks,
            "versions": {"lp_relabel": __version__, "aind_video_utils": video_utils_version},
        }  # fmt: skip
        # Last: its existence marks the clip done.
        sidecar.write_text(json.dumps(info, indent=2) + "\n")
        rows.append({"start_frame": start, "n_frames": n_frames, "clip_path": str(clip), "status": "cut", **checks})
    return pd.DataFrame(rows)
