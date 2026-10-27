"""Frame-exact JPEG extraction with ffmpeg, from a local MP4 or an S3/HTTP URL.

Each frame is addressed by its presentation time read from the MP4 sample tables
(``aind_video_utils``), never by an assumed frame rate, so the JPEG for frame
``k`` is row ``k`` of the prediction CSV even when the file has an irregular
timeline. For URLs only the index (``moov``) and the bytes near each seek are
downloaded, never the whole video.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path

from aind_video_utils import read_mp4_frame_index
from aind_video_utils.utils import http_input_flags

FRAME_NAME = "img{:08d}.jpg"  # Lightning Pose App: FMT_FRAME_INDEX_DIGITS = 8


def resolve_video(source: str) -> str:
    """Local path or URL as is; ``s3://bucket/key`` becomes a presigned HTTPS URL."""
    if not source.startswith("s3://"):
        return source
    try:
        import boto3
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError("s3:// videos need boto3: pip install 'lp-relabel[s3]'") from e
    bucket, _, key = source[len("s3://") :].partition("/")
    return boto3.client("s3").generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=6 * 3600
    )


def context_runs(targets: Iterable[int], context: int, n_frames: int) -> list[tuple[int, int]]:
    """Contiguous ``(start, stop)`` runs covering every target ± ``context`` (stop exclusive)."""
    needed = sorted(
        {f for t in targets for f in range(t - context, t + context + 1) if 0 <= f < n_frames}
    )
    runs: list[tuple[int, int]] = []
    for f in needed:
        if runs and f == runs[-1][1]:
            runs[-1] = (runs[-1][0], f + 1)
        else:
            runs.append((f, f + 1))
    return runs


def _run_command(video: str, seek: float, start: int, count: int, pattern: Path, quality: int):
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-accurate_seek", "-ss", f"{seek:.9f}",
        *http_input_flags(video),
        "-i", video,
        "-map", "0:v:0",
        "-frames:v", str(count),
        "-fps_mode", "passthrough",
        "-q:v", str(quality),
        "-start_number", str(start),
        str(pattern),
    ]  # fmt: skip


def extract_with_context(
    video: str,
    targets: Iterable[int],
    out_dir: str | Path,
    context: int = 2,
    quality: int = 2,
    index=None,
) -> dict[int, Path]:
    """Write each target frame and its ``context`` neighbours on both sides as JPEGs.

    Files are ``out_dir/img########.jpg`` named by source frame index, the layout
    the Lightning Pose App and LP's context models expect. Existing files are kept.
    One ffmpeg call per contiguous run of needed frames.

    Returns target frame -> path of its JPEG.
    """
    targets = sorted({int(t) for t in targets})
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if index is None:
        index = read_mp4_frame_index(video)
    if not index.is_frame_addressing_safe():
        raise ValueError(f"{video}: edit list is not safe for addressing frames by index")
    n_frames = index.n_samples
    bad = [t for t in targets if not 0 <= t < n_frames]
    if bad:
        raise ValueError(f"frames {bad} outside the video ({n_frames} frames)")

    for start, stop in context_runs(targets, context, n_frames):
        names = [FRAME_NAME.format(f) for f in range(start, stop)]
        if all((out_dir / name).exists() for name in names):
            continue
        with tempfile.TemporaryDirectory(dir=out_dir, prefix=".tmp-") as tmp:
            tmp = Path(tmp)
            cmd = _run_command(
                video, index.presentation_seconds(start), start, stop - start,
                tmp / "img%08d.jpg", quality,
            )  # fmt: skip
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed on {video} frames {start}-{stop - 1}:\n{result.stderr}")
            got = sorted(p.name for p in tmp.glob("img*.jpg"))
            if got != names:
                raise RuntimeError(
                    f"{video}: asked for frames {start}-{stop - 1}, ffmpeg wrote {got[:3]}..."
                )
            for name in names:
                if not (out_dir / name).exists():
                    shutil.move(str(tmp / name), out_dir / name)
    return {t: out_dir / FRAME_NAME.format(t) for t in targets}
