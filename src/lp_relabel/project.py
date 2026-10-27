"""Read and write the Lightning Pose App's on-disk project files.

Formats, as of lightning-pose-app 2.4.2.3:

* ``<data_dir>/project.yaml``: ``view_names`` (``[]`` for single view) and
  ``keypoint_names``.
* ``<data_dir>/<name>.csv``: label CSV, header rows scorer / bodyparts / coords
  (``x``, ``y``), index ``labeled-data/<folder>/img########.jpg``.
* ``<data_dir>/<name>.unlabeled.jsonl``: the labeling queue, one JSON object per
  line, ``{"frame_path": ..., "predictions": {"model_name", "date_time",
  "predictions": [{"keypoint_name", "x", "y"}]}}``. The Labeler shows the
  predictions as starting keypoints; saving moves the frame into the CSV.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import yaml

LABELED_DATA = "labeled-data"
_FRAME_RE = re.compile(r"(?:^|/)labeled-data/([^/]+)/img(\d+)\.\w+$")


def safe_folder(name: str) -> str:
    """A folder name for labeled-data: letters, digits, ``-``, ``_`` and ``.`` only."""
    folder = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("._")
    if not folder:
        raise ValueError(f"cannot make a folder name from {name!r}")
    return folder


def frame_path(folder: str, frame: int) -> str:
    """Relative path of a frame, as stored in label CSVs and the queue."""
    return f"{LABELED_DATA}/{folder}/img{int(frame):08d}.jpg"


def parse_frame_path(path: str) -> tuple[str, int] | None:
    """``(folder, frame)`` from a labeled-data path, or None if it is not one."""
    m = _FRAME_RE.search(str(path).replace("\\", "/"))
    return (m.group(1), int(m.group(2))) if m else None


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{time.time_ns()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def ensure_project_yaml(data_dir: Path, keypoints: list[str]) -> None:
    """Write a single-view project.yaml, or check an existing one lists these keypoints."""
    path = data_dir / "project.yaml"
    if path.exists():
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        listed = config.get("keypoint_names") or []
        if listed and list(listed) != list(keypoints):
            raise ValueError(f"{path} lists keypoints {listed}, predictions have {keypoints}")
        if config.get("view_names"):
            raise ValueError(f"{path} is a multiview project; this tool writes single-view queues")
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, yaml.safe_dump({"view_names": [], "keypoint_names": list(keypoints)}, sort_keys=False))


def ensure_label_csv(csv_path: Path, keypoints: list[str]) -> None:
    """Create an empty label CSV with the app's header, or check an existing one."""
    if csv_path.exists():
        existing = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0, nrows=0)
        have = list(dict.fromkeys(existing.columns.get_level_values(1)))
        if have != list(keypoints):
            raise ValueError(f"{csv_path} has keypoints {have}, predictions have {keypoints}")
        return
    columns = pd.MultiIndex.from_product(
        [["scorer"], keypoints, ["x", "y"]], names=["scorer", "bodyparts", "coords"]
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([], index=[], columns=columns).to_csv(csv_path)


def sidecar_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".unlabeled.jsonl")


def known_frames(data_dir: Path, csv_paths: list[Path]) -> set[tuple[str, int]]:
    """``(folder, frame)`` of every row in these label CSVs and their queues."""
    found: set[tuple[str, int]] = set()
    for csv_path in csv_paths:
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
            found.update(p for p in map(parse_frame_path, df.index.astype(str)) if p)
        side = sidecar_path(csv_path)
        if side.exists():
            for line in side.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    p = parse_frame_path(json.loads(line).get("frame_path", ""))
                    if p:
                        found.add(p)
    return found


def queue_entry(path: str, model_name: str, keypoints: list[str], xy, lik, min_likelihood: float) -> dict:
    """One queue line. Keypoints below ``min_likelihood`` or NaN are left unplaced."""
    preds = [
        {"keypoint_name": k, "x": float(x), "y": float(y)}
        for k, (x, y), p in zip(keypoints, xy, lik, strict=True)
        if p >= min_likelihood and pd.notna(x) and pd.notna(y)
    ]
    return {
        "frame_path": path,
        "predictions": {
            "model_name": model_name,
            "date_time": int(time.time() * 1000),
            "predictions": preds,
        },
    }


def append_queue(csv_path: Path, entries: list[dict]) -> int:
    """Add entries to the label file's queue, skipping frames already in it. Returns count added."""
    side = sidecar_path(csv_path)
    lines = side.read_text(encoding="utf-8").splitlines() if side.exists() else []
    lines = [line for line in lines if line.strip()]
    present = {json.loads(line)["frame_path"] for line in lines}
    new = [e for e in entries if e["frame_path"] not in present]
    if new or not side.exists():
        _write_atomic(side, "\n".join(lines + [json.dumps(e) for e in new]) + "\n")
    return len(new)
