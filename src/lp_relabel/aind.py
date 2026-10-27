"""Find sessions with Lightning Pose predictions in AIND's public S3 bucket.

The video-processing pipeline writes one ``<session>_videoprocessed_<date>``
asset per run to ``aind-open-data``. Its ``config.yaml`` names the LP model and
the camera; ``pred_outputs/video_preds/video_predictions.csv`` holds one row per
video frame. The raw video is in the session asset under ``behavior-videos/``.
All of it is readable without credentials over HTTPS.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

BUCKET_URL = "https://aind-open-data.s3.amazonaws.com"
# Files are named after the video's stem: "video" for BottomCamera/video.mp4,
# "bottom_camera" for older sessions' bottom_camera.mp4.
PREDICTIONS = "pred_outputs/video_preds/{stem}_predictions.csv"
CONFIDENCE = "eval_outputs/{stem}/eval_metrics/confidence_bodypart_mean.csv"
_ASSET_RE = re.compile(r"^(behavior_(\d+)_[\d-]+_[\d-]+)_videoprocessed_([\d-]+_[\d-]+)/$")


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


def list_prefixes(prefix: str) -> list[str]:
    """Top-level 'folders' of the bucket starting with ``prefix``."""
    out, token = [], None
    while True:
        q = {"list-type": "2", "prefix": prefix, "delimiter": "/", "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        text = _get(f"{BUCKET_URL}/?{urllib.parse.urlencode(q)}").decode()
        out += re.findall(r"<CommonPrefixes><Prefix>([^<]*)</Prefix>", text)
        m = re.search(r"<NextContinuationToken>([^<]*)<", text)
        if not m:
            return out
        token = m.group(1)


def _asset_info(asset: str) -> dict | None:
    m = _ASSET_RE.match(asset)
    if not m:
        return None
    try:
        config = yaml.safe_load(_get(f"{BUCKET_URL}/{asset}config.yaml")) or {}
    except Exception as e:  # missing or unreadable config: not usable
        logger.debug("%s: %s", asset, e)
        return None
    model = (config.get("lightning_pose_inference") or {}).get("model_name", "")
    session = config.get("behavior_data_asset_name") or m.group(1)
    subpath = config.get("behavior_subpath", "")
    stem = subpath.rsplit("/", 1)[-1].rsplit(".", 1)[0] or "video"
    row = {
        "session": session,
        "mouse": m.group(2),
        "asset": asset.rstrip("/"),
        "processed": m.group(3),
        "model": model,
        # "BottomCamera/video.mp4" (newer sessions) or "bottom_camera.mp4" (older)
        "camera": subpath.split("/")[0] if "/" in subpath else subpath.rsplit(".", 1)[0],
        "pose": f"{BUCKET_URL}/{asset}{PREDICTIONS.format(stem=stem)}",
        "video": f"{BUCKET_URL}/{session}/behavior-videos/{subpath}",
    }
    try:  # per-keypoint mean likelihood, "0.92 ± 0.18" -> 0.92
        conf = pd.read_csv(f"{BUCKET_URL}/{asset}{CONFIDENCE.format(stem=stem)}")
        for col in conf.columns[1:]:
            row[f"conf_{col}"] = float(str(conf[col].iloc[0]).split()[0])
    except Exception:
        pass
    return row


def find_sessions(model: str, mice: list[str] | None = None, workers: int = 32) -> pd.DataFrame:
    """Sessions whose newest videoprocessed asset was made by a model matching ``model``.

    ``model`` is matched as a substring of the model path in the asset config.
    ``mice`` limits the search to those subject IDs (much faster).
    """
    prefixes = [f"behavior_{m}_" for m in mice] if mice else [f"behavior_{d}" for d in "0123456789"]
    with ThreadPoolExecutor(min(workers, len(prefixes))) as ex:
        assets = [a for found in ex.map(list_prefixes, prefixes) for a in found if "_videoprocessed_" in a]
    with ThreadPoolExecutor(workers) as ex:
        rows = [r for r in ex.map(_asset_info, assets) if r and model in r["model"]]
    if not rows:
        return pd.DataFrame(columns=["session", "mouse", "asset", "processed", "model", "camera", "pose", "video"])
    df = pd.DataFrame(rows).sort_values(["session", "processed"])
    return df.drop_duplicates("session", keep="last").reset_index(drop=True)
