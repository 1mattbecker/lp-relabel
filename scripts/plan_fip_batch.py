"""Plan an lp-capsule predict batch over curated FIP sessions: one mid-task window each.

Sessions are the curated FIP list in kinematics_analysis whose bottom camera passes
the video screen (``inputs/video_screen_fip.csv``, ``use``). Per mouse and physical
rig (``rig_id`` without its date suffix), the earliest and latest such session are
taken, so the batch spans mice, rigs and training stages.

Each window is ``--seconds`` long, centred mid-task:

- cameras that lost no frames (the screen's ``frames_lost``): on the go cue of the
  middle trial, put on frames from the first frame's Harp time and the frame
  interval, read from the first rows of the camera CSV only (the CSVs run to
  120 MB). Without lost frames, row ``i`` is ``i`` intervals after the first.
- cameras that lost frames: on the middle of ``video_alignment.task_frame_window``
  (first trial start to last trial end), which puts trial times on frames through
  the timing correction, with the Harp trigger log; it reads the whole CSV.

    python scripts/plan_fip_batch.py ../kinematics_analysis/inputs fip_batch.yaml
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from aind_dynamic_foraging_behavior_video_analysis.video_alignment import (
    _resolve_time_column,
    read_trial_times,
    read_video_csv,
    task_frame_window,
)

BUCKET = "https://aind-open-data.s3.amazonaws.com"
MODEL = "ephys_BottomCamera_Pylon2Test2"
# Enough rows to estimate the frame interval.
HEAD_BYTES = 256 * 1024
TRIGGER_LOG = "behavior/raw.harp/BehaviorEvents/Event_94.bin"


def fetch(url: str, n_bytes: int | None = None) -> bytes:
    """Return a URL's bytes, or only its first ``n_bytes``."""
    request = urllib.request.Request(url)
    if n_bytes:
        request.add_header("Range", f"bytes=0-{n_bytes - 1}")
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def session_files(session: str, source_camera: str) -> tuple[str, str]:
    """Return the camera CSV and the behavior JSON of a session, relative to it."""
    if source_camera == "BottomCamera":
        video_csv = "behavior-videos/BottomCamera/metadata.csv"
    else:
        video_csv = f"behavior-videos/{source_camera}.csv"
    stem = session.removeprefix("behavior_")
    return video_csv, f"behavior/{stem}.json"


def first_frame_and_interval(csv_url: str) -> tuple[float, float]:
    """Return the first frame's Harp time and the median frame interval."""
    head = fetch(csv_url, HEAD_BYTES)
    head = head[: head.rfind(b"\n") + 1]  # drop the partial last line
    with tempfile.NamedTemporaryFile(suffix=".csv") as tmp:
        tmp.write(head)
        tmp.flush()
        table = read_video_csv(tmp.name)
    times = table[_resolve_time_column(table)].to_numpy(dtype=float)
    return float(times[0]), float(np.median(np.diff(times)))


def plan_window(row: dict, seconds: float) -> dict:
    """Return the session's mid-task window, or the reason it has none."""
    session = row["raw_session"]
    video_csv, behavior_json = session_files(session, row["bottom_source_camera"])
    url = f"{BUCKET}/{session}"
    try:
        trials = read_trial_times(f"{url}/{behavior_json}")
        go = np.sort(trials["goCue_start_time"].dropna().to_numpy(dtype=float))
        mid_go = float(go[len(go) // 2])
        t0, dt = first_frame_and_interval(f"{url}/{video_csv}")
        if float(row.get("frames_lost") or 0) > 0:
            with tempfile.TemporaryDirectory() as tmp:
                csv_path, log_path = Path(tmp) / "video.csv", Path(tmp) / "Event_94.bin"
                csv_path.write_bytes(fetch(f"{url}/{video_csv}"))
                try:
                    log_path.write_bytes(fetch(f"{url}/{TRIGGER_LOG}"))
                except Exception:  # noqa: BLE001 -- task_frame_window copes without
                    log_path = None
                task_start, task_end = task_frame_window(f"{url}/{behavior_json}", csv_path, log_path)
            centre, method = (task_start + task_end) // 2, "task_frame_window middle"
        else:
            centre, method = int(round((mid_go - t0) / dt)), "middle go cue"
    except Exception as e:  # noqa: BLE001 -- recorded, the session is skipped
        return {**row, "error": f"{type(e).__name__}: {e}"}
    n = int(round(seconds / dt))
    start = max(centre - n // 2, 0)
    return {
        **row,
        "n_trials": len(go),
        "mid_go_cue_harp_s": round(mid_go, 3),
        "fps": round(1 / dt, 2),
        "start_frame": start,
        "end_frame": start + n,
        "window_method": method,
        "error": "",
    }


def pick_sessions(sessions: pd.DataFrame, per_pair: int) -> pd.DataFrame:
    """Per mouse and rig, the earliest and latest screened session (up to ``per_pair``)."""
    ok = sessions[sessions.bottom_use.astype(str) == "True"].copy()
    ok["rig"] = ok.rig_id.str.rsplit("_", n=1).str[0]
    ok = ok.sort_values("raw_session")
    picked = []
    for _, group in ok.groupby(["subject", "rig"]):
        ends = [group.iloc[0], group.iloc[-1]] if len(group) > 1 else [group.iloc[0]]
        picked.extend(ends[:per_pair])
    return pd.DataFrame(picked).reset_index(drop=True)


def screened_sessions(inputs: Path) -> pd.DataFrame:
    """Join the curated FIP list, its bottom-camera screen, and each session's rig."""
    sessions = pd.read_csv(inputs / "me_sessions_fip_curated.csv")
    screen = pd.read_csv(inputs / "video_screen_fip.csv")
    bottom = screen[screen.camera == "BottomCamera"].rename(
        columns={
            "session": "raw_session",
            "use": "bottom_use",
            "source_camera": "bottom_source_camera",
        }
    )[["raw_session", "bottom_use", "bottom_source_camera", "frames_lost"]]
    out = sessions.merge(bottom, on="raw_session", how="left")

    def rig_id(session: str) -> str | None:
        try:
            return json.loads(fetch(f"{BUCKET}/{session}/session.json")).get("rig_id")
        except Exception:  # noqa: BLE001 -- no rig, the session is skipped
            return None

    with ThreadPoolExecutor(16) as pool:
        out["rig_id"] = list(pool.map(rig_id, out.raw_session))
    return out[out.rig_id.notna()]


def main() -> None:
    """Pick sessions, plan their windows, and write the batch and its table."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", type=Path, help="kinematics_analysis/inputs")
    parser.add_argument("out", type=Path, help="predict batch YAML for lp-capsule")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--per-pair", type=int, default=2, help="sessions per mouse x rig")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    picked = pick_sessions(screened_sessions(args.inputs), args.per_pair)
    with ThreadPoolExecutor(8) as pool:
        rows = list(pool.map(lambda r: plan_window(r, args.seconds), picked.to_dict("records")))
    table = pd.DataFrame(rows)
    table.to_csv(args.out.with_suffix(".sessions.csv"), index=False)
    good = table[table.error == ""]
    for _, r in table[table.error != ""].iterrows():
        print(f"  skipped {r.raw_session}: {r.error}", file=sys.stderr)
    batch = {
        "task": "predict",
        "model": args.model,
        "videos": [
            {
                "asset": r.raw_session,
                "camera": "BottomCamera",
                "windows": [[int(r.start_frame), int(r.end_frame)]],
            }
            for r in good.itertuples()
        ],
    }
    header = (
        f"# {len(good)} curated FIP sessions, one {args.seconds:g} s window each, centred on\n"
        f"# the middle trial's go cue; {good.subject.nunique()} mice, {good.rig.nunique()} rigs.\n"
        f"# Made by lp-relabel scripts/plan_fip_batch.py; per-session table beside it.\n"
    )
    args.out.write_text(header + yaml.safe_dump(batch, sort_keys=False, default_flow_style=None))
    print(f"wrote {args.out}: {len(good)} sessions, {good.subject.nunique()} mice, "
          f"{good.rig.nunique()} rigs ({len(table) - len(good)} skipped)")


if __name__ == "__main__":
    main()
