"""Command line: ``lp-relabel select`` and ``lp-relabel merge``."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from lp_relabel import __version__, project
from lp_relabel.poses import read_pose_csv
from lp_relabel.scoring import LIKELIHOOD_MODES, frame_scores
from lp_relabel.select import pick_frames, read_windows

logger = logging.getLogger("lp_relabel")


def _sessions(args) -> pd.DataFrame:
    if args.manifest:
        df = pd.read_csv(args.manifest)
        need = {"session", "pose", "video"}
        if not need <= set(df.columns):
            raise SystemExit(f"{args.manifest}: needs columns {sorted(need)} (optional: windows)")
        base = Path(args.manifest).parent
        for col in ("pose", "windows"):
            if col in df.columns:
                df[col] = [
                    v if pd.isna(v) or "://" in str(v) or Path(str(v)).is_absolute() else str(base / str(v))
                    for v in df[col]
                ]
        return df
    if not (args.session and args.pose and args.video):
        raise SystemExit("give --manifest, or all of --session, --pose and --video")
    return pd.DataFrame(
        [{"session": args.session, "pose": args.pose, "video": args.video, "windows": args.windows}]
    )


def cmd_select(args) -> None:
    from lp_relabel.frames import extract_with_context, resolve_video

    data_dir = Path(args.data_dir)
    label_csv = data_dir / f"{args.label_file}.csv"
    excluded = project.known_frames(data_dir, [label_csv] + [Path(p) for p in args.exclude_labels])
    keypoints_all: list[str] | None = None

    for row in _sessions(args).itertuples(index=False):
        session = str(row.session)
        folder = project.safe_folder(session)
        track = read_pose_csv(row.pose, args.cache_dir)
        if keypoints_all is None:
            keypoints_all = track.keypoints
            if not args.dry_run:
                project.ensure_project_yaml(data_dir, keypoints_all)
                project.ensure_label_csv(label_csv, keypoints_all)
        elif track.keypoints != keypoints_all:
            raise SystemExit(f"{row.pose}: keypoints {track.keypoints} differ from {keypoints_all}")

        scored = track.subset(args.keypoints) if args.keypoints else track
        scores = frame_scores(
            scored, args.likelihood_mode, args.jump_min_likelihood, args.likelihood_weight, args.jump_weight
        )
        windows = getattr(row, "windows", None)
        allowed = None
        if windows is not None and not pd.isna(windows) and str(windows):
            allowed = read_windows(windows, track.n_frames, args.pre, args.post)
        exclude = {f for (fold, f) in excluded if fold == folder}
        picks = pick_frames(
            scores, args.n, args.min_gap, args.context, args.random_fraction, allowed, exclude, args.seed
        )
        logger.info("%s: picked %d frames (%d by score)", session, len(picks), (picks.reason == "score").sum())

        if not args.dry_run and len(picks):
            from aind_video_utils import read_mp4_frame_index

            video = resolve_video(str(row.video))
            index = read_mp4_frame_index(video)
            if index.n_samples != track.n_frames and not args.allow_length_mismatch:
                raise SystemExit(
                    f"{session}: video has {index.n_samples} frames, predictions have {track.n_frames} rows. "
                    "Rows would not line up with frames; check the pair, or pass --allow-length-mismatch."
                )
            extract_with_context(
                video, picks["frame"], data_dir / project.LABELED_DATA / folder, args.context, index=index
            )
            entries = [
                project.queue_entry(
                    project.frame_path(folder, f), args.model_name, track.keypoints,
                    track.xy[f], track.likelihood[f], args.prefill_min_likelihood,
                )  # fmt: skip
                for f in picks["frame"]
            ]
            added = project.append_queue(label_csv, entries)
            logger.info("%s: queued %d frames in %s", session, added, project.sidecar_path(label_csv).name)

        picks.insert(0, "session", session)
        picks.insert(1, "video", str(row.video))
        picks["frame_path"] = [project.frame_path(folder, f) for f in picks["frame"]]
        picks["model_name"] = args.model_name
        picks["created"] = datetime.now().isoformat(timespec="seconds")
        out = Path(args.selection_out) if args.selection_out else data_dir / f"{args.label_file}.selection.tsv"
        out.parent.mkdir(parents=True, exist_ok=True)
        picks.to_csv(out, mode="a", header=not out.exists(), index=False, sep="\t")


def cmd_merge(args) -> None:
    frames = [pd.read_csv(p, header=[0, 1, 2], index_col=0) for p in [args.base, *args.new]]
    base = frames[0]
    scorer = base.columns.get_level_values(0)[0]
    out = []
    for path, df in zip([args.base, *args.new], frames, strict=True):
        df = df.copy()
        df.columns = pd.MultiIndex.from_tuples(
            [(scorer, b, c) for _, b, c in df.columns], names=base.columns.names
        )
        if set(df.columns) != set(base.columns):
            raise SystemExit(f"{path}: columns differ from {args.base}")
        out.append(df[base.columns])
    merged = pd.concat(out)
    dup = merged.index.duplicated(keep="last")
    if dup.any():
        logger.warning("%d frames appear more than once; keeping the later file's labels", dup.sum())
    merged = merged[~dup]

    if args.data_dir:
        root = Path(args.data_dir)
        missing = []
        for idx in merged.index.astype(str):
            parsed = project.parse_frame_path(idx)
            paths = [root / idx]
            if parsed and args.context:
                folder, f = parsed
                ext = Path(idx).suffix
                width = len(Path(idx).stem) - 3
                paths = [
                    root / project.LABELED_DATA / folder / f"img{g:0{width}d}{ext}"
                    for g in range(max(f - args.context, 0), f + args.context + 1)
                ]
            missing += [str(p) for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"{len(missing)} image files missing under {root}, e.g. {missing[:3]}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out)
    logger.info("wrote %d labeled frames (%d from base) to %s", len(merged), len(base), args.out)


def cmd_aind_sessions(args) -> None:
    from lp_relabel.aind import find_sessions

    df = find_sessions(args.model, args.mice)
    if args.camera:
        df = df[df.camera == args.camera]
    logger.info("%d sessions from %d mice", len(df), df.mouse.nunique() if len(df) else 0)
    if args.per_mouse and len(df):
        df = (
            df.sample(frac=1, random_state=args.seed)
            .groupby("mouse", sort=False)
            .head(args.per_mouse)
            .sort_values("session")
        )
        logger.info("kept %d (up to %d per mouse)", len(df), args.per_mouse)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("wrote %s", args.out)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="lp-relabel", description=__doc__)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser(
        "select",
        help="pick frames from predictions, extract them, queue them in a Lightning Pose App project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = s.add_argument_group("input (a manifest, or one session)")
    src.add_argument("--manifest", help="CSV with columns session, pose, video and optional windows")
    src.add_argument("--session", help="session name; becomes the labeled-data folder")
    src.add_argument("--pose", help="Lightning Pose predictions CSV for the video")
    src.add_argument("--video", help="MP4 path, http(s) URL, or s3:// URI")
    src.add_argument("--windows", help="CSV of start_frame,end_frame or of event frames to search within")
    src.add_argument("--pre", type=int, default=0, help="frames before each event frame")
    src.add_argument("--post", type=int, default=0, help="frames after each event frame")

    o = s.add_argument_group("output")
    o.add_argument("--data-dir", required=True, help="Lightning Pose App project data directory")
    o.add_argument("--label-file", default="CollectedData",
                   help="label file name without .csv; LP App 2.4.2.3 only accepts CollectedData")
    o.add_argument("--model-name", default="predictions", help="name shown with the prefilled keypoints")
    o.add_argument("--selection-out", help="selection log, tab-separated (default <data-dir>/<label-file>.selection.tsv)")
    o.add_argument("--exclude-labels", nargs="*", default=[], help="label CSVs whose frames to skip")
    o.add_argument("--dry-run", action="store_true", help="score and pick only; write the selection log")
    o.add_argument("--cache-dir", help="where downloaded prediction CSVs are kept (default ~/.cache/lp-relabel)")

    c = s.add_argument_group("selection")
    c.add_argument("-n", type=int, default=10, help="frames per session")
    c.add_argument("--min-gap", type=int, default=250, help="minimum frames between picks")
    c.add_argument("--random-fraction", type=float, default=0.2, help="share of picks drawn at random")
    c.add_argument("--context", type=int, default=2, help="neighbour frames written on each side")
    c.add_argument("--keypoints", nargs="*", help="keypoints to score (default all)")
    c.add_argument("--likelihood-mode", choices=LIKELIHOOD_MODES, default="uncertain")
    c.add_argument("--jump-min-likelihood", type=float, default=0.5)
    c.add_argument("--likelihood-weight", type=float, default=1.0)
    c.add_argument("--jump-weight", type=float, default=1.0)
    c.add_argument("--prefill-min-likelihood", type=float, default=0.5,
                   help="leave keypoints below this likelihood unplaced in the labeler")  # fmt: skip
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--allow-length-mismatch", action="store_true")
    s.set_defaults(func=cmd_select)

    a = sub.add_parser(
        "aind-sessions",
        help="list AIND sessions with predictions from a model (public S3), as a manifest for select",
    )
    a.add_argument("--model", required=True, help="substring of the model path, e.g. Pylon2Test2")
    a.add_argument("--mice", nargs="*", help="subject IDs to search (default: all)")
    a.add_argument("--camera", help="keep one camera, e.g. BottomCamera")
    a.add_argument("--per-mouse", type=int, help="keep at most this many random sessions per mouse")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("-o", "--out", required=True, help="manifest CSV to write")
    a.set_defaults(func=cmd_aind_sessions)

    m = sub.add_parser("merge", help="append new label CSVs to a training label CSV")
    m.add_argument("base", help="existing training label CSV")
    m.add_argument("new", nargs="+", help="label CSVs to add (later files win on duplicate frames)")
    m.add_argument("-o", "--out", required=True, help="merged CSV to write")
    m.add_argument("--data-dir", help="check every frame and its context images exist under this directory")
    m.add_argument("--context", type=int, default=2, help="context frames to check on each side")
    m.set_defaults(func=cmd_merge)
    return p


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stderr)
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
