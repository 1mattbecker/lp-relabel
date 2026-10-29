"""Contact sheets of selected frames with their predicted keypoints, one PNG per session.

Reads a selection log (``select``, also with ``--dry-run``) and the manifest it was
made from, decodes each picked frame from its video (``frames.extract_with_context``
with no context, so the images are the ones labeling would use), and draws the
predictions: one colour per keypoint; filled marker for likelihood >= 0.9, hollow
for 0.5-0.9. Below 0.5 a keypoint is usually just absent (tongue, paws), so only the
``--always-visible`` keypoints are drawn there, as grey crosses, and named in the
title with the frame, why it was picked and its scores.

    uv run --with matplotlib python scripts/contact_sheet.py \\
        proj/CollectedData.selection.tsv manifest.csv sheets/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lp_relabel.frames import extract_with_context, resolve_video
from lp_relabel.poses import read_pose_csv

HIGH, LOW = 0.9, 0.5


def draw_frame(ax, image, xy, lik, keypoints, colors, always_visible) -> list[str]:
    """Draw one frame and its keypoints; return always-visible keypoints below ``LOW``."""
    ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    low = []
    for k, name in enumerate(keypoints):
        x, y, p = xy[k, 0], xy[k, 1], lik[k]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        if p >= HIGH:
            ax.plot(x, y, "o", ms=6, mfc=colors[k], mec="white", mew=0.6)
        elif p >= LOW:
            ax.plot(x, y, "o", ms=7, mfc="none", mec=colors[k], mew=1.6)
        elif name in always_visible:
            ax.plot(x, y, "x", ms=7, color="0.75", mew=1.6)
            low.append(name)
    ax.set_xticks([])
    ax.set_yticks([])
    return low


def session_sheet(session, picks, pose, video, cache, out, cols, always_visible) -> Path:
    """Write one session's sheet."""
    track = read_pose_csv(pose)
    paths = extract_with_context(resolve_video(video), picks.frame, cache / session, context=0)
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(track.keypoints)))
    rows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.6, rows * 3.4 + 0.9), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, pick in zip(axes.flat, picks.itertuples()):
        ax.axis("on")
        image = plt.imread(paths[int(pick.frame)])
        r = track.row(int(pick.frame))
        low = draw_frame(
            ax, image, track.xy[r], track.likelihood[r], track.keypoints, colors, always_visible
        )
        ax.set_title(
            f"{pick.frame}  {pick.reason}  lik {pick.lik_score:.2f}  jump {pick.jump_score:.0f}\n"
            f"low: {', '.join(low) if low else '-'}",
            fontsize=8,
            color="firebrick" if low else "black",
        )
    handles = [
        Line2D([], [], marker="o", ls="", mfc=c, mec="white", label=k)
        for k, c in zip(track.keypoints, colors)
    ]
    handles += [
        Line2D([], [], marker="o", ls="", mfc="none", mec="k", label=f"{LOW}-{HIGH}"),
        Line2D([], [], marker="x", ls="", color="0.75", label=f"<{LOW} (always-visible only)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=7, frameon=False)
    fig.suptitle(session, fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97), h_pad=1.6)
    path = out / f"{session}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("selection", type=Path, help="selection log (.tsv) from select")
    parser.add_argument("manifest", type=Path, help="manifest CSV with session, pose, video")
    parser.add_argument("out", type=Path, help="folder for the PNGs")
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--sessions", nargs="*", help="only these sessions")
    parser.add_argument(
        "--always-visible",
        nargs="*",
        default=["nose_tip", "jaw", "spout_r", "spout_l"],
        help="keypoints expected in every frame: drawn and named when below 0.5",
    )
    args = parser.parse_args()

    selection = pd.read_csv(args.selection, sep="\t")
    manifest = pd.read_csv(args.manifest).set_index("session")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / ".frames"
    for session, picks in selection.groupby("session", sort=False):
        if args.sessions and session not in args.sessions:
            continue
        row = manifest.loc[session]
        path = session_sheet(
            session, picks, row.pose, row.video, cache, args.out, args.cols, set(args.always_visible)
        )
        print(path)


if __name__ == "__main__":
    main()
