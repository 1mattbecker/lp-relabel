import json

import numpy as np
import pandas as pd
import pytest
from conftest import decode_image, make_pose_csv

from lp_relabel.cli import main


def run_select(tmp_path, video, pose, *extra):
    main(["select", "--session", "mouse1/2025-07-03", "--pose", str(pose), "--video", str(video),
          "--data-dir", str(tmp_path / "proj"), "--model-name", "m1", "-n", "4",
          "--min-gap", "40", *extra])  # fmt: skip


def test_select_writes_an_app_project(tmp_path, video):
    pose = make_pose_csv(tmp_path / "pred.csv", 600)
    df = pd.read_csv(pose, header=[0, 1, 2], index_col=0)
    df.iloc[200, 3] += 50  # tongue_tip x spike at frame 200
    df.iloc[450, 5] = 0.5  # tongue_tip likelihood 0.5 at frame 450
    df.to_csv(pose)

    run_select(tmp_path, video, pose, "--random-fraction", "0")
    proj = tmp_path / "proj"
    assert "keypoint_names" in (proj / "project.yaml").read_text()
    header = pd.read_csv(proj / "CollectedData.csv", header=[0, 1, 2], index_col=0)
    assert len(header) == 0 and list(header.columns.get_level_values(1)[::2]) == ["jaw", "tongue_tip"]

    lines = [json.loads(x) for x in (proj / "CollectedData.unlabeled.jsonl").read_text().splitlines()]
    paths = [e["frame_path"] for e in lines]
    assert "labeled-data/mouse1_2025-07-03/img00000200.jpg" in paths
    assert "labeled-data/mouse1_2025-07-03/img00000450.jpg" in paths
    for e in lines:
        frame = int(e["frame_path"][-12:-4])
        for f in range(frame - 2, frame + 3):
            img = proj / "labeled-data" / "mouse1_2025-07-03" / f"img{f:08d}.jpg"
            assert decode_image(img) == f
        names = [p["keypoint_name"] for p in e["predictions"]["predictions"]]
        assert e["predictions"]["model_name"] == "m1"
        assert "jaw" in names
    sel = pd.read_csv(proj / "CollectedData.selection.tsv", sep="\t")
    assert len(sel) == 4 and set(sel.reason) == {"score"}

    # A second round skips queued frames and their neighbourhood.
    run_select(tmp_path, video, pose, "--seed", "1")
    lines2 = (proj / "CollectedData.unlabeled.jsonl").read_text().splitlines()
    assert len(lines2) == 8 and len(set(lines2)) == 8


def test_length_mismatch_is_refused(tmp_path, video):
    pose = make_pose_csv(tmp_path / "pred.csv", 599)
    with pytest.raises(SystemExit, match="599 rows"):
        run_select(tmp_path, video, pose)


def test_merge(tmp_path, video):
    pose = make_pose_csv(tmp_path / "pred.csv", 600)
    run_select(tmp_path, video, pose)
    proj = tmp_path / "proj"
    queued = [json.loads(x)["frame_path"] for x in (proj / "CollectedData.unlabeled.jsonl").read_text().splitlines()]
    # Pretend the labeler saved two frames.
    cols = pd.MultiIndex.from_product([["scorer"], ["jaw", "tongue_tip"], ["x", "y"]])
    new = pd.DataFrame(np.ones((2, 4)), index=queued[:2], columns=cols)
    new.to_csv(proj / "CollectedData.csv")
    base = pd.DataFrame(np.zeros((1, 4)), index=["labeled-data/old/img00000010.png"],
                        columns=pd.MultiIndex.from_product([["anna"], ["jaw", "tongue_tip"], ["x", "y"]]))  # fmt: skip
    base.to_csv(tmp_path / "base.csv")
    out = tmp_path / "merged.csv"
    main(["merge", str(tmp_path / "base.csv"), str(proj / "CollectedData.csv"), "-o", str(out)])
    merged = pd.read_csv(out, header=[0, 1, 2], index_col=0)
    assert len(merged) == 3 and set(merged.columns.get_level_values(0)) == {"anna"}
    # Image check: the new frames and their context exist, the old one does not.
    with pytest.raises(SystemExit, match="missing"):
        main(["merge", str(tmp_path / "base.csv"), str(proj / "CollectedData.csv"),
              "-o", str(out), "--data-dir", str(proj)])  # fmt: skip
    main(["merge", str(proj / "CollectedData.csv"), str(proj / "CollectedData.csv"), "-o", str(out),
          "--data-dir", str(proj)])  # fmt: skip
    assert len(pd.read_csv(out, header=[0, 1, 2], index_col=0)) == 2
