import numpy as np
import pandas as pd
import pytest

from lp_relabel.poses import PoseTrack, read_pose_csv
from lp_relabel.scoring import frame_scores, jump_score, likelihood_score
from lp_relabel.select import pick_frames, read_windows


def flat_track(n=1000, k=2, seed=0):
    rng = np.random.default_rng(seed)
    xy = 50 + rng.normal(0, 0.5, (n, k, 2))
    return PoseTrack([f"kp{i}" for i in range(k)], xy, np.full((n, k), 0.99))


def test_uncertain_mode_ignores_confident_absence():
    lik = np.array([[0.01], [0.5], [0.99]])
    assert np.argmax(likelihood_score(lik, "uncertain")) == 1
    assert np.argmax(likelihood_score(lik, "low")) == 0


def test_one_frame_spike_scores_highest():
    t = flat_track()
    t.xy[400, 1] += 40
    assert np.argmax(jump_score(t.xy, t.likelihood)) == 400


def test_jump_ignores_low_likelihood_frames():
    t = flat_track()
    t.xy[400, 1] += 40
    t.likelihood[400, 1] = 0.1
    assert jump_score(t.xy, t.likelihood)[400] < 5


def test_pick_respects_gap_margin_exclude_and_windows():
    t = flat_track()
    for f in (1, 300, 310, 600, 900):
        t.xy[f, 0] += 40
    scores = frame_scores(t)
    allowed = np.zeros(1000, bool)
    allowed[:800] = True
    picks = pick_frames(scores, n=3, min_gap=50, margin=2, random_fraction=0,
                        allowed=allowed, exclude={600})  # fmt: skip
    frames = list(picks.frame)
    assert 300 in frames or 310 in frames
    assert not (300 in frames and 310 in frames)  # min_gap
    assert 1 not in frames  # margin keeps context frames inside the video
    assert all(abs(f - 600) >= 50 for f in frames)  # excluded neighbourhood
    assert all(f < 800 for f in frames)  # windows
    assert all(np.diff(frames) >= 50)


def test_random_fraction():
    picks = pick_frames(frame_scores(flat_track()), n=10, min_gap=20, margin=2, random_fraction=0.3)
    assert len(picks) == 10 and (picks.reason == "random").sum() == 3


def test_read_windows_both_forms(tmp_path):
    p = tmp_path / "w.csv"
    pd.DataFrame({"start_frame": [10], "end_frame": [20]}).to_csv(p, index=False)
    assert np.flatnonzero(read_windows(p, 100)).tolist() == list(range(10, 20))
    pd.DataFrame({"frame": [50]}).to_csv(p, index=False)
    assert np.flatnonzero(read_windows(p, 100, pre=2, post=3)).tolist() == list(range(48, 54))


def test_read_pose_csv_rejects_non_frame_index(tmp_path):
    cols = pd.MultiIndex.from_product([["s"], ["a"], ["x", "y", "likelihood"]])
    p = tmp_path / "p.csv"
    pd.DataFrame(np.ones((3, 3)), columns=cols, index=["img1.png", "img2.png", "img3.png"]).to_csv(p)
    with pytest.raises(ValueError, match="frame numbers"):
        read_pose_csv(p)
