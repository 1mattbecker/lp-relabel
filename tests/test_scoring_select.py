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



def sparse_track(windows=((1000, 1300), (5000, 5300)), k=2, seed=0):
    frames = np.concatenate([np.arange(a, b) for a, b in windows])
    rng = np.random.default_rng(seed)
    xy = 50 + rng.normal(0, 0.5, (len(frames), k, 2))
    return PoseTrack([f"kp{i}" for i in range(k)], xy, np.full((len(frames), k), 0.99), frames)


def test_sparse_scores_keep_video_frames_and_skip_window_gaps():
    t = sparse_track()
    t.xy[299, 0] += 40  # last row of the first window: no next frame, so no jump there
    t.xy[450, 0] += 40  # frame 5150 inside the second window
    scores = frame_scores(t)
    assert scores.frame.iloc[0] == 1000 and scores.frame.iloc[-1] == 5299
    assert scores.jump_score.iloc[299] == 0 and scores.jump_score.iloc[300] == 0
    assert scores.frame.iloc[int(scores.jump_score.idxmax())] == 5150


def test_sparse_picks_use_frame_numbers():
    t = sparse_track()
    for row in (0, 150, 155, 299, 450):
        t.xy[row, 1] += 40
    picks = pick_frames(frame_scores(t), n=4, min_gap=20, margin=2, random_fraction=0,
                        exclude={5150})  # fmt: skip
    frames = list(picks.frame)
    assert 1000 not in frames and 1299 not in frames  # context stays inside the window
    assert not (1150 in frames and 1155 in frames)  # min_gap in frames
    assert all(abs(f - 5150) >= 20 for f in frames)  # excluded frame
    allowed = read_windows_from(t.frames, [(5000, 5100)])
    assert np.flatnonzero(allowed).tolist() == list(range(300, 400))


def read_windows_from(frames, pairs, tmp=None):
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame(pairs, columns=["start_frame", "end_frame"]).to_csv(f.name, index=False)
        return read_windows(f.name, frames)


def test_read_pose_csv_reads_sparse_frame_index(tmp_path):
    cols = pd.MultiIndex.from_product([["s"], ["a"], ["x", "y", "likelihood"]])
    p = tmp_path / "p.csv"
    pd.DataFrame(np.ones((4, 3)), columns=cols, index=[10, 11, 500, 501]).to_csv(p)
    track = read_pose_csv(p)
    assert track.frames.tolist() == [10, 11, 500, 501] and not track.is_dense and track.row(500) == 2
    pd.DataFrame(np.ones((3, 3)), columns=cols, index=[5, 4, 6]).to_csv(p)
    with pytest.raises(ValueError, match="increasing"):
        read_pose_csv(p)
