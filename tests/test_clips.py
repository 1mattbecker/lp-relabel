import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from aind_video_utils import extract_frame_by_index, read_mp4_frame_index
from conftest import decode_image, make_video
from PIL import Image

from lp_relabel import clips
from lp_relabel.cli import main

CLIP = "sess_clip_001"
# The stream-copied clip displays source frames 120..219: clip frame k is source frame 120 + k.
FIRST = 120
LABELED = [0, 1, 50, 99]


def write_labels(path, images):
    cols = pd.MultiIndex.from_product([["scorer"], ["jaw"], ["x", "y"]], names=["scorer", "bodyparts", "coords"])
    pd.DataFrame(np.ones((len(images), 2)), index=images, columns=cols).to_csv(path)
    return path


def frame_png(video, k, path):
    """Frame k of a video as ffmpeg displays it (edit list honoured), as a labeler saw it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{k})",
                    "-frames:v", "1", str(path)], check=True)  # fmt: skip
    return path


@pytest.fixture(scope="module")
def stream_copied(tmp_path_factory):
    """A stream-copied cut starting mid-GOP: 124 samples, an edit list hiding the first 24."""
    tmp = tmp_path_factory.mktemp("src")
    src = make_video(tmp / "long.mp4", 600)
    clip = tmp / "clip.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "0.24", "-i", str(src), "-t", "0.2",
                    "-c", "copy", str(clip)], check=True)  # fmt: skip
    index = read_mp4_frame_index(clip)
    assert index.n_samples == 124 and not index.is_frame_addressing_safe()
    return clip


@pytest.fixture
def source(tmp_path, stream_copied):
    root = tmp_path / "source"
    (root / "videos").mkdir(parents=True)
    (root / "videos" / f"{CLIP}.mp4").write_bytes(stream_copied.read_bytes())
    images = [f"labeled-data/{CLIP}/img{k:03d}.png" for k in LABELED]
    for k, image in zip(LABELED, images):
        frame_png(stream_copied, k, root / image)
    write_labels(root / "CollectedData.csv", images)
    return root


def clean(source, out, *extra):
    main(["clean-clips", "--labels", str(source / "CollectedData.csv"), "--source", str(source),
          "--out", str(out), "--workers", "2", *extra])  # fmt: skip


def test_stream_copied_clip_is_cleaned_and_verified(source, tmp_path):
    out = tmp_path / "clean"
    clean(source, out)
    checks = pd.read_csv(out / "checks.csv")
    assert list(checks.frame) == LABELED
    assert set(checks.clip_frames) == {100} and set(checks.source_samples) == {124}
    assert checks.labeled_vs_clip_mean_abs_diff.max() <= clips.MAX_DIFF

    video = out / "videos" / f"{CLIP}.mp4"
    index = read_mp4_frame_index(video)
    assert clips.is_clean(index) and index.n_samples == 100
    for k in (0, 37, 99):
        Image.fromarray(extract_frame_by_index(video, k, index=index)).save(tmp_path / "f.png")
        assert decode_image(tmp_path / "f.png") == FIRST + k


def test_neighbours_at_clip_edges(source, tmp_path):
    out = tmp_path / "clean"
    clean(source, out)
    folder = out / "labeled-data" / CLIP
    # k-2..k+2 inside the clip only: none before frame 0 or after frame 99.
    expected = {0, 1, 2, 3, 48, 49, 50, 51, 52, 97, 98, 99}
    written = {int(p.stem[3:]): p for p in folder.glob("img*.png")}
    assert set(written) == expected
    assert all(len(p.stem) == 6 for p in written.values())  # zero-padded as the labeled names
    for k, p in written.items():
        assert decode_image(p) == FIRST + k, p.name
    assert list(pd.read_csv(out / "checks.csv").neighbours_written) == [2, 1, 4, 2]


def test_mismatched_png_fails(source, tmp_path, stream_copied):
    # Frame 50's label image is really frame 51: off by one, as a stream-copy misread gives.
    frame_png(stream_copied, 51, source / f"labeled-data/{CLIP}/img050.png")
    with pytest.raises(RuntimeError, match="img050.png: differs from frame 50"):
        clean(source, tmp_path / "clean")
    assert not (tmp_path / "clean_checks" / f"{CLIP}.csv").exists()


def test_resume_skips_finished_clips(source, tmp_path, stream_copied, monkeypatch):
    # A second clip, only cleaned on the second run.
    other = "sess_clip_002"
    (source / "videos" / f"{other}.mp4").write_bytes(stream_copied.read_bytes())
    frame_png(stream_copied, 10, source / f"labeled-data/{other}/img010.png")
    images = [f"labeled-data/{CLIP}/img{k:03d}.png" for k in LABELED] + [f"labeled-data/{other}/img010.png"]
    write_labels(source / "CollectedData.csv", images)

    out = tmp_path / "clean"
    clean(source, out, "--clips", "1")
    assert (tmp_path / "clean_checks" / f"{CLIP}.csv").is_file()
    assert not (out / "videos" / f"{other}.mp4").exists()
    first = (out / "videos" / f"{CLIP}.mp4").stat().st_mtime_ns

    # The second run transcodes only the new clip (lossless + clean).
    calls = []
    orig = clips.transcode
    monkeypatch.setattr(clips, "transcode", lambda src, dst, codec: calls.append(dst) or orig(src, dst, codec))
    clean(source, out)
    assert (out / "videos" / f"{CLIP}.mp4").stat().st_mtime_ns == first
    assert [p.name for p in calls] == ["lossless.mp4", f"{other}.mp4"]
    assert len(pd.read_csv(out / "checks.csv")) == 5


# --- cut-clips ---------------------------------------------------------------


@pytest.fixture(scope="module")
def session_video(tmp_path_factory):
    return make_video(tmp_path_factory.mktemp("sess") / "session.mp4", 3000)


def cut(tmp_path, video, *extra):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([{"session": "mouse1/day1", "video": str(video)}]).to_csv(manifest, index=False)
    labels = write_labels(tmp_path / "labels.csv", ["labeled-data/mouse1_day1/img00001000.png"])
    main(["cut-clips", "--manifest", str(manifest), "--out", str(tmp_path / "videos"), "-n", "3",
          "--labels", str(labels), "--clip-frames", "40", "--span", "0", "1", "--min-gap", "300",
          "--name-suffix", "_Cam", *extra])  # fmt: skip
    return pd.read_csv(tmp_path / "videos_clips.csv")


def test_cut_clips_are_clean_and_frame_exact(tmp_path, session_video):
    table = cut(tmp_path, session_video)
    assert len(table) == 3 and set(table.status) == {"cut"} and not table.renormalized.any()
    starts = sorted(table.start_frame)
    assert all(abs(s - 1000) > 300 for s in starts) and min(np.diff(starts)) > 300
    for row in table.itertuples():
        assert row.clip_path.endswith(f"mouse1_day1_Cam_f{row.start_frame:07d}.mp4")
        index = read_mp4_frame_index(row.clip_path)
        assert clips.is_clean(index) and index.n_samples == 40
        for k in (0, 1, 39):
            Image.fromarray(extract_frame_by_index(row.clip_path, k, index=index)).save(tmp_path / "f.png")
            assert decode_image(tmp_path / "f.png") == row.start_frame + k
        info = json.loads((tmp_path / "videos" / f"{row.clip_path.rsplit('/', 1)[1][:-4]}.json").read_text())
        assert info["start_frame"] == row.start_frame and info["n_frames"] == 40


def test_cut_clips_resume_and_seed(tmp_path, session_video):
    first = cut(tmp_path, session_video)
    again = cut(tmp_path, session_video)
    assert set(again.status) == {"exists"}
    assert list(again.start_frame) == list(first.start_frame)


def test_cut_that_is_not_clean_is_renormalized(tmp_path, session_video, monkeypatch):
    real = clips.is_clean
    seen = []

    def first_cut_unclean(index):
        seen.append(index)
        return False if len(seen) == 1 else real(index)

    monkeypatch.setattr(clips, "is_clean", first_cut_unclean)
    checks = clips.cut_clip(str(session_video), 500, 40, tmp_path / "c.mp4")
    assert checks == {"clip_frames": 40, "renormalized": True}
    index = read_mp4_frame_index(tmp_path / "c.mp4")
    assert real(index) and index.n_samples == 40
    Image.fromarray(extract_frame_by_index(tmp_path / "c.mp4", 7, index=index)).save(tmp_path / "f.png")
    assert decode_image(tmp_path / "f.png") == 507


def test_cut_refuses_an_unsafe_source(tmp_path, stream_copied):
    with pytest.raises(ValueError, match="not safe"):
        clips.cut_clip(str(stream_copied), 0, 10, tmp_path / "c.mp4")


def test_clip_starts_gives_up():
    with pytest.raises(ValueError, match="found"):
        clips.clip_starts(1000, 5, 50, (0, 1), 400, [], np.random.default_rng(0), tries=1000)
