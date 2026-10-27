from conftest import decode_image

from lp_relabel.frames import context_runs, extract_with_context


def test_context_runs_merge_and_clip():
    assert context_runs([1, 4, 20], 2, 21) == [(0, 7), (18, 21)]


def test_every_written_jpeg_shows_its_own_frame(video, tmp_path):
    # Targets on and off keyframes (GOP 50), near the start and end, and close
    # enough together that their context runs merge.
    targets = [2, 49, 50, 51, 123, 125, 300, 597]
    out = extract_with_context(str(video), targets, tmp_path, context=2)
    assert sorted(out) == targets
    written = sorted(tmp_path.glob("img*.jpg"))
    expected = sorted({f for t in targets for f in range(t - 2, t + 3) if 0 <= f < 600})
    assert [int(p.stem[3:]) for p in written] == expected
    for p in written:
        assert decode_image(p) == int(p.stem[3:]), p.name


def test_existing_files_are_kept(video, tmp_path):
    extract_with_context(str(video), [100], tmp_path)
    first = (tmp_path / "img00000100.jpg").stat().st_mtime_ns
    extract_with_context(str(video), [100], tmp_path)
    assert (tmp_path / "img00000100.jpg").stat().st_mtime_ns == first
