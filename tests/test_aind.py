from lp_relabel import aind

CONFIG = b"""behavior_data_asset_name: behavior_751181_2025-02-27_11-24-44
behavior_subpath: BottomCamera/video.mp4
lightning_pose_inference:
  model_name: s3://aind-scratch-data/x/ephys_BottomCamera_Pylon2Test2
"""


def test_asset_info_builds_urls(monkeypatch):
    monkeypatch.setattr(aind, "_get", lambda url: CONFIG)
    monkeypatch.setattr(aind.pd, "read_csv", lambda url: (_ for _ in ()).throw(OSError()))
    row = aind._asset_info("behavior_751181_2025-02-27_11-24-44_videoprocessed_2025-10-29_21-59-06/")
    assert row["mouse"] == "751181" and row["camera"] == "BottomCamera"
    assert row["video"].endswith("/behavior_751181_2025-02-27_11-24-44/behavior-videos/BottomCamera/video.mp4")
    assert row["pose"].endswith("_videoprocessed_2025-10-29_21-59-06/" + aind.PREDICTIONS)


def test_older_layout_camera_name(monkeypatch):
    monkeypatch.setattr(aind, "_get", lambda url: CONFIG.replace(b"BottomCamera/video.mp4", b"bottom_camera.mp4"))
    monkeypatch.setattr(aind.pd, "read_csv", lambda url: (_ for _ in ()).throw(OSError()))
    row = aind._asset_info("behavior_784803_2025-07-01_13-58-23_videoprocessed_2025-11-02_10-00-00/")
    assert row["camera"] == "bottom_camera"


def test_non_asset_is_skipped():
    assert aind._asset_info("behavior_751181_2025-02-27_11-24-44/") is None
