# lp-relabel

Pick the frames a Lightning Pose model is unsure about, cut them out of the
full-session video, and queue them in a Lightning Pose App project with the
model's own keypoints already placed. You drag the wrong points, save, merge the
new labels into the training set, and fine-tune.

```
predictions CSV + video  ──select──▶  LP App project (frames + queue)  ──label in app──▶
CollectedData.csv  ──merge──▶  training CSV  ──train / predict on Code Ocean──▶  new predictions
```

It never downloads a whole video. Frames are read from a local file or straight
from S3 over HTTPS, a few hundred KB per pick.

## Install

```bash
uv venv --python 3.12 && uv pip install -e '.[dev]'   # needs ffmpeg >= 5.1 on PATH
```

The labeling app goes in its own environment (see "Lightning Pose App on a Mac" below).

## One round

```bash
# 1. Sessions with predictions from your model (AIND public S3, ~10 s).
lp-relabel aind-sessions --model Pylon2Test2 --per-mouse 1 -o out/r02_manifest.csv

# 2. Pick, extract and queue frames into a new App project, one folder per round.
lp-relabel select --manifest out/r02_manifest.csv \
    --data-dir ~/LPProjects/pylon2test2_r02/data --model-name Pylon2Test2 -n 10

# 3. Register the project (~/.lightning-pose/projects.toml), then label in the App.
#    [pylon2test2_r02]
#    data_dir = "/Users/you/LPProjects/pylon2test2_r02/data"
#    model_dir = "/Users/you/LPProjects/pylon2test2_r02/models"

# 4. Add the saved labels to the training CSV; checks every image and its
#    context frames exist under --data-dir.
lp-relabel merge CollectedData.masked.csv ~/LPProjects/pylon2test2_r02/data/CollectedData.csv \
    -o CollectedData.merged.csv --data-dir <training data dir with labeled-data copied in>
```

Then copy the round's `labeled-data/<session>/` folders next to the training
`labeled-data/`, train on the merged CSV (set `model.checkpoint` to the last
best `.ckpt` to fine-tune instead of starting over), and rerun video inference
so the next round's predictions come from the new model.

### Inputs

`select` takes one session (`--session`, `--pose`, `--video`) or a manifest CSV
with columns `session, pose, video` and optionally `windows`.

- **pose**: a Lightning Pose predictions CSV (header rows scorer / bodyparts /
  coords; `x`, `y`, `likelihood` per keypoint; index = frame number from 0).
  Local path or http(s) URL. URLs are downloaded once into `~/.cache/lp-relabel`.
- **video**: local MP4, http(s) URL, or `s3://` URI (needs `pip install '.[s3]'`
  and AWS credentials). Must have exactly as many frames as the CSV has rows,
  or `select` stops.
- **windows** (optional): a CSV limiting where frames may be picked, either
  `start_frame,end_frame` rows (end exclusive) or a `frame` column of events
  plus `--pre`/`--post`. Use it to label only around licks, say.
- **session**: becomes the `labeled-data/<session>/` folder name.

### Outputs (in `--data-dir`)

| File | What it is |
|---|---|
| `project.yaml` | single-view App project with the predictions' keypoint names |
| `CollectedData.csv` | empty label file; the App writes saved labels here |
| `CollectedData.unlabeled.jsonl` | the App's labeling queue, with prefilled keypoints |
| `labeled-data/<session>/img########.jpg` | each picked frame and 2 neighbours each side, named by source frame index |
| `CollectedData.selection.tsv` | why each frame was picked (scores, reason, model) |

Rerunning `select` on the same project adds frames and never picks a frame
within `--min-gap` of one already queued or labeled.

## How frames are picked

Per session, every frame gets two scores from the model's own output:

- **likelihood**: `uncertain` (default) scores likelihoods near 0.5 highest.
  Tongue and paw points are absent in most frames and the model correctly says
  ~0 there, so `low` (1 - likelihood) would pick closed-mouth frames over and over.
- **jump**: how far a keypoint is from the midpoint of its neighbours in the
  previous and next frame, relative to that keypoint's median. At 500 Hz real
  motion between frames is tiny, so a one-frame spike is almost always an error.
  Only counted where all three frames have likelihood >= 0.5.

Both are turned into percentile ranks and averaged (`--likelihood-weight`,
`--jump-weight`). The top-scoring frames are taken greedily, at least
`--min-gap` frames apart (default 250, half a second at 500 Hz), and
`--random-fraction` of picks (default 0.2) are random, so each round also covers
ordinary frames. `--keypoints` limits scoring to some keypoints; the queue
always carries all of them.

Prefilled keypoints below `--prefill-min-likelihood` (default 0.5) are left
unplaced, so you place them only if they are really visible.

## Frame accuracy

Each JPEG is located by its presentation time from the MP4 sample tables
(`aind-video-utils`), never by frame rate or OpenCV seeking, so
`img00123456.jpg` is row 123456 of the predictions. `tests/test_frames.py`
checks this on an H.264 video with B-frames whose pixels encode their own frame
number, including frames right after keyframes.

## Lightning Pose App on a Mac

Labeling and viewing work without a GPU. Training and inference do not.

```bash
uv venv ~/.venvs/lp-app --python 3.12
VIRTUAL_ENV=~/.venvs/lp-app uv pip install --no-deps lightning-pose
# Intel Macs: the newest llvmlite has no wheel; pin an older pair.
printf 'llvmlite<0.45\nnumba<0.62\n' > /tmp/lpapp-constraints.txt
VIRTUAL_ENV=~/.venvs/lp-app uv pip install lightning-pose-app omegaconf \
    --only-binary llvmlite,numba,jaxlib -c /tmp/lpapp-constraints.txt
DO_NOT_TRACK=1 ~/.venvs/lp-app/bin/litpose run_app    # http://localhost:8080
```

## Known limits

- **One label file name.** LP App 2.4.2.3 accepts only `CollectedData.csv` (or
  `CollectedData_test.csv`) in a single-view project, hence one project folder
  per round. Newer App code relaxes this.
- **Single view only.** Multiview projects need one queue per view, aligned.
- **Old training frames are not excluded.** Existing labels come from short
  clips named by clip, not by session frame, so `select` cannot tell when a
  pick duplicates one. Picks are rare enough in a 1.8M-frame session that this
  barely matters.
- **AIND finder is AIND-specific.** `aind-sessions` reads the
  `*_videoprocessed_*` assets in `aind-open-data`. Everything else works on any
  LP predictions CSV and MP4.
