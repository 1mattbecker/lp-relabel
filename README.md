# lp-relabel

Pick the frames a Lightning Pose model is unsure about, cut them out of the
full-session video, and queue them in a Lightning Pose App project with the
model's own keypoints already placed. You drag the wrong points, save, merge the
new labels into the training set, and fine-tune.

It also prepares video data for training (`clean-clips`, `cut-clips`, see
"Video data prep for training"): clean clips, context frames and unlabeled clips
that Lightning Pose reads frame-exactly.

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

## Video data prep for training

Two commands turn labeled clips and session videos into a project that
Lightning Pose's context model and unlabeled (semi-supervised) losses read
correctly. Everything they write passes `read_mp4_frame_index(...)`:
frame-addressing-safe edit list and a uniform timeline ("clean"), so DALI and
frame-index addressing see exactly the displayed frames.

### `clean-clips`: clean clips and context frames for a labeled project

Clips cut by stream copy (as Pylon2Test2's were) index more samples than they
display (591-743 vs 502), behind an edit list, and some have timestamp gaps.
DALI and frame-index addressing misread them, so frame `k` of the file is not
`img<k>.png`. Per clip with labeled frames, `clean-clips`:

1. transcodes it losslessly with `transcode_video(normalize_cfr=True)` (a
   temporary file) and checks the result is clean;
2. checks every labeled PNG against `extract_frame_by_index` of that transcode
   and stops if the mean absolute difference exceeds `MAX_DIFF` (0.05 of 255;
   correct frames measure <= 0.004, a neighbouring frame >= 0.9);
3. writes the neighbours `k-2, k-1, k+1, k+2` as PNGs beside the labeled frame,
   named by their real frame index (none past either end of the clip; LP then
   uses the labeled frame itself);
4. writes `videos/<clip>.mp4`, a crf-18 `normalize_cfr` encode, for the
   unlabeled losses.

```bash
lp-relabel clean-clips --labels CollectedData.train.csv \
    --source https://<bucket>.s3.amazonaws.com/<prefix>/<project> --match _clip_ \
    --out ~/LPProjects/p2t2_clean_build/p2t2_clean
```

`--source` (folder, http(s) or `s3://` prefix) holds `labeled-data/<clip>/img<k>.png`
and `videos/<clip>.mp4` (or give `--videos`). `--out` gets `labeled-data/`,
`videos/` and `checks.csv` (per labeled frame: clip frames, source samples,
difference, neighbours written); upload it as is. Each finished clip leaves a
record in `<out>_checks/`; a rerun skips those clips. Downloads retry, and
`--workers` clips run at once. `--clips N` does the first N (a test).

**Context stride.** Only stride 1 is supported. LP 2.4.3 reads a labeled
frame's context as the files numbered `n-2..n+2` (`utils/io.get_context_img_paths`),
and its DALI video readers (unlabeled losses, video inference) read consecutive
frames with no stride option. Files at `k±5, k±10` named by real index would
never be read. Context spanning more time at 500 Hz needs either LP changes
or strided videos with frames renamed to their strided index (then not the
real frame index), with video inference run on the same strided sequence.

### `cut-clips`: unlabeled clips from session videos

```bash
lp-relabel cut-clips --manifest manifest.csv --total 120 --clip-frames 500 \
    --labels data/CollectedData.csv --min-gap 2500 --span 0.1 0.9 \
    --name-suffix _BottomCamera --out <project>/videos
```

Per session of the manifest (`session, video`; video local, URL or `s3://`),
`-n` clips (or `--total` split evenly), each starting within `--span` of the
video and more than `--min-gap` frames from every labeled frame of
`labeled-data/<session>/` in the `--labels` CSVs and from the other starts.
One seeded generator runs over the sessions in manifest order, so a rerun picks
the same clips. Clips are cut as aind-dynamic-foraging-behavior-video-analysis's
`cut_clip` does (seek by the frame's presentation time, exact frame count,
re-encode crf 18 with even timestamps), named `<session><suffix>_f<start:07d>.mp4`
with a JSON sidecar, and checked clean; a cut that is not clean is re-encoded
with `normalize_cfr`, and the command stops if that is not clean either. Rerunning
keeps finished clips. The clip table goes to `<out>_clips.csv` (`--table`).
The source video must itself be frame-addressing-safe.

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
