# Sign Language Segmentation & Annotation Tool

A lightweight web application for segmenting continuous French Sign Language (LSF) videos into signed clips and exporting stabilized MediaPipe hand keypoints in COCO-style JSON format.

## Annotation Workflow

```text
Video Upload -> Automatic Motion Scan -> Manual Boundary Correction -> Clip Splitting -> Stabilized Keypoint Export
```

## Features

- **Temporal Tracking & Smoothing**: Uses MediaPipe in `VIDEO` mode with temporal `track_id` association to minimize hand label swapping (left vs. right).
- **Interactive Web Interface**: Rapid verification, trimming, and splitting of long sign videos.
- **Batch Keypoint Extraction**: Exports tracked landmarks with configurable interpolation and confidence filtering.
- **Self-Hostable**: Deployable behind a reverse proxy (e.g., Nginx + Basic Authentication).

## Installation & Running

```bash
git clone https://github.com/NarenkuII/Segmentation_Annotation_Tool_for_HMH.git
cd Segmentation_Annotation_Tool_for_HMH
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Access the web interface at `http://127.0.0.1:8000`.

## Hand Tracking & Stabilization

MediaPipe runs in video stream mode, leveraging sequential frame timestamps. The JSON export pipeline stabilizes hand detection through a temporal tracking mechanism (`track_id`). Filtering and linear interpolation operate on active tracks rather than single-frame labels, preventing rapid left/right classification jitter.

- **Dominant hand normalization**: Select `Tracked hand -> Right only` and `Target side -> Left side` to target signer perspective.
- **Camera parity correction**: Use `Hand labels -> Swap Left/Right` when working with mirrored or non-mirrored recordings.

## CLI Usage

Extract keypoints from existing video clips directly via CLI:

```bash
python extract_clip_keypoints.py ./clips_folder --hand Right --side left --sample-fps 10 --interpolate-gap 2 --max-jump-px 120
```

With swapped left/right handedness labels:

```bash
python extract_clip_keypoints.py ./clips_folder --hand Both --handedness-mode swap
```

## Reverse Proxy Deployment

To expose the tool remotely, place the Flask instance behind Nginx with HTTP Basic Authentication while binding Flask to `127.0.0.1:8000`.
