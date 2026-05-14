from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import cv2
from flask import Flask, jsonify, request, send_from_directory

from extract_clip_keypoints import (
    assign_hand_tracks,
    bbox_from_keypoints,
    clip_to_coco,
    detect_frame,
    frame_timestamp_ms,
    iter_videos,
    make_landmarker,
    normalize_handedness,
    safe_stem,
)


ROOT = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".avi", ".mkv"}
HAND_MODEL = ROOT / "hand_landmarker.task"
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")


def ensure_hand_model() -> Path:
    if not HAND_MODEL.exists():
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL)
    return HAND_MODEL


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def default_label(index: int) -> str:
    return f"Sign_{index + 1:02d}"


def resolve_video(name: str) -> Path:
    path = (ROOT / name).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise ValueError("Video must be inside the workspace.")
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        raise ValueError("Unsupported video extension.")
    if not path.exists():
        raise ValueError("Video not found.")
    return path


def resolve_workspace_dir(name: str) -> Path:
    path = (ROOT / name).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise ValueError("Folder must be inside the workspace.")
    if not path.exists() or not path.is_dir():
        raise ValueError("Folder not found.")
    return path


def make_segment(index: int, start: float, end: float, duration: float, padding_before: float, padding_after: float) -> dict:
    return {
        "label": default_label(index),
        "activeStart": round(start, 3),
        "activeEnd": round(end, 3),
        "start": round(clamp(start - padding_before, 0, duration), 3),
        "end": round(clamp(end + padding_after, 0, duration), 3),
    }


def scan_video(
    video_path: Path,
    threshold_percent: float,
    tracked_hand: str,
    handedness_mode: str,
    min_landmarks: int,
    min_duration: float,
    rest_gap: float,
    padding_before: float,
    padding_after: float,
    sample_fps: float,
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the video.")

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0
    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = raw_frame_count if raw_frame_count > 0 else 0
    duration = frame_count / fps if frame_count else 0.0
    step_frames = max(1, int(round(fps / max(1.0, sample_fps))))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    threshold_y_px = height * threshold_percent / 100.0

    segments: list[dict] = []
    sampled_frames: list[dict] = []
    detections_by_frame: list[list[dict]] = []

    try:
        ensure_hand_model()
        hands_model = make_landmarker("video")
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % step_frames != 0:
                frame_index += 1
                continue

            time_s = frame_index / fps
            result = detect_frame(hands_model, frame, frame_timestamp_ms(frame_index, fps))

            image_id = len(sampled_frames) + 1
            sampled_frames.append({"image_id": image_id, "time": time_s})
            frame_detections = []
            if result.hand_landmarks:
                for hand_index, landmarks in enumerate(result.hand_landmarks):
                    raw_label = ""
                    score = None
                    if result.handedness and hand_index < len(result.handedness):
                        raw_label = result.handedness[hand_index][0].category_name
                        score = result.handedness[hand_index][0].score
                    label = normalize_handedness(raw_label, handedness_mode)
                    xy_points = [(point.x * width, point.y * height) for point in landmarks]
                    keypoints = []
                    for x, y in xy_points:
                        keypoints.extend([round(x, 3), round(y, 3), 2])
                    frame_detections.append(
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "bbox": bbox_from_keypoints(xy_points, width, height),
                            "area": 0.0,
                            "iscrowd": 0,
                            "num_keypoints": 21,
                            "keypoints": keypoints,
                            "handedness": label,
                            "raw_handedness": raw_label,
                            "score": None if score is None else round(float(score), 6),
                        }
                    )
            detections_by_frame.append(frame_detections)

            frame_index += 1

    finally:
        if "hands_model" in locals():
            hands_model.close()
        cap.release()

    if frame_count <= 0 and frame_index > 0:
        frame_count = frame_index
        duration = frame_count / fps

    tracked_annotations, track_switches = assign_hand_tracks(
        detections_by_frame,
        width,
        height,
        max_missed_frames=max(1, int(round(rest_gap * sample_fps))),
    )
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in tracked_annotations:
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    open_start: float | None = None
    open_end: float | None = None
    last_active = 0.0
    for sampled in sampled_frames:
        time_s = sampled["time"]
        above = 0
        for annotation in annotations_by_image.get(sampled["image_id"], []):
            if tracked_hand != "Both" and annotation.get("handedness") != tracked_hand:
                continue
            keypoints = annotation["keypoints"]
            above += sum(1 for index in range(1, len(keypoints), 3) if keypoints[index] < threshold_y_px)

        active = above >= min_landmarks
        if active:
            last_active = time_s
            if open_start is None:
                open_start = time_s
            open_end = time_s
        elif open_start is not None and time_s - last_active >= rest_gap:
            active_end = open_end if open_end is not None else last_active
            if active_end - open_start >= min_duration:
                segments.append(make_segment(len(segments), open_start, active_end, duration, padding_before, padding_after))
            open_start = None
            open_end = None

    if open_start is not None:
        active_end = open_end if open_end is not None else last_active
        if active_end - open_start >= min_duration:
            segments.append(make_segment(len(segments), open_start, active_end, duration, padding_before, padding_after))

    return {
        "video": video_path.name,
        "fps": fps,
        "duration": duration,
        "sampleFps": sample_fps,
        "trackSwitches": track_switches,
        "segments": segments,
    }


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get("/api/videos")
def videos():
    files = sorted(
        path.name
        for path in ROOT.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return jsonify({"videos": files})


@app.get("/api/clip-dirs")
def clip_dirs():
    folders = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_dir() or path.name.startswith("__"):
            continue
        has_video = any(child.is_file() and child.suffix.lower() in VIDEO_EXTENSIONS for child in path.iterdir())
        if has_video:
            folders.append(path.name)
    return jsonify({"folders": folders})


@app.post("/api/scan")
def scan():
    data = request.get_json(force=True)
    try:
        video_path = resolve_video(data.get("video", ""))
        handedness_mode = data.get("handednessMode", "mediapipe")
        if handedness_mode not in {"mediapipe", "swap"}:
            raise ValueError("Invalid handedness mode.")
        tracked_hand = data.get("trackedHand", "Right")
        if tracked_hand not in {"Right", "Left", "Both"}:
            raise ValueError("Invalid hand filter.")
        result = scan_video(
            video_path=video_path,
            threshold_percent=clamp(float(data.get("thresholdPercent", 62)), 5, 95),
            tracked_hand=tracked_hand,
            handedness_mode=handedness_mode,
            min_landmarks=max(1, int(data.get("minLandmarks", 2))),
            min_duration=max(0.05, float(data.get("minDuration", 0.2))),
            rest_gap=max(0.05, float(data.get("restGap", 0.25))),
            padding_before=max(0.0, float(data.get("paddingBefore", 0.2))),
            padding_after=max(0.0, float(data.get("paddingAfter", 0.3))),
            sample_fps=clamp(float(data.get("sampleFps", 15)), 3, 30),
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400
    return app.response_class(json.dumps(result), mimetype="application/json")


@app.post("/api/extract-keypoints")
def extract_keypoints():
    data = request.get_json(force=True)
    try:
        input_dir = resolve_workspace_dir(data.get("inputDir", ""))
        output_name = data.get("outputDir") or f"{input_dir.name}_coco_keypoints"
        output_dir = (ROOT / output_name).resolve()
        if ROOT not in output_dir.parents and output_dir != ROOT:
            raise ValueError("Output folder must stay inside the workspace.")

        sample_fps_raw = data.get("sampleFps", 10)
        sample_fps = None if sample_fps_raw in ("", None, "all") else max(0.1, float(sample_fps_raw))
        hand = data.get("hand", "Both")
        if hand not in {"Right", "Left", "Both"}:
            raise ValueError("Invalid hand filter.")
        handedness_mode = data.get("handednessMode", "mediapipe")
        if handedness_mode not in {"mediapipe", "swap"}:
            raise ValueError("Invalid handedness mode.")
        interpolate_gap = max(0, int(data.get("interpolateGap", 2)))
        max_jump_px = max(0.0, float(data.get("maxJumpPx", 120)))
        recursive = bool(data.get("recursive", False))

        videos_found = iter_videos(input_dir, recursive)
        if not videos_found:
            raise ValueError("No clip videos found in folder.")

        ensure_hand_model()
        summaries = []
        for video_path in videos_found:
            relative = video_path.relative_to(input_dir)
            output_path = output_dir / relative.parent / f"{safe_stem(relative.name)}.json"
            with make_landmarker("video") as landmarker:
                summaries.append(
                    clip_to_coco(
                        video_path,
                        output_path,
                        landmarker,
                        sample_fps,
                        hand,
                        interpolate_gap,
                        max_jump_px,
                        handedness_mode,
                    )
                )

        manifest = {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "videos": summaries,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(
        {
            "outputDir": output_dir.name,
            "videos": len(summaries),
            "annotations": sum(item["annotations"] for item in summaries),
            "interpolated": sum(item["interpolated"] for item in summaries),
            "smoothed": sum(item["smoothed"] for item in summaries),
            "trackSwitches": sum(item["track_switches"] for item in summaries),
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
