from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import cv2
import mediapipe as mp


ROOT = Path(__file__).resolve().parent
HAND_MODEL = ROOT / "hand_landmarker.task"
VIDEO_EXTENSIONS = {".mp4", ".webm", ".mov", ".avi", ".mkv"}
HAND_LABELS = {"Left", "Right"}
SIDE_FILTERS = {"any", "left", "right"}

HAND_KEYPOINTS = [
    "wrist",
    "thumb_cmc",
    "thumb_mcp",
    "thumb_ip",
    "thumb_tip",
    "index_finger_mcp",
    "index_finger_pip",
    "index_finger_dip",
    "index_finger_tip",
    "middle_finger_mcp",
    "middle_finger_pip",
    "middle_finger_dip",
    "middle_finger_tip",
    "ring_finger_mcp",
    "ring_finger_pip",
    "ring_finger_dip",
    "ring_finger_tip",
    "pinky_mcp",
    "pinky_pip",
    "pinky_dip",
    "pinky_tip",
]

HAND_SKELETON = [
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [1, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [1, 10],
    [10, 11],
    [11, 12],
    [12, 13],
    [1, 14],
    [14, 15],
    [15, 16],
    [16, 17],
    [1, 18],
    [18, 19],
    [19, 20],
    [20, 21],
    [6, 10],
    [10, 14],
    [14, 18],
]


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", stem)
    stem = re.sub(r"\s+", "_", stem).strip("._")
    return stem or "clip"


def iter_videos(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def bbox_from_keypoints(points: list[tuple[float, float]], width: int, height: int) -> list[float]:
    xs = [min(max(x, 0.0), float(width)) for x, _ in points]
    ys = [min(max(y, 0.0), float(height)) for _, y in points]
    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)
    return [
        round(x_min, 3),
        round(y_min, 3),
        round(max(0.0, x_max - x_min), 3),
        round(max(0.0, y_max - y_min), 3),
    ]


def keypoints_to_xy(keypoints: list[float]) -> list[tuple[float, float]]:
    return [(keypoints[index], keypoints[index + 1]) for index in range(0, len(keypoints), 3)]


def keypoints_screen_side(keypoints: list[float], width: int, mirror: bool = False) -> str:
    if not keypoints or width <= 0:
        return "unknown"
    xs = [keypoints[index] for index in range(0, len(keypoints), 3)]
    center_x = sum(xs) / len(xs)
    if mirror:
        center_x = width - center_x
    return "left" if center_x < width / 2 else "right"


def annotation_matches_filters(
    annotation: dict,
    hand_filter: str,
    side_filter: str,
    width: int,
    mirror: bool = False,
) -> bool:
    if hand_filter != "Both" and annotation.get("handedness") != hand_filter:
        return False
    if side_filter != "any" and keypoints_screen_side(annotation.get("keypoints", []), width, mirror) != side_filter:
        return False
    return True


def swap_handedness(label: str) -> str:
    if label == "Left":
        return "Right"
    if label == "Right":
        return "Left"
    return label


def normalize_handedness(label: str, mode: str) -> str:
    if mode == "swap":
        return swap_handedness(label)
    return label


def mean_keypoint_distance(before: list[float], after: list[float]) -> float:
    total = 0.0
    count = 0
    for index in range(0, min(len(before), len(after)), 3):
        total += math.hypot(before[index] - after[index], before[index + 1] - after[index + 1])
        count += 1
    return total / count if count else float("inf")


def weighted_hand_vote(score: float | None) -> float:
    if score is None:
        return 0.5
    return max(0.01, min(1.0, float(score)))


def dominant_hand_label(track: dict) -> str:
    votes = track.get("label_votes") or {}
    if not votes:
        return track.get("last_handedness") or ""
    return max(votes.items(), key=lambda item: item[1])[0]


def assign_hand_tracks(
    detections_by_frame: list[list[dict]],
    width: int,
    height: int,
    max_missed_frames: int,
    max_track_distance_px: float | None = None,
) -> tuple[list[dict], int]:
    tracks: list[dict] = []
    next_track_id = 1
    track_switches = 0
    diagonal = math.hypot(width, height) if width and height else 1000.0
    max_distance = max_track_distance_px if max_track_distance_px else max(90.0, diagonal * 0.18)
    label_mismatch_penalty = max(35.0, max_distance * 0.35)

    def add_detection(track: dict, detection: dict) -> None:
        track["detections"].append(detection)
        track["last_keypoints"] = detection["keypoints"]
        track["last_image_id"] = detection["image_id"]
        handedness = detection.get("handedness") or ""
        if handedness in HAND_LABELS:
            track["label_votes"][handedness] = track["label_votes"].get(handedness, 0.0) + weighted_hand_vote(
                detection.get("score")
            )
            previous = track.get("last_handedness")
            if previous and previous != handedness:
                track["raw_switches"] = track.get("raw_switches", 0) + 1
            track["last_handedness"] = handedness

    for frame_detections in detections_by_frame:
        if not frame_detections:
            continue

        image_id = frame_detections[0]["image_id"]
        active_tracks = [
            track for track in tracks if image_id - track["last_image_id"] <= max(1, max_missed_frames + 1)
        ]
        costs = []
        for detection_index, detection in enumerate(frame_detections):
            for track in active_tracks:
                distance = mean_keypoint_distance(track["last_keypoints"], detection["keypoints"])
                expected_label = dominant_hand_label(track)
                observed_label = detection.get("handedness") or ""
                cost = distance
                if expected_label in HAND_LABELS and observed_label in HAND_LABELS and expected_label != observed_label:
                    cost += label_mismatch_penalty
                costs.append((cost, distance, detection_index, track))

        assigned_detection_indexes: set[int] = set()
        assigned_track_ids: set[int] = set()
        for cost, distance, detection_index, track in sorted(costs, key=lambda item: item[0]):
            if detection_index in assigned_detection_indexes or track["id"] in assigned_track_ids:
                continue
            if distance > max_distance:
                continue
            detection = frame_detections[detection_index]
            detection["track_id"] = track["id"]
            assigned_detection_indexes.add(detection_index)
            assigned_track_ids.add(track["id"])
            before_label = dominant_hand_label(track)
            add_detection(track, detection)
            after_label = dominant_hand_label(track)
            if before_label in HAND_LABELS and after_label in HAND_LABELS and before_label != after_label:
                track_switches += 1

        for detection_index, detection in enumerate(frame_detections):
            if detection_index in assigned_detection_indexes:
                continue
            track = {
                "id": next_track_id,
                "detections": [],
                "last_keypoints": detection["keypoints"],
                "last_image_id": detection["image_id"],
                "last_handedness": detection.get("handedness") or "",
                "label_votes": {},
                "raw_switches": 0,
            }
            next_track_id += 1
            detection["track_id"] = track["id"]
            add_detection(track, detection)
            tracks.append(track)

    annotations = []
    for track in tracks:
        stable_label = dominant_hand_label(track)
        for detection in track["detections"]:
            annotation = dict(detection)
            annotation["handedness"] = stable_label
            annotation["track_id"] = track["id"]
            annotation["raw_track_switches"] = track.get("raw_switches", 0)
            annotations.append(annotation)

    return annotations, track_switches


def interpolate_keypoints(before: list[float], after: list[float], ratio: float) -> list[float]:
    keypoints = []
    for index in range(0, len(before), 3):
        x = before[index] + (after[index] - before[index]) * ratio
        y = before[index + 1] + (after[index + 1] - before[index + 1]) * ratio
        keypoints.extend([round(x, 3), round(y, 3), 1])
    return keypoints


def interpolate_short_gaps(coco: dict, max_gap: int) -> int:
    if max_gap <= 0:
        return 0

    images_by_id = {image["id"]: image for image in coco["images"]}
    annotations_by_track: dict[tuple[str, str], list[dict]] = {}
    for annotation in coco["annotations"]:
        hand = annotation.get("handedness") or "Unknown"
        track = str(annotation.get("track_id") or hand)
        annotations_by_track.setdefault((hand, track), []).append(annotation)

    next_id = max((annotation["id"] for annotation in coco["annotations"]), default=0) + 1
    added = []
    for (hand, track), annotations in annotations_by_track.items():
        annotations.sort(key=lambda item: item["image_id"])
        for before, after in zip(annotations, annotations[1:]):
            gap = after["image_id"] - before["image_id"] - 1
            if gap <= 0 or gap > max_gap:
                continue

            for gap_index in range(1, gap + 1):
                image_id = before["image_id"] + gap_index
                image = images_by_id.get(image_id)
                if not image:
                    continue
                ratio = gap_index / (gap + 1)
                keypoints = interpolate_keypoints(before["keypoints"], after["keypoints"], ratio)
                xy_points = keypoints_to_xy(keypoints)
                bbox = bbox_from_keypoints(xy_points, image["width"], image["height"])
                added.append(
                    {
                        "id": next_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": round(bbox[2] * bbox[3], 3),
                        "iscrowd": 0,
                        "num_keypoints": len(HAND_KEYPOINTS),
                        "keypoints": keypoints,
                        "handedness": hand,
                        "track_id": before.get("track_id"),
                        "score": None,
                        "interpolated": True,
                    }
                )
                next_id += 1

    coco["annotations"].extend(added)
    coco["annotations"].sort(
        key=lambda item: (item["image_id"], item.get("track_id") or 0, item.get("handedness") or "", item["id"])
    )
    return len(added)


def point_distance(a: list[float], b: list[float], keypoint_index: int) -> float:
    offset = keypoint_index * 3
    return math.hypot(a[offset] - b[offset], a[offset + 1] - b[offset + 1])


def smooth_fast_jumps(coco: dict, max_jump_px: float) -> int:
    if max_jump_px <= 0:
        return 0

    images_by_id = {image["id"]: image for image in coco["images"]}
    annotations_by_track: dict[tuple[str, str], list[dict]] = {}
    for annotation in coco["annotations"]:
        hand = annotation.get("handedness") or "Unknown"
        track = str(annotation.get("track_id") or hand)
        annotations_by_track.setdefault((hand, track), []).append(annotation)

    smoothed_points = 0
    for annotations in annotations_by_track.values():
        annotations.sort(key=lambda item: item["image_id"])
        for before, current, after in zip(annotations, annotations[1:], annotations[2:]):
            if current["image_id"] - before["image_id"] != 1:
                continue
            if after["image_id"] - current["image_id"] != 1:
                continue

            changed = False
            changed_points = 0
            keypoints = list(current["keypoints"])
            for keypoint_index in range(len(HAND_KEYPOINTS)):
                prev_to_current = point_distance(before["keypoints"], current["keypoints"], keypoint_index)
                current_to_next = point_distance(current["keypoints"], after["keypoints"], keypoint_index)
                prev_to_next = point_distance(before["keypoints"], after["keypoints"], keypoint_index)

                # A one-frame spike usually jumps far away and immediately comes back.
                if prev_to_current <= max_jump_px or current_to_next <= max_jump_px:
                    continue
                if prev_to_next > max_jump_px:
                    continue

                offset = keypoint_index * 3
                keypoints[offset] = round((before["keypoints"][offset] + after["keypoints"][offset]) / 2, 3)
                keypoints[offset + 1] = round((before["keypoints"][offset + 1] + after["keypoints"][offset + 1]) / 2, 3)
                keypoints[offset + 2] = 1
                changed = True
                changed_points += 1
                smoothed_points += 1

            if changed:
                image = images_by_id[current["image_id"]]
                xy_points = keypoints_to_xy(keypoints)
                bbox = bbox_from_keypoints(xy_points, image["width"], image["height"])
                current["keypoints"] = keypoints
                current["bbox"] = bbox
                current["area"] = round(bbox[2] * bbox[3], 3)
                current["smoothed"] = True
                current["smoothed_keypoints"] = current.get("smoothed_keypoints", 0) + changed_points

    return smoothed_points


def make_landmarker(running_mode: str = "video"):
    if not HAND_MODEL.exists():
        raise FileNotFoundError(
            f"Missing {HAND_MODEL.name}. Run app.py once or keep hand_landmarker.task in this folder."
        )
    mode = running_mode.lower()
    if mode == "image":
        mp_running_mode = mp.tasks.vision.RunningMode.IMAGE
    elif mode == "video":
        mp_running_mode = mp.tasks.vision.RunningMode.VIDEO
    else:
        raise ValueError("running_mode must be 'image' or 'video'.")
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=mp_running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.55,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def frame_timestamp_ms(frame_index: int, fps: float) -> int:
    return int(round((frame_index / fps) * 1000.0)) if fps > 0 else frame_index


def detect_frame(landmarker, frame, timestamp_ms: int | None = None):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    if timestamp_ms is None:
        return landmarker.detect(image)
    return landmarker.detect_for_video(image, timestamp_ms)


def clip_to_coco(
    video_path: Path,
    output_path: Path,
    landmarker,
    sample_fps: float | None,
    hand_filter: str,
    interpolate_gap: int,
    max_jump_px: float,
    handedness_mode: str = "mediapipe",
    side_filter: str = "any",
) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps_reliable = 1.0 <= raw_fps <= 120.0
    fps = raw_fps if fps_reliable else 30.0
    raw_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_count = raw_frame_count if raw_frame_count > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration = frame_count / fps if frame_count else 0.0
    step_frames = 1
    if sample_fps and sample_fps > 0:
        step_frames = max(1, int(round(fps / sample_fps)))

    coco = {
        "info": {
            "description": "MediaPipe Hands keypoints exported in COCO-style JSON",
            "source_video": video_path.name,
            "fps": fps,
            "raw_fps": raw_fps,
            "fps_reliable": fps_reliable,
            "sample_fps": sample_fps or fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
            "interpolate_gap": interpolate_gap,
            "max_jump_px": max_jump_px,
            "handedness_mode": handedness_mode,
            "side_filter": side_filter,
            "mediapipe_running_mode": "VIDEO",
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": 1,
                "name": "hand",
                "supercategory": "person",
                "keypoints": HAND_KEYPOINTS,
                "skeleton": HAND_SKELETON,
            }
        ],
    }

    frame_index = 0
    image_id = 1
    detections_by_frame: list[list[dict]] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_index % step_frames != 0:
            frame_index += 1
            continue

        time_s = frame_index / fps
        coco["images"].append(
            {
                "id": image_id,
                "file_name": f"{video_path.name}#frame_{frame_index:06d}",
                "width": width,
                "height": height,
                "frame_index": frame_index,
                "time": round(time_s, 6),
            }
        )

        frame_detections = []
        result = detect_frame(landmarker, frame, frame_timestamp_ms(frame_index, fps))
        if result.hand_landmarks:
            for hand_index, landmarks in enumerate(result.hand_landmarks):
                raw_handedness = ""
                score = None
                if result.handedness and hand_index < len(result.handedness):
                    raw_handedness = result.handedness[hand_index][0].category_name
                    score = result.handedness[hand_index][0].score
                handedness = normalize_handedness(raw_handedness, handedness_mode)

                xy_points = [(point.x * width, point.y * height) for point in landmarks]
                keypoints = []
                for x, y in xy_points:
                    keypoints.extend([round(x, 3), round(y, 3), 2])
                bbox = bbox_from_keypoints(xy_points, width, height)
                frame_detections.append(
                    {
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": bbox,
                        "area": round(bbox[2] * bbox[3], 3),
                        "iscrowd": 0,
                        "num_keypoints": len(HAND_KEYPOINTS),
                        "keypoints": keypoints,
                        "handedness": handedness,
                        "raw_handedness": raw_handedness,
                        "score": None if score is None else round(float(score), 6),
                    }
                )
        detections_by_frame.append(frame_detections)

        image_id += 1
        frame_index += 1

    if frame_count <= 0 and frame_index > 0:
        frame_count = frame_index
        duration = frame_count / fps
        coco["info"]["frame_count"] = frame_count
        coco["info"]["duration"] = duration

    cap.release()
    tracked_annotations, track_switches = assign_hand_tracks(
        detections_by_frame,
        width,
        height,
        max_missed_frames=max(0, interpolate_gap),
    )
    annotation_id = 1
    for annotation in sorted(
        tracked_annotations,
        key=lambda item: (item["image_id"], item.get("track_id") or 0, item.get("handedness") or ""),
    ):
        if not annotation_matches_filters(annotation, hand_filter, side_filter, width):
            continue
        annotation["id"] = annotation_id
        coco["annotations"].append(annotation)
        annotation_id += 1

    interpolated_count = interpolate_short_gaps(coco, interpolate_gap)
    smoothed_count = smooth_fast_jumps(coco, max_jump_px)
    coco["info"]["interpolated_annotations"] = interpolated_count
    coco["info"]["smoothed_keypoints"] = smoothed_count
    coco["info"]["track_switches"] = track_switches
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    return {
        "video": str(video_path),
        "output": str(output_path),
        "frames": len(coco["images"]),
        "annotations": len(coco["annotations"]),
        "interpolated": interpolated_count,
        "smoothed": smoothed_count,
        "track_switches": track_switches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand keypoints from a folder of video clips into one COCO-style JSON per clip."
    )
    parser.add_argument("input_dir", type=Path, help="Folder containing clip videos.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output folder. Default: <input_dir>_coco_keypoints",
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="Optional frame sampling rate. Omit to process every frame.",
    )
    parser.add_argument(
        "--hand",
        choices=["Right", "Left", "Both"],
        default="Both",
        help="Which MediaPipe handedness to export.",
    )
    parser.add_argument("--recursive", action="store_true", help="Search videos recursively.")
    parser.add_argument(
        "--interpolate-gap",
        type=int,
        default=2,
        help="Fill missing detections for gaps up to this many sampled frames. Default: 2.",
    )
    parser.add_argument(
        "--max-jump-px",
        type=float,
        default=120.0,
        help="Smooth one-frame keypoint spikes above this pixel distance. Use 0 to disable. Default: 120.",
    )
    parser.add_argument(
        "--handedness-mode",
        choices=["mediapipe", "swap"],
        default="mediapipe",
        help="Use MediaPipe labels as-is, or swap Left/Right for non-mirrored camera videos. Default: mediapipe.",
    )
    parser.add_argument(
        "--side",
        choices=sorted(SIDE_FILTERS),
        default="any",
        help="Optional screen-side filter after tracking. Use left/right to keep only one side of the video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 2

    output_dir = args.output_dir.resolve() if args.output_dir else input_dir.with_name(f"{input_dir.name}_coco_keypoints")
    videos = iter_videos(input_dir, args.recursive)
    if not videos:
        print(f"No videos found in: {input_dir}", file=sys.stderr)
        return 1

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Videos: {len(videos)}")
    print(f"Hand filter: {args.hand}")
    print(f"Side filter: {args.side}")
    print(f"Handedness mode: {args.handedness_mode}")
    print(f"Sample FPS: {args.sample_fps or 'all frames'}")

    summaries = []
    for video_path in videos:
        relative = video_path.relative_to(input_dir)
        json_name = f"{safe_stem(relative.name)}.json"
        output_path = output_dir / relative.parent / json_name
        with make_landmarker("video") as landmarker:
            summary = clip_to_coco(
                video_path,
                output_path,
                landmarker,
                args.sample_fps,
                args.hand,
                max(0, args.interpolate_gap),
                max(0.0, args.max_jump_px),
                args.handedness_mode,
                args.side,
            )
            summaries.append(summary)
            print(
                f"OK {relative} -> {output_path.relative_to(output_dir)} "
                f"({summary['frames']} frames, {summary['annotations']} annotations, "
                f"{summary['interpolated']} interpolated, {summary['smoothed']} smoothed points, "
                f"{summary['track_switches']} track switches)"
            )

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "videos": summaries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
