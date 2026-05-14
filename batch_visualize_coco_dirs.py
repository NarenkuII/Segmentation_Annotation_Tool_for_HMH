from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


COLORS = {
    "Right": (0, 210, 255),
    "Left": (80, 255, 80),
    "Unknown": (220, 220, 220),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render skeleton preview videos for every COCO-style JSON in every *_coco_keypoints folder."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Workspace root to scan. Default: current directory.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_skeleton_videos",
        help="Suffix for output folders created next to each *_coco_keypoints folder.",
    )
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS.")
    parser.add_argument("--hand", choices=["Right", "Left", "Both"], default="Both")
    return parser.parse_args()


def annotation_points(annotation: dict) -> list[tuple[int, int, int]]:
    values = annotation["keypoints"]
    return [
        (int(round(values[index])), int(round(values[index + 1])), int(values[index + 2]))
        for index in range(0, len(values), 3)
    ]


def draw_annotation(frame, annotation: dict, skeleton: list[list[int]]) -> None:
    hand = annotation.get("handedness") or "Unknown"
    color = COLORS.get(hand, COLORS["Unknown"])
    line_color = (120, 120, 120) if annotation.get("interpolated") else color
    points = annotation_points(annotation)

    for start, end in skeleton:
        a = points[start - 1]
        b = points[end - 1]
        if a[2] > 0 and b[2] > 0:
            cv2.line(frame, (a[0], a[1]), (b[0], b[1]), line_color, 2, cv2.LINE_AA)

    for x, y, visible in points:
        if visible > 0:
            cv2.circle(frame, (x, y), 3, color, -1, cv2.LINE_AA)


def render_json(json_path: Path, output_path: Path, override_fps: float | None, hand_filter: str) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    info = data.get("info", {})
    width = int(info.get("width") or 1280)
    height = int(info.get("height") or 720)
    fps = float(override_fps or info.get("sample_fps") or info.get("fps") or 30)
    if fps <= 0 or fps > 120:
        fps = 30

    categories = {category["id"]: category for category in data.get("categories", [])}
    skeleton = categories.get(1, {}).get("skeleton", [])

    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in data.get("annotations", []):
        hand = annotation.get("handedness") or "Unknown"
        if hand_filter != "Both" and hand != hand_filter:
            continue
        annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    for image in sorted(data.get("images", []), key=lambda item: item["id"]):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for annotation in annotations_by_image.get(image["id"], []):
            draw_annotation(frame, annotation, skeleton)
        cv2.putText(
            frame,
            f"frame {image.get('frame_index', image['id'])}  t={image.get('time', 0):.2f}s",
            (18, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    return {"json": str(json_path), "video": str(output_path)}


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    coco_dirs = sorted(path for path in root.iterdir() if path.is_dir() and path.name.endswith("_coco_keypoints"))
    if not coco_dirs:
        print(f"No *_coco_keypoints folders found in {root}")
        return 1

    total = 0
    for coco_dir in coco_dirs:
        output_dir = coco_dir.with_name(f"{coco_dir.name}{args.output_suffix}")
        json_files = sorted(
            path for path in coco_dir.iterdir() if path.is_file() and path.suffix.lower() == ".json" and path.name != "manifest.json"
        )
        print(f"{coco_dir.name}: {len(json_files)} json file(s)")
        for json_path in json_files:
            output_path = output_dir / f"{json_path.stem}_skeleton.mp4"
            render_json(json_path, output_path, args.fps, args.hand)
            total += 1
        print(f"  -> {output_dir.name}")

    print(f"Rendered {total} skeleton video(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
