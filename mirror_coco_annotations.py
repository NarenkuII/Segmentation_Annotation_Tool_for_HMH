import argparse
import json
from pathlib import Path


def mirror_x(x: float, width: float) -> float:
    return round(float(width) - float(x), 3)


def mirror_keypoints(keypoints: list[float], width: float) -> list[float]:
    mirrored: list[float] = []
    for index in range(0, len(keypoints), 3):
        x = keypoints[index]
        y = keypoints[index + 1]
        v = keypoints[index + 2]
        mirrored.extend([mirror_x(x, width), y, v])
    return mirrored


def bbox_from_keypoints(keypoints: list[float], width: float, height: float) -> list[float]:
    xs = []
    ys = []
    for index in range(0, len(keypoints), 3):
        x = min(max(float(keypoints[index]), 0.0), float(width))
        y = min(max(float(keypoints[index + 1]), 0.0), float(height))
        xs.append(x)
        ys.append(y)
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


def process_json(path: Path) -> tuple[bool, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    info = data.get("info", {})
    width = info.get("width")
    height = info.get("height")
    if not width or not height:
        return False, 0

    annotations = data.get("annotations", [])
    changed = 0
    for annotation in annotations:
        keypoints = annotation.get("keypoints")
        if not keypoints:
            continue
        mirrored_keypoints = mirror_keypoints(keypoints, width)
        annotation["keypoints"] = mirrored_keypoints
        annotation["bbox"] = bbox_from_keypoints(mirrored_keypoints, width, height)
        annotation["area"] = round(annotation["bbox"][2] * annotation["bbox"][3], 3)
        changed += 1

    if not changed:
        return False, 0

    info["mirrored_geometry"] = True
    info["mirrored_geometry_annotations"] = changed
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, changed


def process_dir(directory: Path) -> tuple[int, int]:
    files_changed = 0
    annotations_changed = 0
    for path in sorted(directory.glob("*.json")):
        if path.name.lower() == "manifest.json":
            continue
        changed, count = process_json(path)
        if changed:
            files_changed += 1
            annotations_changed += count
    return files_changed, annotations_changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror COCO hand keypoint annotations horizontally.")
    parser.add_argument("directories", nargs="+", help="One or more *_coco_keypoints directories")
    args = parser.parse_args()

    total_files = 0
    total_annotations = 0
    for raw_dir in args.directories:
        directory = Path(raw_dir)
        if not directory.is_dir():
            print(f"[skip] {directory} is not a directory")
            continue
        files_changed, annotations_changed = process_dir(directory)
        total_files += files_changed
        total_annotations += annotations_changed
        print(f"{directory}: {files_changed} files changed, {annotations_changed} annotations mirrored")

    print(f"total: {total_files} files changed, {total_annotations} annotations mirrored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
