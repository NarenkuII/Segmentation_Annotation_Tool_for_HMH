import argparse
import json
from pathlib import Path


def swap_handedness(value: str) -> str:
    if value == "Left":
        return "Right"
    if value == "Right":
        return "Left"
    return value


def process_json(path: Path) -> tuple[bool, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = data.get("annotations", [])
    swapped = 0

    for annotation in annotations:
        handedness = annotation.get("handedness")
        new_value = swap_handedness(handedness)
        if new_value != handedness:
            annotation["handedness"] = new_value
            swapped += 1

    if not swapped:
        return False, 0

    info = data.setdefault("info", {})
    info["handedness_swapped"] = True
    info["handedness_swapped_annotations"] = swapped

    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return True, swapped


def process_dir(directory: Path) -> tuple[int, int]:
    files_changed = 0
    annotations_swapped = 0

    for path in sorted(directory.glob("*.json")):
        if path.name.lower() == "manifest.json":
            continue
        changed, swapped = process_json(path)
        if changed:
            files_changed += 1
            annotations_swapped += swapped

    return files_changed, annotations_swapped


def main() -> int:
    parser = argparse.ArgumentParser(description="Swap COCO handedness values in JSON clip exports.")
    parser.add_argument("directories", nargs="+", help="One or more *_coco_keypoints directories")
    args = parser.parse_args()

    grand_total_files = 0
    grand_total_annotations = 0

    for raw_dir in args.directories:
        directory = Path(raw_dir)
        if not directory.is_dir():
            print(f"[skip] {directory} is not a directory")
            continue

        files_changed, annotations_swapped = process_dir(directory)
        grand_total_files += files_changed
        grand_total_annotations += annotations_swapped
        print(f"{directory}: {files_changed} files changed, {annotations_swapped} annotations swapped")

    print(f"total: {grand_total_files} files changed, {grand_total_annotations} annotations swapped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
