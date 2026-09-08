"""Extract frames with a source-group prefix; refuse accidental overwrites."""
import argparse
import json
from pathlib import Path
import re

import cv2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("output_dir")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--group", help="Unique trip/video ID; default: video filename stem")
    a = p.parse_args()
    if a.every < 1:
        raise ValueError("--every must be >= 1")
    group = a.group or Path(a.video).stem
    if not group or not re.fullmatch(r"[\w.-]+", group) or group in (".", ".."):
        raise ValueError("--group may contain letters, digits, underscore, dot and hyphen")
    output = Path(a.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / f"{group}.frames.json"
    if metadata_path.exists() or list(output.glob(f"{group}_frame_*.bmp")):
        raise FileExistsError("This group was already extracted; choose a new output directory or group")
    capture = cv2.VideoCapture(a.video)
    if not capture.isOpened():
        capture.release()
        raise FileNotFoundError(a.video)
    index = saved = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % a.every == 0:
                path = output / f"{group}_frame_{index:06d}.bmp"
                if path.exists():
                    raise FileExistsError(path)
                if not cv2.imwrite(str(path), frame):
                    raise IOError(f"Cannot write {path}")
                saved += 1
            index += 1
    finally:
        capture.release()
    metadata_path.write_text(json.dumps({"group": group, "source": str(Path(a.video).resolve()),
                                         "every": a.every, "frames_decoded": index,
                                         "frames_saved": saved}, indent=2), encoding="utf-8")
    print(f"saved {saved} frames from {index}; group={group}")


if __name__ == "__main__":
    main()
