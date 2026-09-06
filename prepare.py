"""Extract evenly spaced BMP frames from a video for manual labeling."""
import argparse
from pathlib import Path
import cv2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("video"); p.add_argument("output_dir")
    p.add_argument("--every", type=int, default=10, help="save every Nth frame")
    args = p.parse_args()
    if args.every < 1: raise ValueError("--every must be >= 1")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened(): raise FileNotFoundError(args.video)
    index = saved = 0
    while True:
        ok, frame = capture.read()
        if not ok: break
        if index % args.every == 0:
            cv2.imwrite(str(output / f"frame_{index:06d}.bmp"), frame); saved += 1
        index += 1
    capture.release(); print(f"saved {saved} frames from {index}")


if __name__ == "__main__": main()
