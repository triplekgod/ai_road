import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from corridor import CorridorState, build_zones, overlay_zones
from data import IMAGE_SIZE, MEAN, STD
from road_model import load_model


def tensor_from_frame(frame, device):
    image = cv2.resize(frame, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    image = (image - np.asarray(MEAN, np.float32)) / np.asarray(STD, np.float32)
    return torch.from_numpy(image.transpose(2, 0, 1)).unsqueeze(0).to(device)


def frame_source(path):
    path = Path(path)
    if path.is_dir():
        for item in sorted(path.iterdir()):
            frame = cv2.imread(str(item))
            if frame is not None:
                yield item.name, frame
    else:
        capture = cv2.VideoCapture(str(path))
        i = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield f"frame_{i:06d}", frame
            i += 1
        capture.release()


def main():
    parser = argparse.ArgumentParser(description="Segment road and draw stable longitudinal zones")
    parser.add_argument("source", help="video or image folder")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--out", type=Path, default=Path("runs/predictions"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    args.out.mkdir(parents=True, exist_ok=True)
    state = CorridorState()
    print("colors: LEFT=blue CENTER=green RIGHT=red; black holes remain obstacles")

    for index, (name, frame) in enumerate(frame_source(args.source)):
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            probability = model(tensor_from_frame(frame, device)).sigmoid()[0, 0].float().cpu().numpy()
        geometry_probability = cv2.resize(probability, (320, 180), interpolation=cv2.INTER_AREA)
        small_zones, state, info = build_zones(geometry_probability, state, args.threshold)
        zones = cv2.resize(small_zones, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        output = overlay_zones(frame, zones)
        cv2.imwrite(str(args.out / f"{Path(name).stem}_zones.jpg"), output)
        counts = np.bincount(zones.ravel(), minlength=4)
        total = max(int(counts[1:].sum()), 1)
        fps = 1.0 / max(time.perf_counter() - started, 1e-6)
        print(f"frame={index:06d} road_px={total} left={counts[1]/total:.1%} center={counts[2]/total:.1%} right={counts[3]/total:.1%} heading={info['heading_deg']:+.1f}deg confidence={info['confidence']:.3f} fps={fps:.1f}")


if __name__ == "__main__":
    main()
