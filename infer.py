import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from corridor import CorridorState, build_zones, overlay_zones
from data import MEAN, STD
from road_model import load_model


def tensor_from_frame(frame, device, image_size):
    image = cv2.resize(frame, image_size, interpolation=cv2.INTER_AREA)
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
    parser.add_argument("--output-video", type=Path, help="write rendered frames directly to MP4")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 = automatic")
    parser.add_argument("--width", type=int, default=0, help="0 = checkpoint default, 256 = weak CPU")
    parser.add_argument("--no-show", action="store_true", help="disable the live OpenCV window")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and args.threads > 0:
        torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, device)
    image_size = model.input_size if args.width == 0 else (args.width, max(90, round(args.width * 9 / 16)))
    if device.type == "cpu":
        model = model.to(memory_format=torch.channels_last)
    source_path = Path(args.source)
    source_fps = 25.0
    if source_path.is_file():
        probe = cv2.VideoCapture(str(source_path))
        detected_fps = probe.get(cv2.CAP_PROP_FPS)
        probe.release()
        if detected_fps and np.isfinite(detected_fps):
            source_fps = float(detected_fps)
    writer = None
    if args.output_video:
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
    state = CorridorState()
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device} arch={model.architecture} params={parameters/1e6:.2f}M input={image_size[0]}x{image_size[1]} threads={torch.get_num_threads()} output_video={args.output_video}")
    print("colors: LEFT=blue CENTER=green RIGHT=red; black holes remain obstacles")

    for index, (name, frame) in enumerate(frame_source(args.source)):
        started = time.perf_counter()
        with torch.inference_mode(), torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            tensor = tensor_from_frame(frame, device, image_size)
            if device.type == "cpu":
                tensor = tensor.contiguous(memory_format=torch.channels_last)
            probability = model(tensor).sigmoid()[0, 0].float().cpu().numpy()
        geometry_probability = cv2.resize(probability, (320, 180), interpolation=cv2.INTER_AREA)
        small_zones, state, info = build_zones(geometry_probability, state, args.threshold)
        zones = cv2.resize(small_zones, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        output = overlay_zones(frame, zones)
        counts = np.bincount(zones.ravel(), minlength=4)
        total = max(int(counts[1:].sum()), 1)
        fps = 1.0 / max(time.perf_counter() - started, 1e-6)
        status = f"FPS {fps:.1f}  heading {info['heading_deg']:+.1f} deg  confidence {info['confidence']:.2f}"
        cv2.rectangle(output, (0, 0), (min(output.shape[1], 680), 42), (0, 0, 0), -1)
        cv2.putText(output, status, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
        if args.output_video:
            if writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(args.output_video), fourcc, source_fps, (output.shape[1], output.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError(f"Cannot create MP4: {args.output_video}")
                print(f"video_writer={args.output_video} fps={source_fps:.3f} size={output.shape[1]}x{output.shape[0]}")
            writer.write(output)
        print(f"frame={index:06d} road_px={total} left={counts[1]/total:.1%} center={counts[2]/total:.1%} right={counts[3]/total:.1%} heading={info['heading_deg']:+.1f}deg confidence={info['confidence']:.3f} fps={fps:.1f}")
        if not args.no_show:
            cv2.imshow("AI Road - live | Q/Esc: stop | Space: pause", output)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("stopped_by_user=true")
                break
            if key == 32:
                print("paused=true; press any key to continue")
                cv2.waitKey(0)

    if writer is not None:
        writer.release()
        print(f"video_saved={args.output_video}")
    if not args.no_show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
