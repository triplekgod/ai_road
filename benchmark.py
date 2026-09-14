import argparse
import statistics
import time

import torch

from data import ACCURATE_SIZE, FAST_SIZE
from road_model import RoadNet


def main():
    parser = argparse.ArgumentParser(description="Measure pure neural-network CPU speed")
    parser.add_argument("--arch", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--frames", type=int, default=50)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--width", type=int, default=0)
    args = parser.parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    default_size = FAST_SIZE if args.arch == "fast" else ACCURATE_SIZE
    size = default_size if args.width == 0 else (args.width, max(90, round(args.width * 9 / 16)))
    model = RoadNet(args.arch, pretrained=False).eval().to(memory_format=torch.channels_last)
    sample = torch.randn(1, 3, size[1], size[0]).contiguous(memory_format=torch.channels_last)
    with torch.inference_mode():
        for _ in range(10):
            model(sample)
        times = []
        for _ in range(args.frames):
            started = time.perf_counter()
            model(sample)
            times.append(time.perf_counter() - started)
    ordered = sorted(times)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    params = sum(p.numel() for p in model.parameters())
    print(f"arch={args.arch} input={size[0]}x{size[1]} params={params/1e6:.2f}M threads={torch.get_num_threads()}")
    print(f"mean_ms={statistics.mean(times)*1000:.2f} p95_ms={p95*1000:.2f} FPS={1/statistics.mean(times):.1f}")


if __name__ == "__main__":
    main()
