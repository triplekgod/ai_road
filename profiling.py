"""Per-frame service times and capture-to-write latency, excluding warmup."""
import platform
import os
from importlib.metadata import version, PackageNotFoundError

import numpy as np


STAGES = ('read', 'prepare', 'inference', 'temporal', 'geometry', 'render', 'write', 'total')


class FrameProfiler:
    def __init__(self, warmup=30, fps_window=30):
        if warmup < 0 or fps_window < 2:
            raise ValueError("warmup >= 0 and fps_window >= 2 required")
        self.warmup, self.fps_window = warmup, fps_window
        self.count = 0
        self.samples = {key: [] for key in STAGES}
        self.completions = []
        self.first_capture = None

    def record(self, timings, captured_at, completed_at):
        self.count += 1
        if self.count <= self.warmup:
            return
        if self.first_capture is None:
            self.first_capture = captured_at
        for key in STAGES:
            seconds = completed_at - captured_at if key == 'total' else timings.get(key, 0.)
            self.samples[key].append(seconds * 1000)
        self.completions.append(completed_at)

    def summary(self):
        result = {'warmup_frames': self.warmup, 'frames_completed': self.count,
                  'measured_frames': len(self.completions), 'fps_window_frames': self.fps_window,
                  'platform': platform.platform(), 'processor': platform.processor(),
                  'logical_cpus': os.cpu_count(),
                  'stage_ms': {}, 'average_fps': None, 'minimum_sustained_fps': None,
                  'latency_definition': 'read start through write completion, including queue waits',
                  'fps_definition': 'completion intervals after warmup; minimum over rolling windows'}
        result['runtime_versions'] = {}
        for package in ('numpy', 'opencv-python', 'torch', 'onnxruntime', 'openvino'):
            try:
                result['runtime_versions'][package] = version(package)
            except PackageNotFoundError:
                pass
        for key, values in self.samples.items():
            result['stage_ms'][key] = ({'mean': float(np.mean(values)), 'median': float(np.median(values)),
                                       'p95': float(np.percentile(values, 95))} if values else None)
        times = np.asarray(self.completions)
        if len(times) >= 2 and times[-1] > times[0]:
            result['average_fps'] = float((len(times) - 1) / (times[-1] - times[0]))
        n = self.fps_window
        if len(times) >= n:
            spans = times[n-1:] - times[:len(times)-n+1]
            result['minimum_sustained_fps'] = float((n - 1) / max(spans)) if max(spans) > 0 else None
        return result


def print_summary(report):
    print(f"Measured frames: {report['measured_frames']} (warmup: {report['warmup_frames']})")
    print('Stage          mean ms   median ms    p95 ms')
    for name, values in report['stage_ms'].items():
        if values:
            print(f"{name:12s} {values['mean']:9.2f} {values['median']:11.2f} {values['p95']:9.2f}")
    print(f"Average FPS: {report['average_fps']}; minimum sustained FPS: {report['minimum_sustained_fps']}")
