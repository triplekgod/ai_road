"""CPU-first segmentation; geometry runs on the unpadded small ROI."""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from backends import load_backend
from preprocess import prepare_input, validate_input
from road_geometry import Zones, draw_zone_outlines, primary_road, smooth_road_mask, zone_mask
from temporal import TemporalFilter


class RoadAnalyzer:
    def __init__(self, checkpoint=None, threshold=.55, zones=None, temporal_window=5,
                 min_confirmed_frames=4, device='cpu', *, backend='pytorch', threads=1,
                 input_width=None, input_height=None, roi_top=None, temporal_mode='ema',
                 temporal_alpha=.70, low_threshold=.40, high_threshold=.65,
                 scene_threshold=.25, geometry='centerline', geometry_config=None,
                 predictor=None):
        self.predictor = predictor if predictor is not None else load_backend(
            checkpoint, backend=backend, device=device, threads=threads)
        self.metadata = dict(self.predictor.metadata)
        self.width = self.metadata.get('input_width', 256) if input_width is None else input_width
        self.height = self.metadata.get('input_height', 144) if input_height is None else input_height
        self.roi_top = self.metadata.get('roi_top', .20) if roi_top is None else roi_top
        self.preprocess_version = self.metadata.get('preprocess_version', 1)
        validate_input(self.width, self.height, self.roi_top, self.preprocess_version)
        if threads < 1:
            raise ValueError('threads must be positive')
        cv2.setNumThreads(threads)
        for name, value in (('input_width', self.width), ('input_height', self.height), ('roi_top', self.roi_top)):
            if name in self.metadata and value != self.metadata[name]:
                raise ValueError(f"{name} must match model metadata ({self.metadata[name]})")
        if geometry not in ('legacy', 'centerline'):
            raise ValueError("geometry must be legacy or centerline")
        self.zones = zones or Zones()
        self.geometry_mode = geometry
        self.geometry = None
        if geometry == 'centerline':
            from road_geometry import CenterlineGeometry
            self.geometry = CenterlineGeometry(zones=self.zones, config=geometry_config)
        self.temporal = TemporalFilter(temporal_mode, temporal_alpha, low_threshold,
                                       high_threshold, threshold, temporal_window,
                                       min_confirmed_frames, scene_threshold)
        self.last_timings = {}
        self.last_geometry = None
        from dataclasses import asdict
        self.configuration = {'backend': backend, 'device': device, 'threads': threads,
                              'geometry': geometry, 'geometry_config': asdict(self.geometry.config) if self.geometry else None,
                              'temporal_mode': temporal_mode, 'temporal_alpha': temporal_alpha,
                              'low_threshold': low_threshold, 'high_threshold': high_threshold,
                              'legacy_threshold': threshold, 'scene_threshold': scene_threshold,
                              'temporal_window': temporal_window, 'min_confirmed_frames': min_confirmed_frames,
                              'zones': asdict(self.zones)}

    def reset(self):
        self.temporal.reset()
        if self.geometry is not None:
            self.geometry.reset()
        self.last_geometry = None

    def analyze(self, frame):
        start = time.perf_counter()
        tensor, transform = prepare_input(frame, self.width, self.height,
                                           self.roi_top, self.preprocess_version)
        prepared = time.perf_counter()
        probability = transform.unpad(self.predictor.predict(tensor))
        inferred = time.perf_counter()
        small = self.temporal.update(probability, frame)
        if self.temporal.was_reset and self.geometry is not None:
            self.geometry.reset()
        filtered = time.perf_counter()
        if self.geometry is not None:
            result = self.geometry.process(small)
            road_small, colors_small = result.road, result.colors
            self.last_geometry = result
        else:
            road_small = smooth_road_mask(primary_road(small, min_area=max(8, small.size // 700)))
            colors_small = zone_mask(road_small, self.zones)
        colors_small = draw_zone_outlines(colors_small, road_small)
        colors_small[road_small == 0] = 0
        geometry_done = time.perf_counter()
        road = transform.restore(road_small, already_unpadded=True)
        overlay = transform.restore(colors_small, already_unpadded=True)
        overlay[road == 0] = (0, 0, 255)
        rendered = cv2.addWeighted(frame, .55, overlay, .45, 0)
        finished = time.perf_counter()
        self.last_timings = {'prepare': prepared-start, 'inference': inferred-prepared,
                             'temporal': filtered-inferred, 'geometry': geometry_done-filtered,
                             'render': finished-geometry_done}
        return rendered, road


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('video', help='File, stream URL, or camera index')
    p.add_argument('model', help='Checkpoint .pth or .onnx with .onnx.json metadata')
    p.add_argument('--output')
    p.add_argument('--report', help='Save stage timings and frame accounting as JSON')
    p.add_argument('--backend', choices=('pytorch', 'onnxruntime', 'openvino'), default='pytorch')
    p.add_argument('--device', default='cpu')
    p.add_argument('--threads', type=int, default=1)
    p.add_argument('--input-width', type=int, help='Must match checkpoint')
    p.add_argument('--input-height', type=int, help='Must match checkpoint')
    p.add_argument('--roi-top', type=float, help='Must match checkpoint')
    p.add_argument('--left', type=float, default=.15)
    p.add_argument('--center', type=float, default=.70)
    p.add_argument('--right', type=float, default=.15)
    p.add_argument('--geometry', choices=('legacy', 'centerline'), default='centerline')
    p.add_argument('--corridor-mode', choices=('all', 'main'), default='all')
    p.add_argument('--min-branch-length', type=float, default=.12)
    p.add_argument('--min-branch-width', type=float, default=.05)
    p.add_argument('--min-branch-angle', type=float, default=25.)
    p.add_argument('--min-branch-area', type=float, default=.002)
    p.add_argument('--branch-confirm-frames', type=int, default=3)
    p.add_argument('--branch-missing-frames', type=int, default=2)
    p.add_argument('--temporal-mode', choices=('ema', 'legacy', 'none'), default='ema')
    p.add_argument('--temporal-alpha', type=float, default=.70)
    p.add_argument('--low-threshold', type=float, default=.40)
    p.add_argument('--high-threshold', type=float, default=.65)
    p.add_argument('--threshold', type=float, default=.55, help='Only for legacy voting')
    p.add_argument('--temporal-window', type=int, default=5)
    p.add_argument('--min-confirmed-frames', type=int, default=4)
    p.add_argument('--scene-threshold', type=float, default=.25)
    p.add_argument('--source-mode', choices=('file', 'camera'), default='file')
    p.add_argument('--queue-size', type=int, choices=(2, 3), default=2)
    p.add_argument('--camera-fps', type=float, help='Explicit camera FPS if device does not report it')
    p.add_argument('--warmup', type=int, default=30)
    p.add_argument('--fps-window', type=int, default=30)
    p.add_argument('--no-display', action='store_true')
    return p


def main(argv=None):
    from road_geometry import GeometryConfig
    from video_pipeline import run_video
    a = build_parser().parse_args(argv)
    protected = {Path(a.model).resolve(), Path(str(a.model) + '.json').resolve()}
    if any(path and Path(path).resolve() in protected for path in (a.output, a.report)):
        raise ValueError('Output and report must not overwrite the model or its metadata')
    config = GeometryConfig(min_branch_length=a.min_branch_length,
                            min_branch_width=a.min_branch_width,
                            min_branch_angle=a.min_branch_angle,
                            min_branch_area=a.min_branch_area,
                            confirm_frames=a.branch_confirm_frames,
                            missing_frames=a.branch_missing_frames,
                            corridor_mode=a.corridor_mode)
    analyzer = RoadAnalyzer(a.model, a.threshold, Zones(a.left, a.center, a.right),
                            a.temporal_window, a.min_confirmed_frames, a.device,
                            backend=a.backend, threads=a.threads, input_width=a.input_width,
                            input_height=a.input_height, roi_top=a.roi_top,
                            temporal_mode=a.temporal_mode, temporal_alpha=a.temporal_alpha,
                            low_threshold=a.low_threshold, high_threshold=a.high_threshold,
                            scene_threshold=a.scene_threshold, geometry=a.geometry, geometry_config=config)
    return run_video(analyzer, a.video, output=a.output, display=not a.no_display,
                     source_mode=a.source_mode, queue_size=a.queue_size,
                     camera_fps=a.camera_fps, warmup=a.warmup, fps_window=a.fps_window,
                     report_path=a.report)


if __name__ == '__main__':
    main()
