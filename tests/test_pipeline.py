import contextlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from backends import checkpoint_metadata
from infer import RoadAnalyzer
from preprocess import prepare_input, validate_input
from profiling import FrameProfiler
from temporal import TemporalFilter, hysteresis
from video_pipeline import run_video


class StaticPredictor:
    def __init__(self, probability, metadata=None):
        self.probability = probability.astype(np.float32)
        self.metadata = metadata or {"input_width": probability.shape[1],
                                     "input_height": probability.shape[0],
                                     "roi_top": .2, "preprocess_version": 1}
        self.inputs = []

    def predict(self, tensor):
        self.inputs.append(tensor.copy())
        return self.probability.copy()


class IdentityAnalyzer:
    metadata = {"test_predictor": "identity"}
    last_timings = {"prepare": .0001, "inference": .0002,
                    "temporal": .0001, "geometry": .0001, "render": .0001}

    def __init__(self, delay=0, failure=None, reset_failure=None):
        self.delay, self.failure, self.reset_failure = delay, failure, reset_failure
        self.frames, self.resets = [], 0

    def reset(self):
        self.resets += 1
        if self.reset_failure:
            raise self.reset_failure

    def analyze(self, frame):
        if self.failure:
            raise self.failure
        self.frames.append(float(frame.mean()))
        if self.delay:
            time.sleep(self.delay)
        return frame.copy(), None


class FakeCapture:
    def __init__(self, count=30, fps=25):
        self.count, self.fps, self.index = count, fps, 0
        self.released = False

    def isOpened(self):
        return True

    def get(self, field):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_WIDTH: 64,
                cv2.CAP_PROP_FRAME_HEIGHT: 48, cv2.CAP_PROP_FRAME_COUNT: self.count}.get(field, 0)

    def read(self):
        if self.index >= self.count:
            return False, None
        frame = np.full((48, 64, 3), self.index, np.uint8)
        self.index += 1
        return True, frame

    def release(self):
        self.released = True


class FakeWriter:
    def __init__(self, error=None, opened=True):
        self.error, self.opened, self.released = error, opened, False

    def isOpened(self):
        return self.opened

    def write(self, frame):
        if self.error:
            raise self.error

    def release(self):
        self.released = True


class PreprocessTest(unittest.TestCase):
    def test_letterbox_preserves_proportions_and_rgb_layout(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        frame[:] = (20, 80, 200)
        tensor, transform = prepare_input(frame)
        self.assertEqual(tensor.shape, (1, 3, 144, 256))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        self.assertEqual(transform.roi_y, 72)
        self.assertAlmostEqual(transform.content_width / transform.content_height,
                               640 / 288, delta=.02)
        np.testing.assert_allclose(tensor[0, :, transform.top, transform.left],
                                   np.array([200, 80, 20]) / 255, atol=1e-6)
        self.assertFalse(np.any(tensor[:, :, :transform.top]))

    def test_mask_roi_roundtrip_and_padding_exclusion(self):
        frame = np.zeros((360, 640, 3), np.uint8)
        _, transform = prepare_input(frame)
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[72:, 180:460] = 255
        padded = transform.apply_mask(mask)
        restored = transform.restore(padded)
        intersection = np.count_nonzero((mask > 0) & (restored > 0))
        union = np.count_nonzero((mask > 0) | (restored > 0))
        self.assertGreater(intersection / union, .985)
        self.assertFalse(np.any(restored[:72]))
        np.testing.assert_array_equal(restored, transform.restore(transform.unpad(padded), already_unpadded=True))
        # Predictions in letterbox padding must never become road pixels.
        padding = np.full((144, 256), 255, np.uint8)
        padding[transform.top:transform.top + transform.content_height,
                transform.left:transform.left + transform.content_width] = 0
        self.assertFalse(np.any(transform.restore(padding)))

    def test_portrait_frame_horizontal_padding_and_color_restore(self):
        _, transform = prepare_input(np.zeros((640, 360, 3), np.uint8), roi_top=0)
        self.assertGreater(transform.left, 0)
        color = np.zeros((144, 256, 3), np.uint8)
        color[:, transform.left:transform.left + transform.content_width] = (0, 255, 0)
        restored = transform.restore(color)
        self.assertEqual(restored.shape, (640, 360, 3))
        self.assertTrue(np.all(restored[:, :, 1] == 255))

    def test_invalid_dimensions_roi_and_restore_shape(self):
        for arguments in ((255, 144, .2), (256, 145, .2), (256, 144, 1),
                          (256, 144, -.1), (256, 144, float("nan"))):
            with self.assertRaises(ValueError):
                validate_input(*arguments)
        _, transform = prepare_input(np.zeros((100, 200, 3), np.uint8))
        with self.assertRaises(ValueError):
            transform.restore(np.zeros((100, 200), np.uint8))
        with self.assertRaises(ValueError):
            transform.apply_mask(np.zeros((101, 200), np.uint8))

    def test_legacy_checkpoint_preserves_square_and_full_frame(self):
        metadata = checkpoint_metadata({"model": {}, "image_size": 192})
        self.assertEqual((metadata["input_width"], metadata["input_height"],
                          metadata["roi_top"], metadata["preprocess_version"]), (192, 192, 0., 0))
        predictor = StaticPredictor(np.ones((192, 192), np.float32), metadata)
        analyzer = RoadAnalyzer(predictor=predictor, temporal_mode="none")
        rendered, road = analyzer.analyze(np.zeros((144, 320, 3), np.uint8))
        self.assertEqual(predictor.inputs[0].shape, (1, 3, 192, 192))
        self.assertEqual(rendered.shape, (144, 320, 3))
        self.assertTrue(np.all(road == 255))
        with self.assertRaises(ValueError):
            RoadAnalyzer(predictor=predictor, roi_top=.2)

    def test_legacy_nonmultiple_size_remains_loadable(self):
        metadata = checkpoint_metadata({"model": {}, "image_size": 200})
        predictor = StaticPredictor(np.ones((200, 200), np.float32), metadata)
        analyzer = RoadAnalyzer(predictor=predictor, temporal_mode="none")
        _, road = analyzer.analyze(np.zeros((144, 320, 3), np.uint8))
        self.assertEqual(predictor.inputs[0].shape, (1, 3, 200, 200))
        self.assertTrue(np.all(road == 255))

    def test_analyzer_zones_never_leak_beyond_restored_road(self):
        probability = np.zeros((144, 256), np.float32)
        probability[:, 95:162] = .95
        analyzer = RoadAnalyzer(predictor=StaticPredictor(probability), temporal_mode="none")
        frame = np.zeros((360, 640, 3), np.uint8)
        rendered, road = analyzer.analyze(frame)
        outside = road == 0
        self.assertFalse(np.any(road[:72]))
        self.assertTrue(np.any(road[72:]))
        self.assertTrue(np.all(rendered[outside, :2] == 0))
        self.assertTrue(np.all(rendered[outside, 2] == 115))
        self.assertTrue(np.any(rendered[~outside, 1] > 0))
        self.assertFalse(np.any(analyzer.last_geometry.center[analyzer.last_geometry.road == 0]))
        self.assertEqual(set(analyzer.last_timings), {"prepare", "inference", "temporal", "geometry", "render"})
        self.assertTrue(all(value >= 0 for value in analyzer.last_timings.values()))


class TemporalTest(unittest.TestCase):
    frame = np.zeros((48, 64, 3), np.uint8)

    def test_hysteresis_requires_connection_to_high_confidence_seed(self):
        probability = np.zeros((20, 30), np.float32)
        probability[2:12, 3:9] = .5
        probability[6, 5] = .8
        probability[2:12, 20:26] = .5
        result = hysteresis(probability)
        self.assertTrue(np.all(result[2:12, 3:9] == 255))
        self.assertFalse(np.any(result[:, 20:]))
        self.assertEqual(np.count_nonzero(result), 60)

    def test_ema_formula_and_initial_frame(self):
        temporal = TemporalFilter(alpha=.7)
        high = np.full((12, 16), .9, np.float32)
        low = np.full_like(high, .2)
        self.assertTrue(np.all(temporal.update(high, self.frame) == 255))
        self.assertTrue(temporal.was_reset)
        result = temporal.update(low, self.frame)
        np.testing.assert_allclose(temporal.previous, .7 * low + .3 * high)
        self.assertFalse(temporal.was_reset)
        self.assertFalse(np.any(result))

    def test_scene_change_resize_and_explicit_reset(self):
        temporal = TemporalFilter(alpha=.1)
        temporal.update(np.ones((12, 16), np.float32), self.frame)
        black = np.zeros((12, 16), np.float32)
        self.assertFalse(np.any(temporal.update(black, np.full_like(self.frame, 255))))
        self.assertTrue(temporal.was_reset)
        np.testing.assert_array_equal(temporal.previous, black)
        temporal.update(np.ones((10, 14), np.float32), np.zeros((60, 80, 3), np.uint8))
        self.assertTrue(temporal.was_reset)
        self.assertEqual(temporal.previous.shape, (10, 14))
        temporal.reset()
        self.assertIsNone(temporal.previous)
        self.assertEqual(len(temporal.history), 0)

    def test_legacy_vote_and_none_mode(self):
        legacy = TemporalFilter(mode="legacy")
        high = np.ones((8, 8), np.float32)
        for _ in range(3):
            self.assertFalse(np.any(legacy.update(high, self.frame)))
        self.assertTrue(np.all(legacy.update(high, self.frame) == 255))
        self.assertFalse(np.any(legacy.update(np.zeros_like(high), self.frame)))
        none = TemporalFilter(mode="none", alpha=.01)
        none.update(high, self.frame)
        self.assertFalse(np.any(none.update(np.zeros_like(high), self.frame)))

    def test_invalid_nonfinite_probabilities(self):
        probability = np.zeros((8, 8), np.float32)
        probability[0, 0] = np.nan
        with self.assertRaises(ValueError):
            TemporalFilter().update(probability, self.frame)


class ProfilingTest(unittest.TestCase):
    def test_warmup_statistics_and_rolling_minimum_fps(self):
        profiler = FrameProfiler(warmup=2, fps_window=3)
        profiler.record({"inference": 99}, 0, 1)
        profiler.record({"inference": 99}, 1, 2)
        completions = [10., 10.1, 10.2, 10.4]
        inference = [.01, .02, .03, .04]
        for completed, duration in zip(completions, inference):
            profiler.record({"inference": duration}, completed - .05, completed)
        summary = profiler.summary()
        self.assertEqual(summary["frames_completed"], 6)
        self.assertEqual(summary["measured_frames"], 4)
        self.assertAlmostEqual(summary["stage_ms"]["inference"]["mean"], 25)
        self.assertAlmostEqual(summary["stage_ms"]["inference"]["median"], 25)
        self.assertAlmostEqual(summary["stage_ms"]["inference"]["p95"], 38.5)
        self.assertAlmostEqual(summary["stage_ms"]["total"]["mean"], 50)
        self.assertAlmostEqual(summary["average_fps"], 7.5)
        self.assertAlmostEqual(summary["minimum_sustained_fps"], 2 / .3)

    def test_no_measured_frames_is_explicit(self):
        profiler = FrameProfiler(warmup=30)
        profiler.record({}, 0, .1)
        summary = profiler.summary()
        self.assertEqual(summary["measured_frames"], 0)
        self.assertIsNone(summary["average_fps"])
        self.assertIsNone(summary["minimum_sustained_fps"])
        self.assertTrue(all(value is None for value in summary["stage_ms"].values()))


class VideoPipelineTest(unittest.TestCase):
    def run_quiet(self, *args, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return run_video(*args, **kwargs)

    def bounded_error(self, function, error_type, message):
        errors = []
        def invoke():
            try:
                function()
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        worker.join(timeout=3)
        self.assertFalse(worker.is_alive(), "Pipeline hung after an error")
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], error_type)
        self.assertIn(message, str(errors[0]))

    def test_actual_mp4_preserves_order_count_and_fps(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.mp4"
            output = Path(directory) / "output.mp4"
            report_path = Path(directory) / "report.json"
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 12.5, (64, 48))
            self.assertTrue(writer.isOpened(), "Test requires OpenCV mp4v encoder")
            try:
                for index in range(24):
                    writer.write(np.full((48, 64, 3), 12 + index * 8, np.uint8))
            finally:
                writer.release()
            analyzer = IdentityAnalyzer(delay=.001)
            report = self.run_quiet(analyzer, str(source), output=str(output), warmup=3,
                                    fps_window=5, report_path=str(report_path))
            self.assertEqual(analyzer.resets, 1)
            self.assertEqual(report["frames_read"], 24)
            self.assertEqual(report["frames_processed"], 24)
            self.assertEqual(report["frames_written"], 24)
            self.assertEqual(report["frames_dropped"], 0)
            self.assertEqual(report["measured_frames"], 21)
            self.assertEqual(report["output_frames_verified"], 24)
            self.assertAlmostEqual(report["output_fps_verified"], 12.5, places=2)
            self.assertEqual(json.loads(report_path.read_text())["frames_written"], 24)
            capture = cv2.VideoCapture(str(output))
            decoded = []
            try:
                while True:
                    success, frame = capture.read()
                    if not success:
                        break
                    decoded.append(float(frame.mean()))
            finally:
                capture.release()
            self.assertEqual(len(decoded), 24)
            self.assertTrue(np.all(np.diff(decoded) > 4), "Frame sequence must remain strictly ordered")
            np.testing.assert_allclose(decoded, analyzer.frames, atol=4)

    def test_analyzer_failure_stops_workers_and_releases_capture(self):
        capture = FakeCapture(100)
        with patch("video_pipeline.cv2.VideoCapture", return_value=capture):
            self.bounded_error(lambda: self.run_quiet(IdentityAnalyzer(failure=ValueError("analysis failed")), "video"),
                               ValueError, "analysis failed")
        self.assertTrue(capture.released)
        self.assertFalse(any(thread.name in ("video-reader", "video-writer") for thread in threading.enumerate()))

    def test_writer_failure_stops_workers_and_releases_devices(self):
        capture = FakeCapture(100)
        writer = FakeWriter(error=IOError("write failed"))
        with tempfile.TemporaryDirectory() as directory:
            with patch("video_pipeline.cv2.VideoCapture", return_value=capture), \
                    patch("video_pipeline.cv2.VideoWriter", return_value=writer):
                self.bounded_error(lambda: self.run_quiet(IdentityAnalyzer(), "video", output=str(Path(directory) / "out.mp4")),
                                   IOError, "write failed")
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)

    def test_failed_writer_open_releases_capture(self):
        capture = FakeCapture()
        writer = FakeWriter(opened=False)
        with tempfile.TemporaryDirectory() as directory:
            with patch("video_pipeline.cv2.VideoCapture", return_value=capture), \
                    patch("video_pipeline.cv2.VideoWriter", return_value=writer):
                with self.assertRaisesRegex(IOError, "Cannot create output"):
                    self.run_quiet(IdentityAnalyzer(), "video", output=str(Path(directory) / "out.mp4"))
        self.assertTrue(capture.released)
        self.assertTrue(writer.released)

    def test_reset_failure_does_not_open_devices(self):
        capture = FakeCapture()
        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as directory:
            with patch("video_pipeline.cv2.VideoCapture", return_value=capture) as capture_factory, \
                    patch("video_pipeline.cv2.VideoWriter", return_value=writer) as writer_factory:
                with self.assertRaisesRegex(ValueError, "reset failed"):
                    self.run_quiet(IdentityAnalyzer(reset_failure=ValueError("reset failed")), "video",
                                   output=str(Path(directory) / "out.mp4"))
        capture_factory.assert_not_called()
        writer_factory.assert_not_called()

    def test_same_path_is_rejected_before_opening_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "same.mp4"
            source.write_bytes(b"original data")
            with patch("video_pipeline.cv2.VideoCapture") as capture:
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    self.run_quiet(IdentityAnalyzer(), str(source), output=str(source.parent / "." / source.name))
                capture.assert_not_called()
            self.assertEqual(source.read_bytes(), b"original data")

    def test_camera_drops_old_frames_but_keeps_latest(self):
        capture = FakeCapture(count=80)
        analyzer = IdentityAnalyzer(delay=.003)
        with patch("video_pipeline.cv2.VideoCapture", return_value=capture):
            report = self.run_quiet(analyzer, "0", source_mode="camera", warmup=0)
        self.assertTrue(capture.released)
        self.assertGreater(report["frames_dropped"], 0)
        self.assertEqual(report["frames_read"], 80)
        self.assertEqual(report["frames_processed"] + report["frames_dropped"], 80)
        self.assertEqual(analyzer.frames[-1], 79)
        self.assertTrue(np.all(np.diff(analyzer.frames) > 0))

    def test_file_mode_does_not_drop_frames_with_slow_analysis(self):
        capture = FakeCapture(count=20)
        analyzer = IdentityAnalyzer(delay=.002)
        with patch("video_pipeline.cv2.VideoCapture", return_value=capture):
            report = self.run_quiet(analyzer, "video", source_mode="file", warmup=0)
        self.assertEqual(report["frames_dropped"], 0)
        self.assertEqual(analyzer.frames, list(range(20)))


if __name__ == "__main__":
    unittest.main()
