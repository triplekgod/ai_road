"""Bounded read/analyze/write pipeline: lossless files, latest frames for cameras."""
import json
import math
import queue
import threading
import time
from pathlib import Path

import cv2

from profiling import FrameProfiler, print_summary


def run_video(analyzer, source, *, output=None, display=False, source_mode='file',
              queue_size=2, camera_fps=None, warmup=30, fps_window=30, report_path=None):
    if source_mode not in ('file', 'camera') or queue_size not in (2, 3):
        raise ValueError('source_mode file/camera and queue_size 2/3 required')
    if output and Path(str(source)).resolve() == Path(output).resolve():
        raise ValueError('Output must not overwrite input video')
    protected = {Path(str(source)).resolve()}
    if output:
        protected.add(Path(output).resolve())
    if report_path and Path(report_path).resolve() in protected:
        raise ValueError('Report must not overwrite source or output video')
    profiler = FrameProfiler(warmup, fps_window)
    analyzer.reset()
    capture_source = int(source) if source_mode == 'camera' and str(source).isdecimal() else str(source)
    cap = cv2.VideoCapture(capture_source)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f'Cannot open source: {source}')
    fps = cap.get(cv2.CAP_PROP_FPS)
    if source_mode == 'camera' and camera_fps is not None:
        fps = camera_fps
    if not math.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError('Cannot determine source FPS; for cameras provide --camera-fps')
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if source_mode == 'file' else None
    writer = None
    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            cap.release()
            writer.release()
            raise IOError(f'Cannot create output: {output}')
    read_queue, write_queue = queue.Queue(queue_size), queue.Queue(queue_size)
    stop = threading.Event()
    errors = queue.Queue()
    counters = {'frames_read': 0, 'frames_processed': 0, 'frames_written': 0, 'frames_dropped': 0}
    sentinel = object()

    def put(target, item):
        while not stop.is_set():
            try:
                target.put(item, timeout=.1)
                return True
            except queue.Full:
                pass
        return False

    def read_frames():
        try:
            while not stop.is_set():
                started = time.perf_counter()
                ok, frame = cap.read()
                read_time = time.perf_counter() - started
                if not ok:
                    break
                if frame.shape[:2] != (height, width):
                    raise ValueError('Source resolution changed; start a new output file')
                index = counters['frames_read']
                counters['frames_read'] += 1
                item = (index, frame, started, {'read': read_time})
                if source_mode == 'camera':
                    while not stop.is_set():
                        try:
                            read_queue.put_nowait(item)
                            break
                        except queue.Full:
                            try:
                                read_queue.get_nowait()
                                counters['frames_dropped'] += 1
                            except queue.Empty:
                                pass
                elif not put(read_queue, item):
                    break
            put(read_queue, sentinel)
        except BaseException as error:
            errors.put(error)
            stop.set()
        finally:
            cap.release()

    def write_frames():
        try:
            while not stop.is_set():
                try:
                    item = write_queue.get(timeout=.1)
                except queue.Empty:
                    continue
                if item is sentinel:
                    break
                _, rendered, captured, timings = item
                started = time.perf_counter()
                if writer is not None:
                    writer.write(rendered)
                    counters['frames_written'] += 1
                timings['write'] = time.perf_counter() - started
                profiler.record(timings, captured, time.perf_counter())
        except BaseException as error:
            errors.put(error)
            stop.set()
        finally:
            if writer is not None:
                writer.release()

    reader_thread = threading.Thread(target=read_frames, name='video-reader', daemon=True)
    writer_thread = threading.Thread(target=write_frames, name='video-writer', daemon=True)
    start = time.perf_counter()
    interrupted = False
    reader_thread.start()
    writer_thread.start()
    try:
        while not stop.is_set():
            try:
                item = read_queue.get(timeout=.1)
            except queue.Empty:
                continue
            if item is sentinel:
                break
            index, frame, captured, timings = item
            rendered, _ = analyzer.analyze(frame)
            counters['frames_processed'] += 1
            timings.update(analyzer.last_timings)
            if display:
                shown = time.perf_counter()
                cv2.imshow('Quarry road', rendered)
                quit_requested = cv2.waitKey(1) & 0xFF == ord('q')
                timings['render'] = timings.get('render', 0.) + time.perf_counter() - shown
            else:
                quit_requested = False
            if not put(write_queue, (index, rendered, captured, timings)):
                break
            if quit_requested:
                interrupted = True
                break
        if interrupted:
            stop.set()
        else:
            put(write_queue, sentinel)
            writer_thread.join(timeout=30)
            if writer_thread.is_alive():
                raise RuntimeError('Writer did not finish within 30 seconds')
    finally:
        stop.set()
        reader_thread.join(timeout=5)
        writer_thread.join(timeout=5)
        if display:
            cv2.destroyAllWindows()
    if not errors.empty():
        raise errors.get()
    if reader_thread.is_alive() or writer_thread.is_alive():
        raise RuntimeError('Video device did not stop within timeout')
    if not interrupted and source_mode == 'file':
        if counters['frames_read'] != counters['frames_processed']:
            raise RuntimeError('File frame accounting mismatch')
        if expected and expected > 0 and counters['frames_read'] != expected:
            raise IOError(f"Decoded {counters['frames_read']} frames; source reports {expected}. Possible decode failure")
        if output and counters['frames_written'] != counters['frames_read']:
            raise IOError('Output frame count differs from input')
    report = profiler.summary()
    report.update(counters)
    report.update({'source': str(source), 'output': str(output) if output else None,
                   'source_mode': source_mode, 'source_fps': fps, 'resolution': [width, height],
                   'queue_size': queue_size, 'elapsed_seconds': time.perf_counter()-start,
                   'interrupted': interrupted, 'model_metadata': analyzer.metadata,
                   'analyzer_configuration': getattr(analyzer, 'configuration', {})})
    if output and not interrupted:
        check = cv2.VideoCapture(str(output))
        try:
            output_count = int(check.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = check.get(cv2.CAP_PROP_FPS)
        finally:
            check.release()
        report.update({'output_frames_verified': output_count, 'output_fps_verified': output_fps})
        if output_count != counters['frames_written'] or abs(output_fps-fps) > .01:
            raise IOError('Encoded output did not preserve frame count/FPS')
    if report_path:
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print_summary(report)
    return report
