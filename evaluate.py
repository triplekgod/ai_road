"""Evaluate independent groups in original image coordinates, including excluded ROI."""
import argparse
import json
import os
import platform
import time
from pathlib import Path
import cv2
import numpy as np
from backends import load_backend
from data_tools import load_manifest, read_image
from metrics import SegmentationMetrics
from preprocess import prepare_input


def verify_evaluation_provenance(metadata, audit, split="test", allow_unverified=False):
    current = audit["splits"][split]
    groups, hashes = set(current["groups"]), set(current["image_hashes"])
    forbidden_groups = set(metadata.get("training_groups", []))
    forbidden_hashes = set(metadata.get("training_image_hashes", []))
    if split == "test":
        forbidden_groups.update(metadata.get("validation_groups", []))
        forbidden_hashes.update(metadata.get("validation_image_hashes", []))
        calibration = metadata.get("calibration") or {}
        forbidden_groups.update(calibration.get("groups", []))
        forbidden_hashes.update(calibration.get("image_hashes", []))
    if groups.intersection(forbidden_groups) or hashes.intersection(forbidden_hashes):
        raise ValueError(f"Evaluation leakage: {split} overlaps checkpoint training/validation/calibration provenance")
    required = ("training_groups", "training_image_hashes", "validation_groups", "validation_image_hashes")
    verified = all(key in metadata for key in required) and metadata.get("provenance_complete", True)
    if not verified and not allow_unverified:
        raise ValueError("Checkpoint has incomplete data provenance. Use --allow-unverified-provenance only for an explicitly unverified legacy/pretrained comparison.")
    return bool(verified)


def latency_summary(milliseconds):
    values = np.asarray(milliseconds, dtype=np.float64)
    if not values.size:
        raise ValueError("No measured frames")
    return {"mean_ms": float(values.mean()), "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "fps": float(1000 / values.mean()) if values.mean() else None}


def evaluate_model(model_path, samples, audit, *, backend="pytorch", split="test",
                   threshold=None, boundary_ratio=None, device="cpu", threads=1,
                   warmup=30, allow_unverified=False):
    runtime = load_backend(model_path, backend=backend, device=device, threads=threads)
    metadata = runtime.metadata
    threshold = metadata.get("validation_threshold", .55) if threshold is None else threshold
    boundary_ratio = metadata.get("boundary_ratio", .02) if boundary_ratio is None else boundary_ratio
    if not 0 < threshold < 1 or not 0 < boundary_ratio <= 1:
        raise ValueError("threshold must be in (0,1), boundary_ratio in (0,1]")
    verified = verify_evaluation_provenance(metadata, audit, split, allow_unverified)
    preprocessing = {key: metadata[key] for key in ("input_width", "input_height", "roi_top", "preprocess_version")}
    first_image = read_image(samples[0]["image"])
    first_tensor, _ = prepare_input(first_image, **preprocessing)
    for _ in range(warmup):
        runtime.predict(first_tensor)
    metric = SegmentationMetrics(boundary_ratio)
    inference_times, analysis_times = [], []
    for sample in samples:
        start = time.perf_counter()
        image = read_image(sample["image"])
        tensor, transform = prepare_input(image, **preprocessing)
        inference_start = time.perf_counter()
        probabilities = runtime.predict(tensor)
        inference_times.append((time.perf_counter() - inference_start) * 1000)
        if probabilities.shape != (preprocessing["input_height"], preprocessing["input_width"]):
            raise ValueError("Backend output shape does not match preprocessing metadata")
        if not np.isfinite(probabilities).all():
            raise ValueError("Backend produced non-finite road probabilities")
        prediction = transform.restore((probabilities >= threshold).astype(np.uint8))
        analysis_times.append((time.perf_counter() - start) * 1000)
        target = read_image(sample["mask"], cv2.IMREAD_UNCHANGED) > 0
        metric.update(prediction > 0, target)
    artifact = Path(model_path)
    size = artifact.stat().st_size
    if artifact.suffix.lower() == ".xml" and artifact.with_suffix(".bin").exists():
        size += artifact.with_suffix(".bin").stat().st_size
    return {"model": str(artifact.resolve()), "backend": backend, "architecture": metadata["architecture"],
            "preprocessing": preprocessing, "metrics": metric.compute(), "threshold": threshold,
            "metric_space": "original_full_frame_including_area_above_roi",
            "provenance_verified": verified, "split": split,
            "dataset_fingerprint": audit["dataset_fingerprint"], "model_bytes": size,
            "warmup_runs_excluded": warmup, "inference_latency": latency_summary(inference_times),
            "analysis_latency": latency_summary(analysis_times),
            "latency_scope": "analysis includes image decode, prepare, inference and restore; excludes mask decode, metrics, drawing and video encoding",
            "quantization": metadata.get("quantization"),
            "target_cpu_verified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--backend", choices=("pytorch", "onnxruntime", "openvino"), default="pytorch")
    parser.add_argument("--compare", help="Second model evaluated on exactly the same frames and threshold")
    parser.add_argument("--compare-backend", choices=("pytorch", "onnxruntime", "openvino"), default="onnxruntime")
    parser.add_argument("--threshold", type=float, help="Default: saved validation threshold or .55. Freeze before testing.")
    parser.add_argument("--boundary-ratio", type=float, help="Default: saved value or .02")
    parser.add_argument("--max-iou-drop", type=float, default=.02, help="Allowed absolute IoU drop, .02 = 2 percentage points")
    parser.add_argument("--enforce-iou-drop", action="store_true", help="Exit 2 if comparison exceeds allowed IoU drop (automatic for INT8)")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-unverified-provenance", action="store_true")
    parser.add_argument("--output", default="reports/evaluation.json")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    protected = {Path(args.manifest).resolve()}
    for artifact in (args.model, args.compare):
        if artifact:
            protected.update((Path(artifact).resolve(), Path(str(artifact) + ".json").resolve()))
    if output in protected:
        parser.error("--output must not overwrite a model, its metadata sidecar, or the dataset manifest")
    if (args.threshold is not None and not 0 < args.threshold < 1) or (args.boundary_ratio is not None and not 0 < args.boundary_ratio <= 1):
        parser.error("threshold must be in (0,1), boundary-ratio in (0,1]")
    if args.warmup < 0 or args.threads < 1 or not 0 <= args.max_iou_drop <= 1:
        parser.error("warmup must be non-negative, threads positive, max-iou-drop in [0,1]")
    cv2.setNumThreads(args.threads)
    splits, audit = load_manifest(args.manifest)
    options = {"split": args.split, "threshold": args.threshold, "boundary_ratio": args.boundary_ratio,
               "device": args.device, "threads": args.threads, "warmup": args.warmup,
               "allow_unverified": args.allow_unverified_provenance}
    baseline = evaluate_model(args.model, splits[args.split], audit, backend=args.backend, **options)
    report = {"environment": {"platform": platform.platform(), "processor": platform.processor(),
                               "logical_cpus": os.cpu_count(), "threads": args.threads},
              "baseline": baseline}
    failed_gate = False
    if args.compare:
        options["threshold"] = baseline["threshold"]
        options["boundary_ratio"] = baseline["metrics"]["boundary_ratio"]
        comparison = evaluate_model(args.compare, splits[args.split], audit, backend=args.compare_backend, **options)
        delta = comparison["metrics"]["iou"] - baseline["metrics"]["iou"]
        exceeded = delta < -args.max_iou_drop - 1e-12
        failed_gate = exceeded and (bool(comparison.get("quantization")) or args.enforce_iou_drop)
        report["comparison"] = comparison
        report["difference"] = {"iou_delta_percentage_points": delta * 100,
                                "boundary_iou_delta_percentage_points": (comparison["metrics"]["boundary_iou"] - baseline["metrics"]["boundary_iou"]) * 100,
                                "maximum_allowed_iou_drop_percentage_points": args.max_iou_drop * 100,
                                "iou_drop_within_limit": not exceeded,
                                "qat_recommended_if_int8": exceeded,
                                "acceptance_gate_failed": failed_gate,
                                "inference_speedup": baseline["inference_latency"]["mean_ms"] / comparison["inference_latency"]["mean_ms"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved: {output.resolve()}")
    if failed_gate:
        print("Comparison exceeds the permitted IoU drop; INT8 requires QAT/recalibration before acceptance.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
