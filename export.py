"""Export either architecture to a static ONNX graph and check backend parity."""
import argparse
import inspect
import json
from pathlib import Path

import numpy as np

from backends import load_backend, sidecar_path, validate_output_path, validate_report_path, write_metadata


def parity_inputs(metadata, image_paths=None, count=5, seed=42):
    if image_paths:
        from data_tools import read_image
        from preprocess import prepare_input

        for path in image_paths:
            value, _ = prepare_input(read_image(path), input_width=metadata["input_width"],
                                     input_height=metadata["input_height"], roi_top=metadata["roi_top"],
                                     preprocess_version=metadata["preprocess_version"])
            yield value
    else:
        generator = np.random.default_rng(seed)
        shape = (1, 3, metadata["input_height"], metadata["input_width"])
        for _ in range(count):
            yield generator.random(shape, dtype=np.float32)


def compare_backends(reference, candidate, inputs, atol=1e-5, rtol=1e-4):
    """Numerical probability agreement; this is not a segmentation quality score."""
    errors, maxima, within_tolerance = [], [], True
    keys = ("input_width", "input_height", "roi_top", "preprocess_version", "architecture")
    if any(reference.metadata[k] != candidate.metadata[k] for k in keys):
        raise ValueError("Cannot compare backends with different model/preprocessing metadata")
    for value in inputs:
        expected, actual = reference.predict(value), candidate.predict(value)
        difference = np.abs(expected - actual)
        errors.append(float(difference.mean()))
        maxima.append(float(difference.max()))
        within_tolerance &= bool(np.allclose(actual, expected, atol=atol, rtol=rtol))
    if not errors:
        raise ValueError("Numerical comparison requires at least one input frame")
    return {"frames": len(errors), "max_absolute_error": max(maxima),
            "mean_absolute_error": float(np.mean(errors)), "within_tolerance": within_tolerance,
            "absolute_tolerance": atol, "relative_tolerance": rtol}


def export_checkpoint(checkpoint_path, output_path, check=False, threads=1,
                      image_paths=None, check_backend="onnxruntime", atol=1e-5, rtol=1e-4):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("ONNX export is optional: pip install -r requirements-onnx.txt") from exc
    import torch

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("Export output must use the .onnx extension")
    if output_path.resolve() == Path(checkpoint_path).resolve():
        raise ValueError("Output cannot overwrite the source checkpoint")
    validate_output_path(sidecar_path(output_path), checkpoint_path)
    backend = load_backend(checkpoint_path, threads=threads)
    metadata = dict(backend.metadata)
    # Networks return logits even if an invalid user checkpoint claimed otherwise.
    if metadata["output_kind"] != "logits":
        raise ValueError("PyTorch export requires a logits-output checkpoint")
    dummy = torch.zeros(1, 3, metadata["input_height"], metadata["input_width"], dtype=torch.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    options = {"input_names": ["image"], "output_names": ["logits"],
               "opset_version": 17, "export_params": True, "do_constant_folding": True}
    # torch>=2.1 remains supported; newer releases default to another exporter.
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(backend.model, dummy, str(output_path), **options)
    graph = onnx.load(str(output_path))
    onnx.helper.set_model_props(graph, {"ai_road_metadata": json.dumps(metadata, ensure_ascii=False)})
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output_path))
    write_metadata(output_path, metadata)
    report = {"architecture": metadata["architecture"],
              "parameters": sum(parameter.numel() for parameter in backend.model.parameters()),
              "input_shape": list(dummy.shape), "output": str(output_path.resolve()),
              "model_bytes": output_path.stat().st_size,
              "data": "image_frames" if image_paths else "synthetic_random_inputs"}
    if check:
        candidate = load_backend(output_path, backend=check_backend, threads=threads)
        report["parity"] = compare_backends(backend, candidate, parity_inputs(metadata, image_paths), atol, rtol)
        report["parity"]["backend"] = check_backend
        if not report["parity"]["within_tolerance"]:
            raise RuntimeError(f"FP32 backend parity check failed: {json.dumps(report['parity'])}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--check", action="store_true", help="Compare PyTorch with an exported backend")
    parser.add_argument("--check-backend", choices=("onnxruntime", "openvino"), default="onnxruntime")
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--check-images", nargs="+", help="Representative image paths for numerical parity")
    data.add_argument("--check-manifest", help="Compare every test frame in an audited manifest")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--report", help="Write numerical parity JSON; no accuracy/FPS claim is implied")
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("Tolerances must be nonnegative")
    validate_report_path(args.report, args.checkpoint, args.output, sidecar_path(args.output),
                         args.check_manifest, *(args.check_images or []))
    for destination in (args.output, sidecar_path(args.output)):
        validate_output_path(destination, args.checkpoint, args.check_manifest, *(args.check_images or []))
    paths = args.check_images
    if args.check_manifest:
        from data_tools import load_manifest
        splits, _ = load_manifest(args.check_manifest)
        paths = [sample["image"] for sample in splits["test"]]
    report = export_checkpoint(args.checkpoint, args.output,
                               check=args.check or bool(paths), threads=args.threads,
                               image_paths=paths, check_backend=args.check_backend,
                               atol=args.atol, rtol=args.rtol)
    if args.report:
        path = Path(args.report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
