"""Static QDQ INT8 calibration on 200-500 unique TRAIN frames only.

Use evaluate.py on the untouched test split for FP32/INT8 accuracy and latency.
Calibration drift reported here is numerical diagnostics, not test quality.
"""
import argparse
import hashlib
import json
import random
import tempfile
from pathlib import Path

from backends import load_backend, read_metadata, sidecar_path, validate_output_path, validate_report_path, write_metadata
from export import compare_backends, parity_inputs


def select_calibration_samples(manifest_path, count=200, seed=42):
    from data_tools import image_digest, load_manifest, read_image

    if not 200 <= count <= 500:
        raise ValueError("Static calibration requires 200-500 representative frames")
    splits, audit = load_manifest(manifest_path)
    rng = random.Random(seed)
    groups = {}
    for sample in splits["train"]:
        groups.setdefault(sample["group"], []).append(sample)
    for samples in groups.values():
        rng.shuffle(samples)
    names = sorted(groups)
    rng.shuffle(names)
    # Round-robin groups avoids using only one long trip when shorter trips exist.
    selected, hashes = [], set()
    while names and len(selected) < count:
        remaining = []
        for group in names:
            samples = groups[group]
            if not samples:
                continue
            sample = samples.pop()
            digest = image_digest(read_image(sample["image"]))
            if digest not in hashes:
                selected.append(sample)
                hashes.add(digest)
            if samples:
                remaining.append(group)
            if len(selected) == count:
                break
        names = remaining
    if len(selected) < count:
        raise ValueError(f"Need {count} unique training images for calibration, found {len(selected)}")
    provenance = {"split": "train", "count": len(selected), "seed": seed,
                  "groups": sorted({sample["group"] for sample in selected}),
                  "image_hashes": sorted(hashes), "dataset_fingerprint": audit["dataset_fingerprint"]}
    return selected, provenance


def quantize_model(model_path, manifest_path, output_path, samples=200, seed=42, threads=1):
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import (
            CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static,
        )
        from onnxruntime.quantization.shape_inference import quant_pre_process
    except ImportError as exc:
        raise RuntimeError("INT8 tools are optional: pip install -r requirements-onnx.txt") from exc

    model_path, output_path = Path(model_path), Path(output_path)
    if model_path.resolve() == output_path.resolve():
        raise ValueError("INT8 output must not overwrite the FP32 model")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("INT8 output must use the .onnx extension")
    for destination in (output_path, sidecar_path(output_path)):
        validate_output_path(destination, model_path, sidecar_path(model_path), manifest_path)
    metadata = read_metadata(model_path)
    if metadata.get("quantization"):
        raise ValueError("Input model is already quantized; provide the FP32 export")
    records, provenance = select_calibration_samples(manifest_path, samples, seed)
    paths = [sample["image"] for sample in records]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class FrameReader(CalibrationDataReader):
        def __init__(self, input_name):
            self.input_name = input_name
            self.rewind()

        def get_next(self):
            value = next(self.values, None)
            return None if value is None else {self.input_name: value}

        def rewind(self):
            self.values = iter(parity_inputs(metadata, paths))

    with tempfile.TemporaryDirectory(prefix="ai-road-quant-", dir=output_path.parent) as directory:
        prepared = Path(directory) / "prepared.onnx"
        # Static shapes are known; ORT recommends shape inference before quantization.
        quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=True)
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = threads
        session_options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(prepared), sess_options=session_options, providers=["CPUExecutionProvider"])
        reader = FrameReader(session.get_inputs()[0].name)
        del session
        quantize_static(str(prepared), str(output_path), reader,
                        quant_format=QuantFormat.QDQ, per_channel=True,
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                        op_types_to_quantize=["Conv"], calibrate_method=CalibrationMethod.MinMax,
                        calibration_providers=["CPUExecutionProvider"],
                        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True})
    metadata["quantization"] = {"format": "QDQ", "activation": "int8", "weights": "int8",
                                "per_channel": True, "operators": ["Conv"], "method": "MinMax"}
    metadata["calibration"] = provenance
    graph = onnx.load(str(output_path))
    properties = {entry.key: entry.value for entry in graph.metadata_props}
    properties["ai_road_metadata"] = json.dumps(metadata, ensure_ascii=False)
    onnx.helper.set_model_props(graph, properties)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output_path))
    write_metadata(output_path, metadata)
    reference = load_backend(model_path, backend="onnxruntime", threads=threads)
    candidate = load_backend(output_path, backend="onnxruntime", threads=threads)
    drift = compare_backends(reference, candidate, parity_inputs(metadata, paths))
    report = {"input": str(model_path.resolve()), "output": str(output_path.resolve()),
              "fp32_bytes": model_path.stat().st_size, "int8_bytes": output_path.stat().st_size,
              "calibration": {k: v for k, v in provenance.items() if k != "image_hashes"},
              "calibration_manifest_sha256": hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest(),
              "calibration_probability_drift": drift,
              "acceptance": "pending independent test IoU/Boundary IoU and target-CPU benchmark",
              "qat_gate": "If test Road IoU decreases by more than 0.02 absolute, reject PTQ and train with QAT."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="FP32 ONNX with its JSON sidecar")
    parser.add_argument("--manifest", required=True, help="Audited train/validation/test manifest")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--report")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    path = Path(args.report) if args.report else Path(str(args.output) + ".calibration.json")
    validate_report_path(path, args.model, sidecar_path(args.model), args.output,
                         sidecar_path(args.output), args.manifest)
    report = quantize_model(args.model, args.manifest, args.output, args.samples, args.seed, args.threads)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
