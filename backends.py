"""Consistent CPU inference: normalized NCHW float32 -> HW road probability.

ONNX/OpenVINO are optional and imported only when selected. Exported artifacts
carry a .onnx.json (or .xml.json) sidecar to preserve preprocessing/provenance.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


BACKENDS = ("pytorch", "onnxruntime", "openvino")
ARCHITECTURES = ("lite_road_net", "fast_scnn_lite")
PROVENANCE_FIELDS = (
    "training_groups", "validation_groups", "training_image_hashes",
    "validation_image_hashes", "dataset_fingerprint", "provenance_complete",
    "validation_threshold", "boundary_ratio",
)


def checkpoint_metadata(checkpoint):
    """Normalize old square checkpoints without silently changing their input."""
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a mapping with model weights and metadata")
    rectangular = "input_width" in checkpoint or "input_height" in checkpoint
    if rectangular and not ("input_width" in checkpoint and "input_height" in checkpoint):
        raise ValueError("Both input_width and input_height are required")
    version = int(checkpoint.get("preprocess_version", 1 if rectangular else 0))
    if version not in (0, 1):
        raise ValueError(f"Unsupported preprocess_version: {version}")
    if rectangular:
        width, height = int(checkpoint["input_width"]), int(checkpoint["input_height"])
    elif "image_size" in checkpoint:
        width = height = int(checkpoint["image_size"])
    else:
        raise ValueError("Checkpoint lacks input dimensions (input_width/input_height or legacy image_size)")
    if width < 16 or height < 16:
        raise ValueError("Input width and height must be at least 16")
    if version == 1 and (width % 16 or height % 16):
        raise ValueError("Version 1 input dimensions must be multiples of 16")
    roi_top = float(checkpoint.get("roi_top", .2 if version == 1 else 0.0))
    if not 0.0 <= roi_top < 1.0:
        raise ValueError("roi_top must be in [0, 1)")
    if version == 0 and (roi_top != 0.0 or width != height):
        raise ValueError("Legacy preprocessing requires square input and roi_top=0")
    architecture = checkpoint.get("architecture", "lite_road_net")
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown checkpoint architecture: {architecture!r}")
    metadata = {
        "architecture": architecture, "input_width": width, "input_height": height,
        "roi_top": roi_top, "preprocess_version": version,
        "class_names": checkpoint.get("class_names", ["background", "road"]),
        "output_kind": checkpoint.get("output_kind", "logits"),
    }
    for key in (*PROVENANCE_FIELDS, "epoch", "best_validation_score", "quantization", "calibration"):
        if key in checkpoint:
            metadata[key] = checkpoint[key]
    if metadata["output_kind"] not in ("logits", "probabilities"):
        raise ValueError("output_kind must be logits or probabilities")
    return metadata


def sidecar_path(model_path):
    return Path(str(model_path) + ".json")


def validate_report_path(report_path, *protected_paths):
    """Keep diagnostic JSON from overwriting any model or source input."""
    if report_path is None:
        return
    resolved = Path(report_path).resolve()
    if any(path is not None and Path(path).resolve() == resolved for path in protected_paths):
        raise ValueError("Report path must differ from model, sidecar and input paths")


def validate_output_path(output_path, *protected_paths):
    resolved = Path(output_path).resolve()
    if any(path is not None and Path(path).resolve() == resolved for path in protected_paths):
        raise ValueError("Output path must differ from source model, sidecar and input paths")


def model_sha256(model_path):
    digest = hashlib.sha256()
    with Path(model_path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_metadata(model_path, metadata):
    data = dict(metadata)
    data["model_sha256"] = model_sha256(model_path)
    sidecar_path(model_path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_metadata(model_path):
    path = sidecar_path(model_path)
    if not path.is_file():
        raise ValueError(f"Missing preprocessing metadata: {path}. Export using export.py and keep its sidecar.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "model_sha256" in data and model_sha256(model_path) != data["model_sha256"]:
        raise ValueError("Model and metadata sidecar do not match (SHA256 mismatch)")
    return checkpoint_metadata(data)


def _validated_input(value, metadata):
    value = np.asarray(value)
    expected = (1, 3, metadata["input_height"], metadata["input_width"])
    if value.shape != expected:
        raise ValueError(f"Expected one NCHW input {expected}, got {value.shape}")
    if value.dtype != np.float32:
        raise ValueError("Backend input must have dtype float32")
    if not np.isfinite(value).all():
        raise ValueError("Backend input contains non-finite values")
    return np.ascontiguousarray(value)


def _probabilities(output, metadata):
    values = np.asarray(output, dtype=np.float32)
    expected = (1, 1, metadata["input_height"], metadata["input_width"])
    if values.shape != expected or not np.isfinite(values).all():
        raise ValueError(f"Invalid model output; expected finite logits/probabilities of shape {expected}")
    values = values[0, 0]
    if metadata["output_kind"] == "logits":
        # Stable sigmoid without overflow for large negative logits.
        exponential = np.exp(-np.abs(values))
        values = np.where(values >= 0, 1 / (1 + exponential), exponential / (1 + exponential))
    elif (values < 0).any() or (values > 1).any():
        raise ValueError("Model probability output is outside [0, 1]")
    return values.astype(np.float32, copy=False)


class PyTorchBackend:
    def __init__(self, model_path, device="cpu", threads=1):
        import torch
        from model import create_model

        torch.set_num_threads(threads)
        self.device = torch.device(device)
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        self.metadata = checkpoint_metadata(checkpoint)
        if "model" not in checkpoint:
            raise ValueError("Checkpoint has no 'model' state dictionary")
        self.model = create_model(self.metadata["architecture"])
        try:
            self.model.load_state_dict(checkpoint["model"], strict=True)
        except RuntimeError as exc:
            raise ValueError(f"Weights do not match architecture {self.metadata['architecture']!r}: {exc}") from exc
        self.model.to(self.device).eval()

    def predict(self, nchw):
        import torch

        value = _validated_input(nchw, self.metadata)
        with torch.inference_mode():
            output = self.model(torch.from_numpy(value).to(self.device)).cpu().numpy()
        return _probabilities(output, self.metadata)


class ONNXRuntimeBackend:
    def __init__(self, model_path, device="cpu", threads=1):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError("ONNX Runtime is optional: pip install -r requirements-onnx.txt") from exc
        if device.lower() != "cpu":
            raise ValueError("onnxruntime backend currently supports --device cpu")
        self.metadata = read_metadata(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        expected = [1, 3, self.metadata["input_height"], self.metadata["input_width"]]
        if len(inputs) != 1 or inputs[0].shape != expected or inputs[0].type != "tensor(float)":
            raise ValueError(f"ONNX artifact must have one static float32 NCHW input {expected}")
        self.input_name = inputs[0].name

    def predict(self, nchw):
        value = _validated_input(nchw, self.metadata)
        return _probabilities(self.session.run(None, {self.input_name: value})[0], self.metadata)


class OpenVINOBackend:
    def __init__(self, model_path, device="cpu", threads=1):
        try:
            import openvino as ov
        except ImportError as exc:
            raise RuntimeError("OpenVINO is optional: pip install -r requirements-openvino.txt") from exc
        if device.lower() != "cpu":
            raise ValueError("openvino backend currently supports --device cpu")
        self.metadata = read_metadata(model_path)
        core = ov.Core()
        model = core.read_model(str(model_path))
        expected = [1, 3, self.metadata["input_height"], self.metadata["input_width"]]
        if len(model.inputs) != 1 or list(model.input(0).shape) != expected:
            raise ValueError(f"OpenVINO artifact must have one static NCHW input {expected}")
        # Explicit FP32 avoids silently using BF16 in numerical parity checks.
        self.compiled = core.compile_model(model, "CPU", {
            "INFERENCE_NUM_THREADS": threads, "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f32",
        })
        self.request = self.compiled.create_infer_request()

    def predict(self, nchw):
        value = _validated_input(nchw, self.metadata)
        self.request.infer({0: value})
        return _probabilities(self.request.get_output_tensor(0).data, self.metadata).copy()


def load_backend(model_path, backend="pytorch", device="cpu", threads=1):
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer")
    implementations = {"pytorch": PyTorchBackend, "onnxruntime": ONNXRuntimeBackend, "openvino": OpenVINOBackend}
    if backend not in implementations:
        raise ValueError(f"Unknown backend {backend!r}; choose from {BACKENDS}")
    if not Path(model_path).is_file():
        raise FileNotFoundError(model_path)
    return implementations[backend](model_path, device=device, threads=threads)
