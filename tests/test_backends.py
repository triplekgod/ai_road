import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import torch

from backends import checkpoint_metadata, load_backend, read_metadata, sidecar_path, validate_report_path, write_metadata
from export import compare_backends, export_checkpoint, parity_inputs
from model import ARCHITECTURES, create_model
from quantize import quantize_model, select_calibration_samples


HAS_ONNX = bool(importlib.util.find_spec("onnx"))
HAS_ORT = bool(importlib.util.find_spec("onnxruntime"))
HAS_OPENVINO = bool(importlib.util.find_spec("openvino"))


def make_checkpoint(path, architecture="lite_road_net", width=256, height=144):
    torch.manual_seed(3)
    model = create_model(architecture)
    checkpoint = {"model": model.state_dict(), "architecture": architecture,
                  "input_width": width, "input_height": height, "roi_top": .2,
                  "preprocess_version": 1, "class_names": ["background", "road"],
                  "epoch": 0, "training_groups": ["trip-train"],
                  "validation_groups": ["trip-validation"], "dataset_fingerprint": "test-only"}
    torch.save(checkpoint, path)
    return checkpoint


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_parameter_counts_and_rectangular_logits(self):
        self.assertEqual(sum(p.numel() for p in create_model().parameters()), 19233)
        fast_count = sum(p.numel() for p in create_model("fast_scnn_lite").parameters())
        self.assertGreaterEqual(fast_count, 300000)
        self.assertLessEqual(fast_count, 1000000)
        for architecture in ARCHITECTURES:
            model = create_model(architecture).eval()
            with torch.inference_mode():
                for height, width in ((144, 256), (128, 224)):
                    result = model(torch.rand(1, 3, height, width))
                    self.assertEqual(tuple(result.shape), (1, 1, height, width))
                    self.assertTrue(torch.isfinite(result).all())

    def test_fast_model_backpropagates_at_batch_one(self):
        model = create_model("fast_scnn_lite").train()
        result = model(torch.rand(1, 3, 128, 224))
        torch.nn.functional.binary_cross_entropy_with_logits(result, torch.ones_like(result)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_architecture_validation(self):
        with self.assertRaisesRegex(ValueError, "Unknown architecture"):
            create_model("incorrect")


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.checkpoint = self.directory / "model.pth"
        make_checkpoint(self.checkpoint)

    def tearDown(self):
        self.temporary.cleanup()

    def test_probability_interface(self):
        backend = load_backend(self.checkpoint)
        value = np.random.default_rng(7).random((1, 3, 144, 256), dtype=np.float32)
        probability = backend.predict(value)
        with torch.inference_mode():
            expected = backend.model(torch.from_numpy(value)).sigmoid()[0, 0].numpy()
        np.testing.assert_allclose(probability, expected, atol=1e-7)
        self.assertEqual(probability.shape, (144, 256))
        self.assertEqual(probability.dtype, np.float32)
        for invalid in (value[0], value.astype(np.float64), value[:, :, :-1], value * np.nan):
            with self.assertRaises(ValueError):
                backend.predict(invalid)

    def test_legacy_metadata_preserves_input_contract(self):
        torch.save({"model": create_model().state_dict(), "image_size": 192}, self.checkpoint)
        backend = load_backend(self.checkpoint)
        self.assertEqual(backend.metadata["preprocess_version"], 0)
        self.assertEqual(backend.metadata["roi_top"], 0)
        self.assertEqual(backend.predict(np.zeros((1, 3, 192, 192), np.float32)).shape, (192, 192))
        with self.assertRaisesRegex(ValueError, "Legacy preprocessing"):
            checkpoint_metadata({"image_size": 192, "roi_top": .2})
        with self.assertRaisesRegex(ValueError, "lacks input"):
            checkpoint_metadata({"model": {}})
        with self.assertRaisesRegex(ValueError, "multiples of 16"):
            checkpoint_metadata({"input_width": 250, "input_height": 144})

    def test_mismatched_architecture_is_rejected(self):
        checkpoint = make_checkpoint(self.checkpoint)
        checkpoint["architecture"] = "fast_scnn_lite"
        torch.save(checkpoint, self.checkpoint)
        with self.assertRaisesRegex(ValueError, "Weights do not match architecture"):
            load_backend(self.checkpoint)

    def test_sidecar_preserves_provenance_and_detects_wrong_weights(self):
        artifact = self.directory / "model.onnx"
        artifact.write_bytes(b"sample artifact")
        metadata = checkpoint_metadata(make_checkpoint(self.checkpoint))
        write_metadata(artifact, metadata)
        restored = read_metadata(artifact)
        self.assertEqual(restored["training_groups"], ["trip-train"])
        self.assertEqual(restored["roi_top"], .2)
        artifact.write_bytes(b"different artifact")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            read_metadata(artifact)

    def test_optional_dependency_error_explains_installation(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-onnx.txt"):
                load_backend(self.checkpoint, "onnxruntime")
        with patch.dict("sys.modules", {"openvino": None}):
            with self.assertRaisesRegex(RuntimeError, "requirements-openvino.txt"):
                load_backend(self.checkpoint, "openvino")

    def test_report_must_not_overwrite_model_or_input(self):
        before = self.checkpoint.read_bytes()
        with self.assertRaisesRegex(ValueError, "Report path must differ"):
            validate_report_path(self.checkpoint.parent / "." / self.checkpoint.name, self.checkpoint)
        self.assertEqual(self.checkpoint.read_bytes(), before)
        validate_report_path(self.directory / "parity.json", self.checkpoint)

    @unittest.skipUnless(HAS_ONNX and HAS_ORT, "optional ONNX/ORT dependencies unavailable")
    def test_both_architectures_onnx_numerical_parity(self):
        for architecture in ARCHITECTURES:
            for width, height in ((256, 144), (224, 128)):
                with self.subTest(architecture=architecture, shape=(width, height)):
                    make_checkpoint(self.checkpoint, architecture, width, height)
                    output = self.directory / f"{architecture}-{width}.onnx"
                    report = export_checkpoint(self.checkpoint, output, check=True)
                    self.assertTrue(report["parity"]["within_tolerance"])
                    self.assertTrue(sidecar_path(output).is_file())
                    self.assertEqual(read_metadata(output)["training_groups"], ["trip-train"])
                    if HAS_OPENVINO:
                        reference = load_backend(self.checkpoint)
                        candidate = load_backend(output, "openvino")
                        result = compare_backends(reference, candidate, parity_inputs(reference.metadata, count=2))
                        self.assertTrue(result["within_tolerance"], result)


def make_calibration_manifest(directory, train_count=200):
    splits = {"train": [], "validation": [], "test": []}
    generator = np.random.default_rng(19)
    for split, count in (("train", train_count), ("validation", 1), ("test", 1)):
        for index in range(count):
            image = generator.integers(0, 256, (32, 64, 3), dtype=np.uint8)
            image_path = directory / f"{split}-{index}.png"
            mask_path = directory / f"{split}-{index}-mask.png"
            cv2.imwrite(str(image_path), image)
            cv2.imwrite(str(mask_path), np.zeros((32, 64), np.uint8))
            splits[split].append({"image": image_path.name, "mask": mask_path.name, "group": split})
    path = directory / "manifest.json"
    path.write_text(json.dumps({"version": 1, "splits": splits}), encoding="utf-8")
    return path


class CalibrationTests(unittest.TestCase):
    def test_calibration_count_and_non_test_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_calibration_manifest(directory)
            samples, provenance = select_calibration_samples(manifest, count=200)
            self.assertEqual(len(samples), 200)
            self.assertEqual({sample["group"] for sample in samples}, {"train"})
            self.assertEqual(provenance["split"], "train")
            self.assertEqual(len(provenance["image_hashes"]), 200)
            with self.assertRaisesRegex(ValueError, "200-500"):
                select_calibration_samples(manifest, count=100)
            with self.assertRaisesRegex(ValueError, "unique training images"):
                select_calibration_samples(manifest, count=201)

    @unittest.skipUnless(HAS_ONNX and HAS_ORT, "optional ONNX/ORT dependencies unavailable")
    def test_static_int8_synthetic_plumbing_not_quality(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = make_calibration_manifest(directory)
            checkpoint, exported, quantized = (directory / name for name in ("model.pth", "model.onnx", "int8.onnx"))
            make_checkpoint(checkpoint, width=64, height=32)
            export_checkpoint(checkpoint, exported)
            report = quantize_model(exported, manifest, quantized, samples=200)
            self.assertEqual(report["calibration"]["count"], 200)
            self.assertIn("pending", report["acceptance"])
            backend = load_backend(quantized, "onnxruntime")
            result = backend.predict(np.zeros((1, 3, 32, 64), np.float32))
            self.assertTrue(np.isfinite(result).all())
            self.assertTrue(np.all((result >= 0) & (result <= 1)))
            self.assertEqual(backend.metadata["quantization"]["format"], "QDQ")
            self.assertEqual(backend.metadata["calibration"]["groups"], ["train"])
            import onnx
            graph = onnx.load(quantized)
            operations = {node.op_type for node in graph.graph.node}
            self.assertIn("QuantizeLinear", operations)
            self.assertIn("DequantizeLinear", operations)
            embedded = json.loads({entry.key: entry.value for entry in graph.metadata_props}["ai_road_metadata"])
            self.assertEqual(embedded["calibration"]["groups"], ["train"])


if __name__ == "__main__":
    unittest.main()
