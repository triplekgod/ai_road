import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from data_tools import discover_pairs, inspect_pair, load_manifest, split_groups
from dataset import RoadDataset, augment_pair
from metrics import SegmentationMetrics, boundary_mask
from train import loss_fn, main as train_main, restore_training_state, run_epoch


def write_pair(root, name, color, road=False):
    image = np.full((48, 80, 3), color, np.uint8)
    image[0, 0] = [color, (color + 3) % 256, (color + 7) % 256]
    mask = np.zeros((48, 80), np.uint8)
    if road:
        mask[12:, 20:60] = 255
    image_path, mask_path = root / f"{name}.png", root / f"{name}_mask.png"
    assert cv2.imwrite(str(image_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    return {"image": str(image_path), "mask": str(mask_path), "group": name.split("_")[0]}


@pytest.fixture
def manifest(tmp_path):
    samples = []
    for group_index in range(3):
        for frame_index in range(2):
            samples.append(write_pair(tmp_path, f"trip{group_index}_{frame_index}",
                                      20 + 30 * group_index + 4 * frame_index, bool(frame_index)))
    document = split_groups(samples, seed=8)
    path = tmp_path / "splits.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_manifest_has_independent_groups_and_negative_frames(manifest):
    splits, audit = load_manifest(manifest)
    assert all(len(samples) == 2 for samples in splits.values())
    assert all(summary["negative_frames"] == 1 for summary in audit["splits"].values())
    groups = [set(summary["groups"]) for summary in audit["splits"].values()]
    assert not groups[0] & groups[1] and not groups[1] & groups[2]
    assert len(audit["dataset_fingerprint"]) == 64


def test_split_70_15_15_uses_groups_and_is_deterministic():
    samples = [{"image": str(i), "mask": str(i), "group": str(i // 4)} for i in range(80)]
    first = split_groups(samples, 42)
    assert first == split_groups(samples, 42)
    assert [len({s["group"] for s in first["splits"][split]}) for split in ("train", "validation", "test")] == [14, 3, 3]
    with pytest.raises(ValueError, match="three"):
        split_groups(samples[:8])


def test_manifest_rejects_group_leakage(manifest):
    document = json.loads(manifest.read_text())
    document["splits"]["validation"][0]["group"] = document["splits"]["train"][0]["group"]
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="Group leakage"):
        load_manifest(manifest)


def test_manifest_rejects_content_leakage_even_with_new_name(manifest):
    document = json.loads(manifest.read_text())
    train_image = document["splits"]["train"][0]["image"]
    test_image = document["splits"]["test"][0]["image"]
    cv2.imwrite(test_image, cv2.imread(train_image))
    with pytest.raises(ValueError, match="content leakage"):
        load_manifest(manifest)


def test_manifest_rejects_duplicate_paths(manifest):
    document = json.loads(manifest.read_text())
    document["splits"]["train"].append(document["splits"]["train"][0])
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="overlapping"):
        load_manifest(manifest)


def test_manifest_requires_no_road_examples_in_every_split(manifest):
    document = json.loads(manifest.read_text())
    for sample in document["splits"]["test"]:
        cv2.imwrite(sample["mask"], np.full((48, 80), 255, np.uint8))
    with pytest.raises(ValueError, match="no frames without road"):
        load_manifest(manifest)


@pytest.mark.parametrize("fault, expected", [
    ("missing", "Missing file"), ("corrupt", "Unreadable"), ("size", "size mismatch"),
    ("binary", "Non-binary"), ("channels", "single-channel"),
])
def test_pair_validation(tmp_path, fault, expected):
    sample = write_pair(tmp_path, "trip0_frame", 40)
    path = Path(sample["mask"])
    if fault == "missing":
        path.unlink()
    elif fault == "corrupt":
        path.write_bytes(b"broken image")
    elif fault == "size":
        cv2.imwrite(str(path), np.zeros((20, 20), np.uint8))
    elif fault == "binary":
        cv2.imwrite(str(path), np.full((48, 80), 127, np.uint8))
    else:
        cv2.imwrite(str(path), np.zeros((48, 80, 3), np.uint8))
    with pytest.raises(ValueError, match=expected):
        inspect_pair(sample["image"], sample["mask"])


def test_discovery_rejects_missing_and_orphan_masks(tmp_path):
    images, masks = tmp_path / "images", tmp_path / "masks"
    images.mkdir()
    masks.mkdir()
    cv2.imwrite(str(images / "a.bmp"), np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(ValueError, match="Missing mask"):
        discover_pairs(images, masks, "trip")
    cv2.imwrite(str(masks / "a_mask.bmp"), np.zeros((8, 8), np.uint8))
    cv2.imwrite(str(masks / "orphan_mask.bmp"), np.zeros((8, 8), np.uint8))
    with pytest.raises(ValueError, match="Orphan"):
        discover_pairs(images, masks, "trip")


def test_dataset_rectangular_preprocessing_and_binary_mask(manifest):
    splits, _ = load_manifest(manifest)
    dataset = RoadDataset(samples=splits["train"], input_width=64, input_height=32, roi_top=.20)
    image, mask = dataset[1]
    assert image.shape == (3, 32, 64)
    assert mask.shape == (1, 32, 64)
    assert image.dtype == mask.dtype == torch.float32
    assert set(mask.unique().tolist()) <= {0., 1.}
    assert image.min() >= 0 and image.max() <= 1


def test_all_augmentations_keep_negative_masks_empty_and_shapes():
    image = np.full((48, 80, 3), 127, np.uint8)
    negative = np.zeros((48, 80), np.uint8)
    for seed in range(30):
        np.random.seed(seed)
        augmented, mask = augment_pair(image, negative)
        assert augmented.shape == image.shape and augmented.dtype == np.uint8
        assert mask.shape == negative.shape and not mask.any()


def test_metrics_known_counts_and_negative_frame_rates():
    metric = SegmentationMetrics()
    metric.update(np.array([[1, 1], [0, 0]]), np.array([[1, 0], [1, 0]]))
    metric.update(np.array([[1, 0], [0, 0]]), np.zeros((2, 2)))
    scores = metric.compute()
    assert scores["iou"] == pytest.approx(1 / 4)
    assert scores["dice_f1"] == pytest.approx(2 / 5)
    assert scores["precision"] == pytest.approx(1 / 3)
    assert scores["recall"] == .5
    assert scores["no_road_pixel_fpr"] == .25
    assert scores["no_road_frame_fpr"] == 1
    assert scores["negative_frames"] == 1


def test_empty_metrics_and_boundary_border_definition():
    metric = SegmentationMetrics()
    empty = np.zeros((20, 30), np.uint8)
    metric.update(empty, empty)
    assert metric.compute()["iou"] == 1.
    assert metric.compute()["boundary_iou"] == 1.
    assert metric.compute()["no_road_pixel_fpr"] == 0.
    full = np.ones((20, 30), np.uint8)
    boundary = boundary_mask(full, .001)
    assert boundary[0].all() and boundary[-1].all()
    assert not boundary[2:-2, 2:-2].any()


def test_boundary_metric_detects_small_shift():
    truth = np.zeros((48, 80), np.uint8)
    truth[10:40, 20:60] = 1
    prediction = np.roll(truth, 2, axis=1)
    metric = SegmentationMetrics(.01)
    metric.update(prediction, truth)
    scores = metric.compute()
    assert 0 < scores["boundary_iou"] < scores["iou"] < 1


def test_loss_is_sample_weighted_with_partial_last_batch():
    logits = torch.tensor([-.7, .4, 2.]).reshape(3, 1, 1, 1).expand(3, 1, 4, 4)
    target = torch.tensor([0., 1., 0.]).reshape(3, 1, 1, 1).expand_as(logits)
    dataset = TensorDataset(logits, target)
    expected = loss_fn(logits, target).item()
    partial, _ = run_epoch(torch.nn.Identity(), DataLoader(dataset, batch_size=2), torch.device("cpu"))
    complete, _ = run_epoch(torch.nn.Identity(), DataLoader(dataset, batch_size=3), torch.device("cpu"))
    assert partial == pytest.approx(expected, abs=1e-6)
    assert complete == pytest.approx(expected, abs=1e-6)


def test_resume_restores_optimizer_epoch_but_finetune_does_not():
    original = torch.nn.Conv2d(1, 1, 1)
    optimizer = torch.optim.AdamW(original.parameters(), lr=.023)
    original(torch.ones(1, 1, 2, 2)).sum().backward()
    optimizer.step()
    checkpoint = {"model": original.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": 7, "best_validation_score": .8}
    resumed = torch.nn.Conv2d(1, 1, 1)
    restored_optimizer = torch.optim.AdamW(resumed.parameters(), lr=.001)
    assert restore_training_state(checkpoint, resumed, restored_optimizer, True) == (7, .8)
    assert restored_optimizer.param_groups[0]["lr"] == .023
    assert restored_optimizer.state
    fresh = torch.nn.Conv2d(1, 1, 1)
    fresh_optimizer = torch.optim.AdamW(fresh.parameters(), lr=.001)
    assert restore_training_state(checkpoint, fresh, fresh_optimizer, False) == (0, float("-inf"))
    assert not fresh_optimizer.state
    with pytest.raises(ValueError, match="finetune"):
        restore_training_state({"model": checkpoint["model"]}, fresh, fresh_optimizer, True)


def test_training_resume_and_independent_evaluation_smoke(manifest, tmp_path):
    from evaluate import evaluate_model
    output = tmp_path / "model.pth"
    common = ["--manifest", str(manifest), "--output", str(output),
              "--input-width", "64", "--input-height", "32", "--batch-size", "2", "--threads", "1"]
    train_main(common + ["--epochs", "1", "--threshold", ".61", "--boundary-ratio", ".03"])
    latest = tmp_path / "model.last.pth"
    first = torch.load(latest, weights_only=True)
    assert first["epoch"] == 1 and first["optimizer"]["state"]
    assert first["preprocess_version"] == 1 and first["provenance_complete"]
    train_main(common + ["--epochs", "2", "--resume", str(latest)])
    resumed = torch.load(latest, weights_only=True)
    assert resumed["epoch"] == 2
    assert resumed["validation_threshold"] == .61 and resumed["boundary_ratio"] == .03
    with pytest.raises(ValueError, match="match saved validation"):
        train_main(common + ["--epochs", "3", "--resume", str(latest), "--threshold", ".55"])
    splits, audit = load_manifest(manifest)
    report = evaluate_model(output, splits["test"], audit, warmup=0)
    assert report["provenance_verified"]
    assert report["metrics"]["frames"] == 2
    assert report["metrics"]["negative_frames"] == 1
    assert report["metric_space"] == "original_full_frame_including_area_above_roi"
    assert report["threshold"] == .61


def test_evaluation_rejects_known_leakage_even_with_legacy_override(manifest):
    from evaluate import verify_evaluation_provenance
    _, audit = load_manifest(manifest)
    contaminated = {"training_groups": audit["splits"]["test"]["groups"]}
    with pytest.raises(ValueError, match="leakage"):
        verify_evaluation_provenance(contaminated, audit, allow_unverified=True)
    with pytest.raises(ValueError, match="incomplete"):
        verify_evaluation_provenance({}, audit)
    assert not verify_evaluation_provenance({}, audit, allow_unverified=True)
    calibration = {"calibration": {"image_hashes": audit["splits"]["test"]["image_hashes"]}}
    with pytest.raises(ValueError, match="leakage"):
        verify_evaluation_provenance(calibration, audit, allow_unverified=True)


def test_finetuning_partial_provenance_remains_unverified(manifest):
    from train import _provenance
    _, audit = load_manifest(manifest)
    partial = {"training_groups": ["old_unknown_trip"], "provenance_complete": True}
    merged = _provenance(audit, partial)
    assert merged["provenance_complete"] is False


def test_int8_quality_gate_saves_report_and_returns_failure(manifest, tmp_path, monkeypatch):
    import evaluate
    calls = []

    def fake_evaluate(path, samples, audit, **kwargs):
        calls.append(kwargs)
        return {"metrics": {"iou": .8 if len(calls) == 1 else .7, "boundary_iou": .5, "boundary_ratio": .03},
                "threshold": .61, "quantization": None if len(calls) == 1 else {"weights": "int8"},
                "inference_latency": {"mean_ms": 10.}}

    monkeypatch.setattr(evaluate, "evaluate_model", fake_evaluate)
    output = tmp_path / "comparison.json"
    status = evaluate.main(["fp32.onnx", "--manifest", str(manifest), "--compare", "int8.onnx", "--output", str(output)])
    assert status == 2
    report = json.loads(output.read_text())
    assert report["difference"]["acceptance_gate_failed"]
    assert calls[1]["threshold"] == .61 and calls[1]["boundary_ratio"] == .03


@pytest.mark.parametrize("destination", ["model.onnx", "model.onnx.json", "other.onnx", "other.onnx.json", "splits.json"])
def test_evaluation_never_overwrites_inputs(manifest, tmp_path, monkeypatch, destination):
    import evaluate
    model, comparison = tmp_path / "model.onnx", tmp_path / "other.onnx"
    output = tmp_path / destination
    if not output.exists():
        output.write_bytes(b"keep input bytes")
    original = output.read_bytes()
    monkeypatch.setattr(evaluate, "load_manifest", lambda *_: pytest.fail("Validation must happen before loading data"))
    with pytest.raises(SystemExit) as failure:
        evaluate.main([str(model), "--compare", str(comparison), "--manifest", str(manifest), "--output", str(output)])
    assert failure.value.code == 2
    assert output.read_bytes() == original


@pytest.mark.parametrize("manifest_name", ["output.pth", "output.last.pth", "output.history.json", "output.pth.tmp"])
def test_training_output_files_cannot_replace_manifest(manifest, tmp_path, manifest_name):
    protected_manifest = tmp_path / manifest_name
    protected_manifest.write_bytes(manifest.read_bytes())
    original = protected_manifest.read_bytes()
    with pytest.raises(SystemExit) as failure:
        train_main(["--manifest", str(protected_manifest), "--output", str(tmp_path / "output.pth")])
    assert failure.value.code == 2
    assert protected_manifest.read_bytes() == original
