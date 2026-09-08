"""Train with explicit source-group splits, validation metrics and resumable state."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data_tools import load_manifest
from dataset import RoadDataset
from metrics import SegmentationMetrics
from model import create_model


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").flatten(1).mean(1)
    prob, truth = logits.sigmoid().flatten(1), target.flatten(1)
    dice = 1 - (2 * (prob * truth).sum(1) + 1) / (prob.sum(1) + truth.sum(1) + 1)
    return (bce + dice).mean()


def run_epoch(model, loader, device, optimizer=None, threshold=.55, boundary_ratio=.02):
    training = optimizer is not None
    model.train(training)
    total, count = 0., 0
    metrics = SegmentationMetrics(boundary_ratio)
    with torch.set_grad_enabled(training):
        for image, target in loader:
            image, target = image.to(device), target.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(image)
            loss = loss_fn(logits, target)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect data and learning rate")
            if training:
                loss.backward()
                optimizer.step()
            total += float(loss.detach()) * image.shape[0]
            count += image.shape[0]
            if not training:
                metrics.update(logits.detach().sigmoid().cpu().numpy() >= threshold, target.cpu().numpy() > .5)
    if not count:
        raise ValueError("Empty dataloader")
    return total / count, metrics.compute() if not training else None


def restore_training_state(checkpoint, model, optimizer, resume=False):
    model.load_state_dict(checkpoint["model"])
    if not resume:
        return 0, float("-inf")
    if any(key not in checkpoint for key in ("optimizer", "epoch", "best_validation_score")):
        raise ValueError("Checkpoint has no full training state; use --finetune to load only weights")
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["epoch"]), float(checkpoint["best_validation_score"])


def _save_checkpoint(checkpoint, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def _provenance(audit, prior=None):
    prior = prior or {}
    current = audit["splits"]
    for split in ("validation", "test"):
        forbidden_groups = set(prior.get("training_groups", []))
        forbidden_hashes = set(prior.get("training_image_hashes", []))
        if split == "test":
            forbidden_groups.update(prior.get("validation_groups", []))
            forbidden_hashes.update(prior.get("validation_image_hashes", []))
            forbidden_groups.update((prior.get("calibration") or {}).get("groups", []))
            forbidden_hashes.update((prior.get("calibration") or {}).get("image_hashes", []))
        if forbidden_groups.intersection(current[split]["groups"]) or forbidden_hashes.intersection(current[split]["image_hashes"]):
            raise ValueError(f"Pretrained checkpoint provenance overlaps current {split} split")
    required = ("training_groups", "training_image_hashes", "validation_groups", "validation_image_hashes")
    complete = all(key in prior for key in required) and prior.get("provenance_complete", True)
    fields = {"dataset_fingerprint": audit["dataset_fingerprint"],
              "provenance_complete": not prior or bool(complete)}
    for prefix, split in (("training", "train"), ("validation", "validation")):
        for key in ("groups", "image_hashes"):
            name = f"{prefix}_{key}"
            fields[name] = sorted(set(prior.get(name, [])).union(current[split][key]))
    return fields


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Validated train/validation/test group manifest")
    parser.add_argument("--architecture", choices=("lite_road_net", "fast_scnn_lite"))
    parser.add_argument("--input-width", type=int)
    parser.add_argument("--input-height", type=int)
    parser.add_argument("--roi-top", type=float)
    parser.add_argument("--epochs", type=int, default=50, help="Total completed epochs, including --resume epochs")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--threshold", type=float, help="Choose using validation only; default: checkpoint value or .55")
    parser.add_argument("--boundary-ratio", type=float, help="Default: checkpoint value or .02")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="models/lite_road_model.pth")
    load = parser.add_mutually_exclusive_group()
    load.add_argument("--resume", help="Restore model, optimizer, completed epoch and best validation IoU")
    load.add_argument("--finetune", help="Load weights and start a new optimizer/epoch counter")
    args = parser.parse_args(argv)
    output = Path(args.output)
    last_path = output.with_name(output.stem + ".last" + output.suffix)
    history_path = output.with_suffix(".history.json")
    destinations = (output, last_path, history_path, output.with_suffix(output.suffix + ".tmp"),
                    last_path.with_suffix(last_path.suffix + ".tmp"))
    if Path(args.manifest).resolve() in {path.resolve() for path in destinations}:
        parser.error("Checkpoint, history and temporary outputs must not overwrite the dataset manifest")
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.workers < 0 or args.lr <= 0:
        parser.error("epochs, batch-size, threads and lr must be positive; workers must be non-negative")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    splits, audit = load_manifest(args.manifest)
    checkpoint_path = args.resume or args.finetune
    prior = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    for key, checkpoint_key, default in (("threshold", "validation_threshold", .55), ("boundary_ratio", "boundary_ratio", .02)):
        requested, saved = getattr(args, key), (prior or {}).get(checkpoint_key, default)
        if args.resume and requested is not None and requested != saved:
            raise ValueError(f"--{key.replace('_', '-')} must match saved validation metrics when resuming; use --finetune to change it")
        setattr(args, key, saved if requested is None else requested)
    if not 0 < args.threshold < 1 or not 0 < args.boundary_ratio <= 1:
        parser.error("threshold must be in (0,1), boundary-ratio in (0,1]")
    config = {}
    defaults = {"architecture": "lite_road_net", "input_width": 256, "input_height": 144, "roi_top": .20}
    for key, default in defaults.items():
        requested = getattr(args, key)
        saved = (prior or {}).get(key, default)
        if args.resume and requested is not None and requested != saved:
            raise ValueError(f"--{key.replace('_', '-')} must match the checkpoint when resuming")
        config[key] = saved if requested is None else requested
    if args.resume and prior.get("preprocess_version") != 1:
        raise ValueError("Resume requires a version-1 checkpoint; use --finetune for legacy weights")
    if args.resume and prior.get("dataset_fingerprint") != audit["dataset_fingerprint"]:
        raise ValueError("Resume manifest differs from saved training data; use --finetune for a new dataset")
    provenance = _provenance(audit, prior)
    dataset_config = {key: config[key] for key in ("input_width", "input_height", "roi_top")}
    train_set = RoadDataset(samples=splits["train"], augment=True, validate=False, **dataset_config)
    valid_set = RoadDataset(samples=splits["validation"], augment=False, validate=False, **dataset_config)
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True, num_workers=args.workers)
    valid_loader = DataLoader(valid_set, args.batch_size, num_workers=args.workers)
    device = torch.device(args.device)
    model = create_model(config["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best = restore_training_state(prior, model, optimizer, bool(args.resume)) if prior else (0, float("-inf"))
    if start >= args.epochs:
        raise ValueError(f"Checkpoint already completed {start} epochs; --epochs must be greater")
    history = []
    if args.resume and history_path.is_file():
        history = json.loads(history_path.read_text(encoding="utf-8")).get("epochs", [])
        history = [record for record in history if record["epoch"] <= start]
    for epoch in range(start, args.epochs):
        training_loss, _ = run_epoch(model, train_loader, device, optimizer)
        validation_loss, scores = run_epoch(model, valid_loader, device, threshold=args.threshold, boundary_ratio=args.boundary_ratio)
        improved = scores["iou"] > best
        best = max(best, scores["iou"])
        saved = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                 "best_validation_score": best, "class_names": ["background", "road"], "preprocess_version": 1,
                 "validation_threshold": args.threshold, "boundary_ratio": args.boundary_ratio,
                 **config, **provenance}
        _save_checkpoint(saved, last_path)
        if improved:
            _save_checkpoint(saved, output)
        history.append({"epoch": epoch + 1, "train_loss": training_loss, "validation_loss": validation_loss,
                        "validation": scores, "saved_best": improved})
        history_path.write_text(json.dumps({"metric_space": "network_input_canvas", "manifest": str(Path(args.manifest).resolve()),
                                           "audit": audit, "epochs": history}, indent=2), encoding="utf-8")
        print(f"{epoch + 1}/{args.epochs}: train={training_loss:.5f} val={validation_loss:.5f} "
              f"IoU={scores['iou']:.4f} Dice={scores['dice_f1']:.4f} BoundaryIoU={scores['boundary_iou']:.4f} "
              f"{'saved best' if improved else ''}")
    print(f"Latest resumable checkpoint: {last_path.resolve()}")


if __name__ == "__main__":
    main()
