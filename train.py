import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import RoadDataset, find_pairs
from road_model import RoadNet


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1.0)).mean()
    return bce + dice


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = dict(loss=0.0, intersection=0.0, union=0.0, pred=0.0, target=0.0, empty=0, empty_ok=0)
    for images, masks, _ in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        totals["loss"] += loss_fn(logits, masks).item() * len(images)
        pred = logits.sigmoid() > 0.5
        truth = masks > 0.5
        totals["intersection"] += (pred & truth).sum().item()
        totals["union"] += (pred | truth).sum().item()
        totals["pred"] += pred.sum().item()
        totals["target"] += truth.sum().item()
        empty = truth.flatten(1).sum(1) == 0
        totals["empty"] += empty.sum().item()
        totals["empty_ok"] += ((pred.flatten(1).float().mean(1) < 0.005) & empty).sum().item()
    i = totals["intersection"]
    return {
        "loss": totals["loss"] / len(loader.dataset),
        "iou": i / max(totals["union"], 1),
        "dice": 2 * i / max(totals["pred"] + totals["target"], 1),
        "no_road_acc": totals["empty_ok"] / totals["empty"] if totals["empty"] else float("nan"),
    }


def main():
    parser = argparse.ArgumentParser(description="Train road segmentation on one trip")
    parser.add_argument("data", type=Path, help="folder containing images/ and masks/")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/roadnet.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = find_pairs(args.data)
    if len(pairs) < 10:
        raise SystemExit(f"Need at least 10 labeled pairs, found {len(pairs)}")
    if args.batch < 2:
        raise SystemExit("DeepLabV3 training requires --batch 2 or larger because of BatchNorm")

    # Chronological split prevents almost-identical neighbouring frames leaking into validation.
    cut = max(1, int(len(pairs) * 0.8))
    train_pairs, val_pairs = pairs[:cut], pairs[cut:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(RoadDataset(train_pairs, True), args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(RoadDataset(val_pairs), args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = RoadNet(pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_iou = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={len(train_pairs)} val={len(val_pairs)} size=512x288")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = 0.0
        for images, masks, _ in train_loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = loss_fn(outputs["out"], masks) + 0.3 * loss_fn(outputs["aux"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)

        metrics = validate(model, val_loader, device)
        elapsed = time.perf_counter() - started
        print(f"epoch={epoch:03d} train_loss={train_loss/len(train_pairs):.4f} val_loss={metrics['loss']:.4f} IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} no_road_acc={metrics['no_road_acc']:.3f} sec={elapsed:.1f}")
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": metrics}, args.out)
            print(f"  saved={args.out} best_IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
