import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import ACCURATE_SIZE, FAST_SIZE, RoadDataset, find_pairs
from road_model import RoadNet


def _clock(seconds):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def show_progress(stage, current, total, loss, started):
    width = 28
    ratio = current / max(total, 1)
    filled = min(width, int(width * ratio))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.perf_counter() - started
    eta = elapsed / max(current, 1) * max(total - current, 0)
    print(
        f"\r{stage:<13} [{bar}] {ratio:6.1%} "
        f"batch={current}/{total} loss={loss:.4f} "
        f"time={_clock(elapsed)} eta={_clock(eta)}",
        end="\n" if current == total else "",
        flush=True,
    )


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = logits.sigmoid()
    intersection = (probability * target).sum((1, 2, 3))
    dice = 1.0 - ((2 * intersection + 1.0) / (probability.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1.0)).mean()
    return bce + dice


@torch.no_grad()
def validate(model, loader, device, epoch, epochs):
    model.eval()
    totals = dict(loss=0.0, intersection=0.0, union=0.0, pred=0.0, target=0.0, empty=0, empty_ok=0)
    started = time.perf_counter()
    seen = 0
    for batch_index, (images, masks, _) in enumerate(loader, 1):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        totals["loss"] += loss_fn(logits, masks).item() * len(images)
        seen += len(images)
        pred = logits.sigmoid() > 0.5
        truth = masks > 0.5
        totals["intersection"] += (pred & truth).sum().item()
        totals["union"] += (pred | truth).sum().item()
        totals["pred"] += pred.sum().item()
        totals["target"] += truth.sum().item()
        empty = truth.flatten(1).sum(1) == 0
        totals["empty"] += empty.sum().item()
        totals["empty_ok"] += ((pred.flatten(1).float().mean(1) < 0.005) & empty).sum().item()
        show_progress(f"VAL {epoch}/{epochs}", batch_index, len(loader), totals["loss"] / seen, started)
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
    parser.add_argument("--arch", choices=("fast", "accurate"), default="fast")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/roadnet.pt"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = find_pairs(args.data)
    if len(pairs) < 10:
        raise SystemExit(f"Need at least 10 labeled pairs, found {len(pairs)}")
    if args.arch == "accurate" and args.batch < 2:
        raise SystemExit("DeepLabV3 training requires --batch 2 or larger because of BatchNorm")

    # Chronological split prevents almost-identical neighbouring frames leaking into validation.
    cut = max(1, int(len(pairs) * 0.8))
    train_pairs, val_pairs = pairs[:cut], pairs[cut:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = FAST_SIZE if args.arch == "fast" else ACCURATE_SIZE
    train_loader = DataLoader(RoadDataset(train_pairs, True, image_size), args.batch, shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=args.arch == "accurate")
    val_loader = DataLoader(RoadDataset(val_pairs, image_size=image_size), args.batch, num_workers=args.workers, pin_memory=device.type == "cuda")
    model = RoadNet(args.arch, pretrained=not args.no_pretrained).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_iou = -1.0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    parameters = sum(p.numel() for p in model.parameters())
    print(f"device={device} arch={args.arch} params={parameters/1e6:.2f}M train={len(train_pairs)} val={len(val_pairs)} size={image_size[0]}x{image_size[1]} output={args.out}")

    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        train_loss = 0.0
        seen = 0
        for batch_index, (images, masks, _) in enumerate(train_loader, 1):
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                outputs = model(images)
                loss = loss_fn(outputs["out"], masks)
                if "aux" in outputs:
                    loss = loss + 0.3 * loss_fn(outputs["aux"], masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * len(images)
            seen += len(images)
            show_progress(f"TRAIN {epoch}/{args.epochs}", batch_index, len(train_loader), train_loss / seen, started)

        metrics = validate(model, val_loader, device, epoch, args.epochs)
        elapsed = time.perf_counter() - started
        print(f"EPOCH {epoch}/{args.epochs} train_loss={train_loss/seen:.4f} val_loss={metrics['loss']:.4f} IoU={metrics['iou']:.4f} Dice={metrics['dice']:.4f} no_road_acc={metrics['no_road_acc']:.3f} time={_clock(elapsed)}")
        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            torch.save({"model": model.state_dict(), "architecture": args.arch, "input_size": image_size, "epoch": epoch, "val": metrics}, args.out)
            print(f"  saved={args.out} best_IoU={best_iou:.4f}")


if __name__ == "__main__":
    main()
