import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset import RoadDataset
from model import LiteRoadNet


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = logits.sigmoid()
    dice = 1 - (2 * (prob * target).sum() + 1) / (prob.sum() + target.sum() + 1)
    return bce + dice


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images_dir"); p.add_argument("masks_dir")
    p.add_argument("--val-images-dir", help="separate validation frames; recommended")
    p.add_argument("--val-masks-dir", help="masks for --val-images-dir")
    p.add_argument("--epochs", type=int, default=50); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--size", type=int, default=192); p.add_argument("--output", default="lite_road_model.pth")
    p.add_argument("--resume", help="Checkpoint from a previous training run for fine-tuning")
    args = p.parse_args()
    if bool(args.val_images_dir) != bool(args.val_masks_dir):
        raise ValueError("Provide both --val-images-dir and --val-masks-dir")
    dataset = RoadDataset(args.images_dir, args.masks_dir, args.size, augment=True)
    if args.val_images_dir:
        train_set = dataset
        valid_set = RoadDataset(args.val_images_dir, args.val_masks_dir, args.size, augment=False)
    else:
        if len(dataset) < 2: raise ValueError("Need at least two labeled frames")
        n_train = max(1, int(.8 * len(dataset)))
        indices = torch.randperm(len(dataset)).tolist()
        # Validation must not receive random training augmentation.
        validation_data = RoadDataset(args.images_dir, args.masks_dir, args.size, augment=False)
        train_set = Subset(dataset, indices[:n_train])
        valid_set = Subset(validation_data, indices[n_train:])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LiteRoadNet().to(device)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if checkpoint.get("image_size", args.size) != args.size:
            raise ValueError("--size must match the checkpoint image_size")
        model.load_state_dict(checkpoint["model"])
        print(f"resumed from: {Path(args.resume).resolve()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if not args.resume else 1e-4, weight_decay=1e-4)
    best = float("inf")
    for epoch in range(args.epochs):
        model.train(); total = 0.
        for image, mask in DataLoader(train_set, args.batch_size, shuffle=True):
            image, mask = image.to(device), mask.to(device); optimizer.zero_grad()
            loss = loss_fn(model(image), mask); loss.backward(); optimizer.step(); total += loss.item()
        model.eval(); validation = 0.
        with torch.inference_mode():
            for image, mask in DataLoader(valid_set, args.batch_size):
                validation += loss_fn(model(image.to(device)), mask.to(device)).item()
        validation /= max(1, len(valid_set)); print(f"{epoch + 1}/{args.epochs}: train={total/max(1,len(train_set)):.4f} val={validation:.4f}")
        if validation < best:
            best = validation
            torch.save({"model": model.state_dict(), "image_size": args.size}, args.output)
            print(f"saved: {Path(args.output).resolve()}")


if __name__ == "__main__": main()
