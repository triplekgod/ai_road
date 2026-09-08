"""Auditable splits by source trip/video; no random frame-level split is provided."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SPLITS = ("train", "validation", "test")


def read_image(path, flags=cv2.IMREAD_COLOR):
    """imdecode supports non-ASCII Windows filenames, unlike some imread builds."""
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Missing file: {path}")
    try:
        result = cv2.imdecode(np.fromfile(path, dtype=np.uint8), flags)
    except (ValueError, cv2.error) as error:
        raise ValueError(f"Unreadable image: {path}") from error
    if result is None:
        raise ValueError(f"Unreadable image: {path}")
    return result


def image_digest(image):
    """Hash decoded pixels, so merely re-encoding a frame cannot hide leakage."""
    return hashlib.sha256(str(image.shape).encode() + image.tobytes()).hexdigest()


def inspect_pair(image_path, mask_path):
    image = read_image(image_path)
    raw_mask = read_image(mask_path, cv2.IMREAD_UNCHANGED)
    if raw_mask.ndim != 2:
        raise ValueError(f"Mask must be single-channel: {mask_path}")
    if image.shape[:2] != raw_mask.shape:
        raise ValueError(f"Image/mask size mismatch: {image_path} / {mask_path}")
    values = set(np.unique(raw_mask).tolist())
    if not (values <= {0, 1} or values <= {0, 255}):
        raise ValueError(f"Non-binary mask {mask_path}: {sorted(values)[:12]}")
    return {"negative": not bool(np.any(raw_mask)), "image_hash": image_digest(image),
            "mask_hash": image_digest((raw_mask > 0).astype(np.uint8)),
            "height": int(raw_mask.shape[0]), "width": int(raw_mask.shape[1])}


def discover_pairs(images_dir, masks_dir, group):
    images_dir, masks_dir = Path(images_dir).resolve(), Path(masks_dir).resolve()
    if not str(group).strip():
        raise ValueError("Every source requires an explicit non-empty group id")
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise ValueError(f"Image/mask directory missing: {images_dir} / {masks_dir}")
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    masks = sorted(p for p in masks_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No image files in {images_dir}")
    mask_index = {}
    for mask in masks:
        stem = mask.stem[:-5] if mask.stem.endswith("_mask") else mask.stem
        if stem in mask_index:
            raise ValueError(f"Ambiguous masks for {stem}: {masks_dir}")
        mask_index[stem] = mask
    seen, samples = set(), []
    for image in images:
        if image.stem in seen:
            raise ValueError(f"Duplicate image stem: {image.stem}")
        seen.add(image.stem)
        mask = mask_index.pop(image.stem, None)
        if mask is None:
            raise ValueError(f"Missing mask for {image}")
        inspect_pair(image, mask)
        samples.append({"image": str(image), "mask": str(mask), "group": str(group)})
    if mask_index:
        raise ValueError(f"Orphan masks without images: {list(mask_index)[:5]}")
    return samples


def _resolve_sample(record, base):
    if not isinstance(record, dict) or not all(k in record for k in ("image", "mask", "group")):
        raise ValueError("Each sample requires image, mask and group")
    if not isinstance(record["group"], str) or not record["group"].strip():
        raise ValueError("Group id must be a non-empty string identifying the source video/trip")
    return {"image": str((base / record["image"]).resolve()),
            "mask": str((base / record["mask"]).resolve()), "group": record["group"]}


def load_manifest(path, require_negatives=True):
    """Validate all splits before exposing any samples to training or evaluation."""
    path = Path(path).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    if document.get("version") != 1 or set(document.get("splits", {})) != set(SPLITS):
        raise ValueError("Manifest must have version=1 and train, validation, test splits")
    splits, report, group_owner, path_owner, hash_owner = {}, {}, {}, {}, {}
    fingerprint_records = []
    for split in SPLITS:
        records = document["splits"][split]
        if not isinstance(records, list) or not records:
            raise ValueError(f"Split {split} is empty")
        samples, negatives, groups, hashes = [], 0, set(), set()
        for record in records:
            sample = _resolve_sample(record, path.parent)
            group = sample["group"]
            if group in group_owner and group_owner[group] != split:
                raise ValueError(f"Group leakage: {group} occurs in {group_owner[group]} and {split}")
            group_owner[group] = split
            groups.add(group)
            for key in ("image", "mask"):
                value = sample[key]
                if value in path_owner:
                    raise ValueError(f"Duplicate or overlapping sample path: {value}")
                path_owner[value] = split
            info = inspect_pair(sample["image"], sample["mask"])
            digest = info["image_hash"]
            if digest in hash_owner and hash_owner[digest] != split:
                raise ValueError(f"Image content leakage between {hash_owner[digest]} and {split}: {sample['image']}")
            hash_owner[digest] = split
            hashes.add(digest)
            negatives += int(info["negative"])
            samples.append(sample)
            fingerprint_records.append((split, group, digest, info["mask_hash"]))
        if require_negatives and not negatives:
            raise ValueError(f"Split {split} has no frames without road; include hard-negative frames")
        splits[split] = samples
        report[split] = {"samples": len(samples), "groups": sorted(groups),
                         "negative_frames": negatives, "image_hashes": sorted(hashes)}
    fingerprint = hashlib.sha256(json.dumps(sorted(fingerprint_records)).encode()).hexdigest()
    return splits, {"version": 1, "dataset_fingerprint": fingerprint, "splits": report}


def split_groups(samples, seed=42):
    """70/15/15 by group count, rounded with at least one group per split."""
    groups = sorted({sample["group"] for sample in samples})
    if len(groups) < 3:
        raise ValueError("At least three independent videos/trips are required")
    random.Random(seed).shuffle(groups)
    counts = [max(1, int(len(groups) * ratio)) for ratio in (.70, .15, .15)]
    while sum(counts) > len(groups):
        index = max(range(3), key=lambda i: counts[i])
        counts[index] -= 1
    while sum(counts) < len(groups):
        index = max(range(3), key=lambda i: len(groups) * (.70, .15, .15)[i] - counts[i])
        counts[index] += 1
    assignment, start = {}, 0
    for split, count in zip(SPLITS, counts):
        for group in groups[start:start + count]:
            assignment[group] = split
        start += count
    return {"version": 1, "split_unit": "source_group", "seed": seed,
            "target_group_ratios": [0.70, 0.15, 0.15],
            "splits": {split: [s for s in samples if assignment[s["group"]] == split] for split in SPLITS}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    split_parser = commands.add_parser("split", help="Split explicitly named video/trip groups")
    split_parser.add_argument("--groups", required=True, help="JSON: {groups: [{group, images_dir, masks_dir}]}")
    split_parser.add_argument("--output", required=True)
    split_parser.add_argument("--seed", type=int, default=42)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("manifest")
    validate_parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "split":
        source = Path(args.groups).resolve()
        groups = json.loads(source.read_text(encoding="utf-8-sig")).get("groups", [])
        if not groups:
            parser.error("groups JSON must include a non-empty groups list")
        samples = []
        for group in groups:
            samples.extend(discover_pairs(source.parent / group["images_dir"],
                                          source.parent / group["masks_dir"], group["group"]))
        result = split_groups(samples, args.seed)
        output = Path(args.output).resolve()
        # Validate before replacing an existing user's manifest.
        for split in SPLITS:
            if not any(inspect_pair(s["image"], s["mask"])["negative"] for s in result["splits"][split]):
                raise ValueError(f"Generated {split} split has no negative frames. Add negatives to source groups or choose another seed.")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        _, report = load_manifest(temporary)
        temporary.replace(output)
        print(f"Saved validated group split: {output}")
    else:
        _, report = load_manifest(args.manifest)
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({split: {k: v for k, v in summary.items() if k != "image_hashes"}
                      for split, summary in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
