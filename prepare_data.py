import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Extract a reproducible subset from the supplied archive")
    parser.add_argument("zip", type=Path)
    parser.add_argument("--out", type=Path, default=Path("dataset"))
    parser.add_argument("--trip", default="trip01")
    parser.add_argument("--limit", type=int, default=0, help="0 extracts all; otherwise sample evenly")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as archive:
        prefix = f"{args.trip}/images/"
        images = sorted(n for n in archive.namelist() if n.startswith(prefix) and n.lower().endswith(".bmp"))
        pairs = []
        empty_masks = []
        for image_name in images:
            stem = Path(image_name).stem
            mask_name = f"{args.trip}/masks/{stem}_mask.bmp"
            if mask_name not in archive.namelist():
                continue
            with archive.open(mask_name) as stream:
                mask = np.asarray(Image.open(stream).convert("L"))
            is_empty = (mask > 127).mean() < 0.002
            pairs.append((image_name, mask_name, is_empty))

        if args.limit and len(pairs) > args.limit:
            ids = np.linspace(0, len(pairs) - 1, args.limit).round().astype(int)
            pairs = [pairs[i] for i in ids]

        target = args.out / args.trip
        for folder in (target / "images", target / "masks"):
            folder.mkdir(parents=True, exist_ok=True)
        empty_masks = [mask_name for _, mask_name, is_empty in pairs if is_empty]
        for image_name, mask_name, _ in pairs:
            for name, folder in ((image_name, "images"), (mask_name, "masks")):
                destination = target / folder / Path(name).name
                with archive.open(name) as source, destination.open("wb") as output:
                    output.write(source.read())

    report = {"trip": args.trip, "pairs": len(pairs), "empty_no_road_masks": empty_masks}
    (target / "extract.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
