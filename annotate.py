"""Interactive polygon annotation of BMP frames into binary road masks."""
import argparse
from pathlib import Path
import cv2
import numpy as np


WINDOW = "Road annotation"


class Annotator:
    def __init__(self, images_dir, masks_dir):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir); self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.files = sorted(self.images_dir.glob("*.bmp"))
        if not self.files: raise ValueError("No BMP files found")
        self.index = 0; self.points = []; self.image = None
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.mouse)
        self.load(0)

    @property
    def path(self): return self.files[self.index]

    @property
    def mask_path(self): return self.masks_dir / f"{self.path.stem}_mask.bmp"

    def load(self, index):
        self.index = max(0, min(index, len(self.files) - 1))
        self.image = cv2.imread(str(self.path))
        if self.image is None: raise ValueError(f"Cannot read {self.path}")
        self.points = []

    def mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.save()

    def save(self):
        if len(self.points) < 3:
            print("Need at least 3 points to save a road polygon")
            return
        mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(self.points, dtype=np.int32)], 255)
        if not cv2.imwrite(str(self.mask_path), mask):
            raise IOError(f"Cannot write {self.mask_path}")
        print(f"saved: {self.mask_path.name}")

    def render(self):
        view = self.image.copy()
        if self.mask_path.exists():
            cv2.putText(view, "ALREADY MARKED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 220, 0), 2)
        if self.points:
            points = np.asarray(self.points, dtype=np.int32)
            cv2.polylines(view, [points], len(points) > 2, (0, 255, 255), 2)
            for point in self.points: cv2.circle(view, point, 4, (0, 0, 255), -1)
        status = f"{self.index + 1}/{len(self.files)}  {self.path.name}  points: {len(self.points)}"
        cv2.putText(view, status, (10, view.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        cv2.putText(view, "LMB:add  RMB:save  u:undo  c:clear  n/p:next/prev  q:quit", (10, view.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, .43, (255, 255, 255), 1)
        cv2.imshow(WINDOW, view)

    def run(self):
        while True:
            self.render()
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27: break
            if key == ord("n") or key == 83: self.load(self.index + 1)
            elif key == ord("p") or key == 81: self.load(self.index - 1)
            elif key == ord("u") and self.points: self.points.pop()
            elif key == ord("c"): self.points = []
            elif key == ord("s"): self.save()
        cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description="Mark road polygons on BMP frames")
    p.add_argument("images_dir"); p.add_argument("masks_dir")
    args = p.parse_args(); Annotator(args.images_dir, args.masks_dir).run()


if __name__ == "__main__": main()
