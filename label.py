import argparse
from pathlib import Path

import cv2
import numpy as np


class Labeler:
    def __init__(self, images_dir, masks_dir):
        self.images = [p for p in sorted(Path(images_dir).iterdir()) if p.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}]
        self.masks_dir = Path(masks_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.index = 0
        self.brush = 28
        self.mode = "brush"
        self.points = []
        self.history = []
        self.drawing = False
        self.value = 255

    def mask_file(self):
        return self.masks_dir / f"{self.images[self.index].stem}_mask.bmp"

    def load(self):
        self.image = cv2.imread(str(self.images[self.index]))
        path = self.mask_file()
        self.mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else np.zeros(self.image.shape[:2], np.uint8)
        self.points.clear()
        self.history.clear()

    def save(self):
        cv2.imwrite(str(self.mask_file()), self.mask)
        print(f"saved={self.mask_file()} road_fraction={(self.mask > 0).mean():.3f}")

    def mouse(self, event, x, y, flags, _):
        if self.mode == "polygon":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.points.append((x, y))
            elif event == cv2.EVENT_RBUTTONDOWN and self.points:
                self.points.pop()
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            self.history.append(self.mask.copy())
            self.drawing = True
            self.value = 255 if event == cv2.EVENT_LBUTTONDOWN else 0
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.drawing = False
        if self.drawing or event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            cv2.circle(self.mask, (x, y), self.brush, self.value, -1)

    def view(self):
        color = np.zeros_like(self.image)
        color[:, :, 1] = self.mask
        shown = cv2.addWeighted(self.image, 0.65, color, 0.35, 0)
        if len(self.points) > 1:
            cv2.polylines(shown, [np.asarray(self.points)], False, (0, 255, 255), 2)
        text = f"{self.index+1}/{len(self.images)} {self.mode} brush={self.brush} | LMB road RMB erase | B/P mode ENTER fill | S save N/PgDn next A/PgUp prev Z undo Q quit"
        cv2.rectangle(shown, (0, 0), (shown.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(shown, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return shown

    def run(self):
        if not self.images:
            raise SystemExit("No images found")
        cv2.namedWindow("Road labeler", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Road labeler", self.mouse)
        self.load()
        while True:
            cv2.imshow("Road labeler", self.view())
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self.save()
            elif key in (ord("n"), 83):
                self.save(); self.index = min(self.index + 1, len(self.images) - 1); self.load()
            elif key in (ord("a"), 81):
                self.save(); self.index = max(self.index - 1, 0); self.load()
            elif key == ord("b"):
                self.mode = "brush"; self.points.clear()
            elif key == ord("p"):
                self.mode = "polygon"; self.points.clear()
            elif key in (10, 13) and len(self.points) >= 3:
                self.history.append(self.mask.copy()); cv2.fillPoly(self.mask, [np.asarray(self.points)], 255); self.points.clear()
            elif key == ord("z") and self.history:
                self.mask = self.history.pop()
            elif key in (ord("+"), ord("=")):
                self.brush = min(150, self.brush + 4)
            elif key == ord("-"):
                self.brush = max(2, self.brush - 4)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Simple road mask editor")
    parser.add_argument("images")
    parser.add_argument("masks")
    args = parser.parse_args()
    Labeler(args.images, args.masks).run()


if __name__ == "__main__":
    main()

