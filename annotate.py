"""Multiple additive/subtractive polygons with editable binary masks."""
import argparse
from pathlib import Path

import cv2
import numpy as np

WINDOW = "Road annotation"


class MaskEditor:
    def __init__(self, mask):
        self.mask = np.where(mask > 0, 255, 0).astype(np.uint8)
        self.history = []

    def _remember(self):
        self.history.append(self.mask.copy())
        self.history = self.history[-20:]

    def polygon(self, points, subtract=False):
        if len(points) < 3:
            return False
        self._remember()
        cv2.fillPoly(self.mask, [np.asarray(points, dtype=np.int32)], 0 if subtract else 255)
        return True

    def black(self):
        self._remember()
        self.mask.fill(0)

    def undo(self):
        if self.history:
            self.mask = self.history.pop()


class Annotator:
    def __init__(self, images_dir, masks_dir):
        self.images_dir, self.masks_dir = Path(images_dir), Path(masks_dir)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.files = sorted(self.images_dir.glob("*.bmp"))
        if not self.files:
            raise ValueError("No BMP files found")
        self.index = 0
        self.load(0)
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW, self.mouse)

    @property
    def path(self):
        return self.files[self.index]

    @property
    def mask_path(self):
        return self.masks_dir / f"{self.path.stem}_mask.bmp"

    def load(self, index):
        self.index = max(0, min(index, len(self.files)-1))
        self.image = cv2.imread(str(self.path))
        if self.image is None:
            raise ValueError(f"Cannot read {self.path}")
        mask = cv2.imread(str(self.mask_path), cv2.IMREAD_GRAYSCALE) if self.mask_path.exists() else np.zeros(self.image.shape[:2], np.uint8)
        if mask is None or mask.shape != self.image.shape[:2]:
            raise ValueError(f"Cannot edit invalid mask: {self.mask_path}")
        self.editor = MaskEditor(mask)
        self.points, self.subtract = [], False
        self.dirty, self.overwrite_armed = False, False
        self.pending_navigation = None
        self.notice = ""

    def changed(self):
        self.dirty, self.overwrite_armed = True, False
        self.pending_navigation = None
        self.notice = ''

    def mouse(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((x, y))
            self.changed()
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.commit()

    def commit(self):
        if self.editor.polygon(self.points, self.subtract):
            self.points = []
            self.changed()
        elif self.points:
            self.notice = "Need at least 3 points"

    def save(self):
        if self.points:
            if len(self.points) < 3:
                self.notice = "Finish polygon or press c to discard points"
                return
            self.commit()
        if self.mask_path.exists() and not self.overwrite_armed:
            self.overwrite_armed = True
            self.notice = "OVERWRITE? Press s again to confirm"
            return
        if not cv2.imwrite(str(self.mask_path), self.editor.mask):
            raise IOError(f"Cannot write {self.mask_path}")
        print(f"saved: {self.mask_path}")
        if self.index < len(self.files)-1:
            self.load(self.index+1)
        else:
            self.dirty, self.overwrite_armed = False, False
            self.notice = "Saved final frame"

    def navigate(self, index):
        if self.dirty and self.pending_navigation != index:
            self.pending_navigation = index
            self.notice = "Unsaved changes. Repeat navigation to discard"
            return
        self.load(index)

    def render(self):
        tint = self.image.copy()
        tint[self.editor.mask > 0] = (0, 220, 0)
        view = cv2.addWeighted(self.image, .6, tint, .4, 0)
        if self.points:
            points = np.asarray(self.points, np.int32)
            cv2.polylines(view, [points], len(points) > 2, (0, 0, 255) if self.subtract else (0, 255, 255), 2)
            for point in self.points:
                cv2.circle(view, tuple(point), 3, (255, 255, 255), -1)
        mode = "SUBTRACT" if self.subtract else "ADD"
        lines = [f"{self.index+1}/{len(self.files)} {self.path.name} {mode}",
                 "LMB:point RMB/Enter:polygon a:add d:subtract s:save",
                 "b:black u:undo c:discard points n/p:navigate q:quit", self.notice]
        for i, line in enumerate(lines):
            cv2.putText(view, line, (8, 22+i*21), cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 255, 255), 1)
        cv2.imshow(WINDOW, view)

    def run(self):
        try:
            while True:
                self.render()
                key = cv2.waitKey(20) & 0xFF
                if key in (ord("q"), 27):
                    if self.dirty and self.notice != "Unsaved changes. Press q again to quit":
                        self.notice = "Unsaved changes. Press q again to quit"
                    else:
                        break
                elif key == ord("n"):
                    self.navigate(self.index+1)
                elif key == ord("p"):
                    self.navigate(self.index-1)
                elif key == ord("u"):
                    if self.points:
                        self.points.pop()
                    else:
                        self.editor.undo()
                    self.changed()
                elif key == ord("c"):
                    self.points = []
                    self.changed()
                elif key in (ord("a"), ord("d")):
                    self.subtract = key == ord("d")
                elif key == ord("b"):
                    self.points = []
                    self.editor.black()
                    self.changed()
                elif key in (10, 13):
                    self.commit()
                elif key == ord("s"):
                    self.save()
        finally:
            cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("images_dir")
    p.add_argument("masks_dir")
    a = p.parse_args()
    Annotator(a.images_dir, a.masks_dir).run()


if __name__ == "__main__":
    main()
