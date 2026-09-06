import argparse
import time
import cv2
import numpy as np
import torch
from model import LiteRoadNet
from road_geometry import Zones, draw_zone_outlines, primary_road, smooth_road_mask, zone_mask


class RoadAnalyzer:
    def __init__(self, checkpoint, threshold=.55, zones=Zones(), device=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        data = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.size = data.get("image_size", 192); self.threshold, self.zones = threshold, zones
        self.model = LiteRoadNet().to(self.device); self.model.load_state_dict(data["model"]); self.model.eval()

    @torch.inference_mode()
    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(self.device)
        prob = self.model(tensor).sigmoid()[0, 0].cpu().numpy()
        raw = cv2.resize((prob >= self.threshold).astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
        road = smooth_road_mask(primary_road(raw, min_area=max(300, w * h // 700)))
        overlay = np.zeros_like(frame); overlay[road == 0] = (0, 0, 255)  # off-road red
        zones = draw_zone_outlines(zone_mask(road, self.zones), road)
        overlay[road > 0] = zones[road > 0]
        return cv2.addWeighted(frame, .55, overlay, .45, 0), road


def main():
    p = argparse.ArgumentParser(description="Fast quarry-road segmentation")
    p.add_argument("video"); p.add_argument("model"); p.add_argument("--output")
    p.add_argument("--threshold", type=float, default=.55); p.add_argument("--left", type=float, default=.15)
    p.add_argument("--center", type=float, default=.70); p.add_argument("--right", type=float, default=.15)
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args(); analyzer = RoadAnalyzer(a.model, a.threshold, Zones(a.left, a.center, a.right))
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened(): raise FileNotFoundError(a.video)
    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        raise ValueError("Cannot determine source video FPS; output would not preserve it")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(a.output, cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (w, h)) if a.output else None
    if out is not None and not out.isOpened():
        cap.release()
        raise IOError(f"Cannot create output video: {a.output}")
    print(f"Source: {w}x{h}, {source_fps:.3f} FPS. Output FPS is preserved.")
    frames = 0; start = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok: break
        result, _ = analyzer.analyze(frame); frames += 1
        processing_fps = frames / (time.perf_counter() - start)
        cv2.putText(result, f"Processing: {processing_fps:.1f} FPS", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .7, (255,255,255), 2)
        if out: out.write(result)
        if not a.no_display:
            cv2.imshow("Quarry road", result)
            if cv2.waitKey(1) & 0xFF == ord("q"): break
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__": main()
