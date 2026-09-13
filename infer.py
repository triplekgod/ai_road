import argparse
import time
from collections import deque
import cv2
import numpy as np
import torch
from model import LiteRoadNet
from road_geometry import Zones, draw_zone_outlines, primary_road, smooth_road_mask
from corridor_tracker import CorridorConfig, CorridorTracker


class RoadAnalyzer:
    def __init__(self, checkpoint, threshold=.55, zones=Zones(), temporal_window=5, min_confirmed_frames=4, device=None, corridor_config=None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        data = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.size = data.get("image_size", 192); self.threshold, self.zones = threshold, zones
        self.model = LiteRoadNet().to(self.device); self.model.load_state_dict(data["model"]); self.model.eval()
        if not 1 <= min_confirmed_frames <= temporal_window:
            raise ValueError("min_confirmed_frames must be between 1 and temporal_window")
        self.temporal_window = temporal_window
        self.min_confirmed_frames = min_confirmed_frames
        self.mask_history = deque(maxlen=temporal_window)
        self.tracker = CorridorTracker(self.zones, corridor_config)

    def stable_mask(self, current):
        """Keep a pixel only if it is confirmed by recent frames and current."""
        self.mask_history.append(current > 0)
        votes = np.sum(self.mask_history, axis=0)
        return np.where((current > 0) & (votes >= self.min_confirmed_frames), 255, 0).astype(np.uint8)

    @torch.inference_mode()
    def analyze(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        small = cv2.resize(rgb, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(small).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(self.device)
        prob = self.model(tensor).sigmoid()[0, 0].cpu().numpy()
        # Temporal voting happens at model resolution: it is inexpensive and
        # rejects short-lived false detections without changing output FPS.
        stable_small = self.stable_mask((prob >= self.threshold).astype(np.uint8) * 255)
        road_small = smooth_road_mask(primary_road(stable_small, min_area=max(8, self.size * self.size // 700)))
        tracking = self.tracker.update(road_small, cv2.cvtColor(small, cv2.COLOR_RGB2GRAY))
        road = cv2.resize(road_small, (w, h), interpolation=cv2.INTER_NEAREST)
        overlay = np.zeros_like(frame); overlay[road == 0] = (0, 0, 255)  # off-road red
        zones = cv2.resize(tracking.colors, (w, h), interpolation=cv2.INTER_NEAREST)
        zones = draw_zone_outlines(zones, road)
        overlay[road > 0] = zones[road > 0]
        result = cv2.addWeighted(frame, .55, overlay, .45, 0)
        cv2.putText(result, f"CORRIDOR {tracking.confidence:.0%} | branches: {tracking.branch_count}", (12, 58), cv2.FONT_HERSHEY_SIMPLEX, .55, (255,255,255), 2)
        return result, road


def main():
    p = argparse.ArgumentParser(description="Fast quarry-road segmentation")
    p.add_argument("video"); p.add_argument("model"); p.add_argument("--output")
    p.add_argument("--threshold", type=float, default=.55); p.add_argument("--left", type=float, default=.15)
    p.add_argument("--center", type=float, default=.70); p.add_argument("--right", type=float, default=.15)
    p.add_argument("--temporal-window", type=int, default=5, help="number of recent masks to compare")
    p.add_argument("--min-confirmed-frames", type=int, default=4, help="votes required for a road pixel")
    p.add_argument("--anchor-x", type=float, default=.50, help="vehicle center X as a fraction of frame width")
    p.add_argument("--anchor-y", type=float, default=.88, help="vehicle point Y as a fraction of frame height")
    p.add_argument("--branch-confirm-frames", type=int, default=8)
    p.add_argument("--branch-hold-frames", type=int, default=12)
    p.add_argument("--no-display", action="store_true")
    a = p.parse_args()
    tracker_config = CorridorConfig(anchor_x=a.anchor_x, anchor_y=a.anchor_y,
                                    branch_confirm_frames=a.branch_confirm_frames,
                                    branch_hold_frames=a.branch_hold_frames)
    analyzer = RoadAnalyzer(a.model, a.threshold, Zones(a.left, a.center, a.right), a.temporal_window, a.min_confirmed_frames, corridor_config=tracker_config)
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
