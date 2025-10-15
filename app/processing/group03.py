# processing/group03.py

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from collections import defaultdict
import time


class Group03Processor:
    # Initialize models and tracking only once
    model = YOLO("yolo11n-seg.pt")  # YOLOv11 segmentation model
    tracker = DeepSort(max_age=30)
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorbike"]

    # Define lane regions (adjust for your video)
    lanes = {
        "Lane 1": [(100, 700), (300, 400)],
        "Lane 2": [(350, 700), (600, 400)]
    }

    vehicle_counts = defaultdict(int)
    lane_counts = defaultdict(int)
    report_data = []
    last_update = time.time()

    @staticmethod
    def crossed_lane(x, y, lane_line):
        """Check if centroid crossed the lane line."""
        (x1, y1), (x2, y2) = lane_line
        return y2 < y < y1  # simple vertical check

    @staticmethod
    def process(frame):
        """Process a single frame for Flask streaming."""
        results = Group03Processor.model(frame, verbose=False)
        detections = []

        # === Collect detections from YOLO ===
        for result in results:
            boxes = result.boxes
            masks = result.masks
            names = Group03Processor.model.names

            if boxes is None:
                continue

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            cls_name = names[cls_id]

            if cls_name not in Group03Processor.VEHICLE_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            detections.append(([x1, y1, x2, y2], conf, cls_name))

            # --- FIXED MASK OVERLAY ---
            if masks is not None:
                mask = masks.data[i].cpu().numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask = (mask > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(frame, dtype=np.uint8)
                colored_mask[mask == 1] = (0, 255, 0)
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # === DeepSort Tracking ===
        tracks = Group03Processor.tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            cls_name = track.get_det_class()

            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(frame, f"{cls_name} {track_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

            # === Lane crossing check ===
            for lane_name, lane_line in Group03Processor.lanes.items():
                if Group03Processor.crossed_lane(cx, cy, lane_line):
                    Group03Processor.vehicle_counts[cls_name] += 1
                    Group03Processor.lane_counts[lane_name] += 1

        # === Draw lanes ===
        for lane_name, (p1, p2) in Group03Processor.lanes.items():
            cv2.line(frame, p1, p2, (255, 255, 255), 3)
            cv2.putText(frame, lane_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # === Display count overlay ===
        y_offset = 30
        for v_type, count in Group03Processor.vehicle_counts.items():
            cv2.putText(frame, f"{v_type}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            y_offset += 25

        for lane_name, count in Group03Processor.lane_counts.items():
            cv2.putText(frame, f"{lane_name}: {count}", (200, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 25

        return frame
