# group01.py
import cv2
from ultralytics import YOLO
import numpy as np
import matplotlib
matplotlib.use('Agg')   # use non-interactive backend for server environments
import matplotlib.pyplot as plt
import io

# Load models (path names same as your original)
car_model = YOLO("yolov8n.pt")          # Detect cars, trucks, bikes, etc.
lot_model = YOLO("yolov8n-seg.pt")      # Parking lot segmentation model (pretrained)


class Group01Processor:
    @staticmethod
    def _classify_vehicle(label, box_xyxy):
        """
        Heuristic classification into Sedan, Compact, SUV/Truck, Van, Bike
        label: string from model (lowercase)
        box_xyxy: (x1,y1,x2,y2)
        """
        x1, y1, x2, y2 = box_xyxy
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        aspect = width / height

        label = label.lower()
        if label in ('motorbike', 'motorcycle', 'bicycle', 'bike'):
            return "Bike"
        if label in ('truck', 'bus'):
            return "SUV/Truck"
        # Heuristic for van: wide box but not as long as truck and moderate height
        if label == 'car':
            if aspect > 1.8 and width > 160:
                return "Van"
            if height > 220 or label == 'truck':
                return "SUV/Truck"
            if aspect > 1.5:
                return "Sedan"
            return "Compact"
        # fallback
        return "Unknown"

    @staticmethod
    def _draw_stats_chart(counts, width, height):
        """
        Draw a vertical bar chart with matplotlib and return it as a BGR OpenCV image.
        counts: dict of label -> int
        width, height: desired pixel size of chart image
        """
        # Prepare data
        categories = list(counts.keys())
        values = [counts[c] for c in categories]

        # Create figure
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        # Draw bars
        ax.barh(categories, values)
        ax.set_xlim(0, max(1, max(values) * 1.2))
        ax.set_xlabel("Count")
        ax.set_title("Detected Vehicles")
        # Show counts to the right of bars
        for i, v in enumerate(values):
            ax.text(v + max(1, max(values) * 0.02), i, str(v), va='center')

        plt.tight_layout()

        # Convert figure to image (RGBA), then to BGR
        buf = io.BytesIO()
        fig.canvas.print_png(buf)
        buf.seek(0)
        chart_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        chart_img = cv2.imdecode(chart_arr, cv2.IMREAD_UNCHANGED)  # RGBA possibly
        plt.close(fig)
        if chart_img is None:
            # fallback: blank image
            chart_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # If RGBA convert to BGR
            if chart_img.shape[2] == 4:
                alpha = chart_img[:, :, 3] / 255.0
                rgb = chart_img[:, :, :3].astype(np.float32)
                background = np.ones_like(rgb) * 255.0
                comp = rgb * alpha[:, :, None] + background * (1 - alpha[:, :, None])
                chart_img = comp.astype(np.uint8)
            # Resize to requested size
            chart_img = cv2.resize(chart_img, (width, height))

        # Ensure 3-channel BGR
        if chart_img.ndim == 2:
            chart_img = cv2.cvtColor(chart_img, cv2.COLOR_GRAY2BGR)
        elif chart_img.shape[2] == 4:
            chart_img = chart_img[:, :, :3]

        return chart_img

    @staticmethod
    def process(frame):
        """
        Detect cars & bikes, draw bounding boxes, classify vehicle types,
        use segmentation to find empty vs occupied parking lots,
        compose a sidebar bar-chart and return the combined frame.
        """

        frame_h, frame_w = frame.shape[:2]

        # ----------------------------
        # 1. Detect vehicles (stream=True for speed)
        # ----------------------------
        car_results = car_model(frame, stream=True)
        detected_cars = []  # keep bounding boxes for occupancy checking
        # counters
        counts = {
            "Sedan": 0,
            "Compact": 0,
            "SUV/Truck": 0,
            "Van": 0,
            "Bike": 0,
            "Unknown": 0
        }

        # iterate detection results
        for r in car_results:
            # r.boxes: iterable of boxes
            for box in r.boxes:
                # safe extraction of class and conf (ultralytics format)
                try:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                except Exception:
                    # skip malformed box
                    continue

                label = car_model.names.get(cls_id, str(cls_id)).lower()

                # only consider relevant labels
                if label not in ['car', 'truck', 'bus', 'motorbike', 'bicycle', 'motorcycle', 'van']:
                    continue

                # bounding box coords
                xy = box.xyxy[0]
                x1, y1, x2, y2 = map(int, xy)
                # keep within frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame_w - 1, x2), min(frame_h - 1, y2)

                car_type = Group01Processor._classify_vehicle(label, (x1, y1, x2, y2))
                counts.setdefault(car_type, 0)
                counts[car_type] += 1

                # color per type
                color_map = {
                    "Sedan": (0, 255, 0),
                    "Compact": (255, 0, 0),
                    "SUV/Truck": (0, 255, 255),
                    "Van": (200, 120, 0),
                    "Bike": (255, 0, 255),
                    "Unknown": (150, 150, 150)
                }
                color = color_map.get(car_type, (255, 255, 255))

                # draw box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{car_type} ({conf:.2f})",
                            (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

                detected_cars.append((x1, y1, x2, y2))

        # total vehicles detected
        total_detected = sum(counts.get(k, 0) for k in counts)

        # ----------------------------
        # 2. Detect parking areas (segmentation)
        # ----------------------------
        lot_results = lot_model(frame)
        empty_count, occupied_count = 0, 0
        # Mask iteration
        for result in lot_results:
            if getattr(result, 'masks', None) is None:
                continue
            # result.masks.data is typically a tensor-like list/ndarray
            for mask in result.masks.data:
                try:
                    mask_np = mask.cpu().numpy()
                except Exception:
                    # if already numpy
                    mask_np = np.array(mask)

                # mask may be smaller than frame - resize
                mask_resized = cv2.resize(mask_np, (frame_w, frame_h), interpolation=cv2.INTER_NEAREST)
                bin_mask = (mask_resized > 0.5).astype(np.uint8)

                # find bounding box of mask
                ys, xs = np.where(bin_mask == 1)
                if len(xs) == 0 or len(ys) == 0:
                    continue
                x1_m, y1_m, x2_m, y2_m = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

                # check overlap with any detected car
                occupied_flag = False
                for (cx1, cy1, cx2, cy2) in detected_cars:
                    if not (cx2 < x1_m or cx1 > x2_m or cy2 < y1_m or cy1 > y2_m):
                        occupied_flag = True
                        break

                # Draw transparent overlay for slot
                overlay = frame.copy()
                color = (0, 255, 0) if not occupied_flag else (0, 0, 255)
                cv2.rectangle(overlay, (x1_m, y1_m), (x2_m, y2_m), color, -1)
                cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

                if occupied_flag:
                    occupied_count += 1
                else:
                    empty_count += 1

        # ----------------------------
        # 3. Draw small summary legend on frame (left-top)
        # ----------------------------
        cv2.rectangle(frame, (10, 10), (330, 150), (0, 0, 0), -1)
        cv2.putText(frame, f"Occupied: {occupied_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Empty: {empty_count}", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Total vehicles: {total_detected}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Legend: Red=Occupied, Green=Empty", (20, 135),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # ----------------------------
        # 4. Create stats sidebar (chart)
        # ----------------------------
        # Ensure order for consistent display
        ordered_keys = ["SUV/Truck", "Van", "Sedan", "Compact", "Bike", "Unknown"]
        ordered_counts = {k: counts.get(k, 0) for k in ordered_keys}

        sidebar_w = 320
        sidebar_h = frame_h
        chart_img = Group01Processor._draw_stats_chart(ordered_counts, sidebar_w, sidebar_h)

        # Compose final image: frame on left, chart on right
        # Make sure chart height equals frame height
        if chart_img.shape[0] != frame_h:
            chart_img = cv2.resize(chart_img, (sidebar_w, frame_h))

        combined = np.zeros((frame_h, frame_w + sidebar_w, 3), dtype=np.uint8)
        combined[:, :frame_w] = frame
        combined[:, frame_w:frame_w + sidebar_w] = chart_img

        # Optionally draw a separator line
        cv2.line(combined, (frame_w, 0), (frame_w, frame_h), (200, 200, 200), 2)

        return combined